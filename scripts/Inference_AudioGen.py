# ruff: noqa: E402
"""CosyVoice3 audio extension for the unified multimodal inference entry point.

``Inference_Multimodal.py`` is the public entry point. This module provides
the optional stage-3 audio-output path:

1. generate assistant text with the foundation model;
2. collect the generated assistant-token hidden states;
3. feed those hidden states to the CosyVoice3 audio decoder to generate speech tokens.

Waveform reconstruction is optional because it requires the full CosyVoice runtime
dependencies and a prompt/reference voice. The current VeOmni venv used for training
does not necessarily include those runtime dependencies.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# The waveform decoder is an inference-only dependency and is kept outside the
# trimmed Git package. Override this when deploying the package elsewhere.
COSYVOICE_ROOT = os.environ.get("COSYVOICE_REPO_PATH", os.path.join(PROJECT_ROOT, "CosyVoice"))
MATCHA_TTS_ROOT = os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS")
DEFAULT_SPEAKER_PROFILE = os.path.join(COSYVOICE_ROOT, "spk2info.pt")

from scripts.Inference_Multimodal import InferenceConfig, Qwen35WhisperOmniInfer
from veomni.models.seed_omni.decoder.cosyvoice3_audio_model.text_normalization import CosyVoiceTextNormalizer
from veomni.utils import helper


logger = helper.create_logger(__name__)


@dataclass
class AudioGenerationResult:
    text: str
    speech_tokens: list[list[int]]
    speech_text: Optional[str] = None
    speech_token_path: Optional[str] = None
    speech_chunk_manifest_path: Optional[str] = None
    audio_path: Optional[str] = None
    audio_status: str = ""
    speech_stop_token_ids: Optional[list[Optional[int]]] = None
    speech_stopped_by_eos: Optional[list[bool]] = None
    speech_hit_max_length: Optional[list[bool]] = None


@dataclass
class _SpeechChunk:
    index: int
    text: str
    text_unit_count: int
    token_start: int
    token_end: int
    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    speech_text: Optional[str] = None


class Qwen35CosyVoice3OmniInfer(Qwen35WhisperOmniInfer):
    _text_unit_pattern = re.compile(
        r"[\u3400-\u9fff]|[A-Za-z0-9\u00c0-\u024f]+(?:['-][A-Za-z0-9\u00c0-\u024f]+)*"
    )
    _sentence_end_pattern = re.compile(r"[。！？!?；;][\"'”’）】》]*\s*$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cosyvoice_runtime = None
        self._cosyvoice_runtime_model_dir = None

    @staticmethod
    def _load_sft_waveform_condition(
        speaker_profile: str,
        speaker_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        profiles = torch.load(speaker_profile, map_location="cpu", weights_only=True)
        if speaker_id not in profiles:
            raise KeyError(f"Speaker {speaker_id!r} not found; available={sorted(profiles)}")
        profile = profiles[speaker_id]
        if set(profile) != {"embedding"}:
            raise ValueError(
                f"SFT speaker profile must contain only an `embedding` field; got fields={sorted(profile)}"
            )
        embedding = profile["embedding"].detach().cpu().float()
        if embedding.shape != (1, 192) or not torch.isfinite(embedding).all():
            raise ValueError(f"SFT speaker embedding must be finite with shape (1, 192), got {tuple(embedding.shape)}")
        prompt_token = torch.empty((1, 0), dtype=torch.int32)
        prompt_feat = torch.empty((1, 0, 80), dtype=torch.float32)
        return prompt_token, prompt_feat, embedding

    def _normalize_speech_text(self, text: str) -> str:
        normalizer = getattr(self, "_speech_text_normalizer", None)
        if normalizer is None:
            normalizer = CosyVoiceTextNormalizer(COSYVOICE_ROOT)
            self._speech_text_normalizer = normalizer
        speech_text = normalizer.normalize(text)
        if self._debug_enabled():
            logger.info(f"[INFER_DEBUG] speech_text={speech_text!r}")
        return speech_text

    def _get_audio_decoder(self):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        decoder = getattr(self.model, "decoder", None)
        audio_decoder = getattr(decoder, "audio_decoder", None)
        if audio_decoder is None:
            raise RuntimeError(
                "The loaded model does not contain decoder.audio_decoder. "
                "Use a stage-3 CosyVoice3 checkpoint exported with the audio decoder."
            )
        return audio_decoder

    def _generate_text_with_hidden_states(
        self,
        prompt: str = "",
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        use_audio_in_video: Optional[bool] = None,
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """Generate text and collect the hidden state at each generated-token position."""
        inputs = self._prepare_inputs(
            prompt=prompt,
            image_path=image_path,
            video_path=video_path,
            audio_path=audio_path,
            use_audio_in_video=use_audio_in_video,
        )
        restore_lm_encode = self._install_audio_debug_wrapper()
        tokenizer = self.processor.tokenizer
        eos_token_id = tokenizer.eos_token_id
        eos_token_ids = set(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else {eos_token_id}
        eos_token_ids.discard(None)

        generated_ids: list[int] = []
        generated_hidden_states: list[torch.Tensor] = []

        def model_forward():
            step_inputs = dict(inputs)
            self._add_varlen_attention_kwargs(step_inputs)
            return self.model(
                **step_inputs,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
                logits_to_keep=1,
            )

        try:
            with torch.no_grad():
                for step in range(max_new_tokens):
                    outputs = model_forward()
                    if generated_ids:
                        generated_hidden_states.append(outputs.hidden_states[-1][0, -1].detach())

                    next_logits = outputs.logits[0, -1].float()
                    next_logits = self._apply_repetition_penalty(next_logits, inputs["input_ids"], repetition_penalty)
                    if step < min_new_tokens:
                        for token_id in eos_token_ids:
                            next_logits[token_id] = -float("inf")

                    next_token = self._sample_next_token(next_logits, temperature, top_p)
                    next_token_id = int(next_token.item())
                    self._debug_generation_step(tokenizer, step, next_logits, next_token_id, eos_token_ids)
                    if next_token_id in eos_token_ids:
                        if self._debug_enabled():
                            logger.info(
                                f"[INFER_DEBUG] generation stopped by EOS at step={step}; "
                                f"generated_token_count={len(generated_ids)}"
                            )
                        break

                    generated_ids.append(next_token_id)
                    self._append_generated_token(inputs, next_token)

                if len(generated_hidden_states) < len(generated_ids):
                    outputs = model_forward()
                    generated_hidden_states.append(outputs.hidden_states[-1][0, -1].detach())
        finally:
            if restore_lm_encode is not None:
                self.model.encoder.audio_encoder.lm_encode = restore_lm_encode

        text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if generated_hidden_states:
            hidden_states = torch.stack(generated_hidden_states, dim=0).unsqueeze(0)
        else:
            audio_decoder = self._get_audio_decoder()
            hidden_size = getattr(audio_decoder.config, "output_size", None) or self.model.config.hidden_size
            hidden_states = torch.zeros(1, 0, hidden_size, device=self.device, dtype=self.compute_dtype)
        generated_token_ids = torch.tensor(generated_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return text, generated_token_ids, hidden_states

    @torch.no_grad()
    def generate_speech_tokens(
        self,
        hidden_states: torch.Tensor,
        answer_token_ids: torch.Tensor,
        speech_text: Optional[str] = None,
        sampling: int = 25,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
        max_speech_tokens: Optional[int] = None,
        return_generation_info: bool = False,
        bypass_text_condition_bridge: bool = False,
    ) -> list:
        audio_decoder = self._get_audio_decoder()
        audio_decoder.eval()
        if hidden_states.dim() != 3 or answer_token_ids.dim() != 2:
            raise ValueError(
                "CosyVoice expects hidden_states [batch, tokens, hidden] and "
                f"answer_token_ids [batch, tokens], got {tuple(hidden_states.shape)} and {tuple(answer_token_ids.shape)}"
            )
        if hidden_states.shape[:2] != answer_token_ids.shape:
            raise ValueError(
                "CosyVoice LLM condition/token alignment mismatch: "
                f"hidden_states={tuple(hidden_states.shape)}, answer_token_ids={tuple(answer_token_ids.shape)}"
            )
        if not torch.isfinite(hidden_states).all():
            raise ValueError("CosyVoice LLM condition contains NaN/Inf.")
        hidden_states = hidden_states.to(device=self.device, dtype=self.compute_dtype)
        answer_text_mask = torch.ones(
            hidden_states.shape[:2],
            device=hidden_states.device,
            dtype=torch.bool,
        )
        answer_text_len = answer_text_mask.long().sum(dim=1)
        cosy_text_token = None
        if speech_text is not None:
            cosy_token_ids = audio_decoder.cosy_text_tokenizer.encode(
                speech_text,
                add_special_tokens=False,
            )
            if not cosy_token_ids:
                raise ValueError("Normalized speech text produced an empty CosyVoice token sequence.")
            cosy_text_token = torch.tensor(
                cosy_token_ids,
                dtype=torch.long,
                device=self.device,
            )
        return audio_decoder.lm_generate(
            hidden_states=hidden_states,
            answer_text_mask=answer_text_mask,
            answer_text_len=answer_text_len,
            answer_token_ids=answer_token_ids.to(device=self.device, dtype=torch.long),
            cosy_text_token=cosy_text_token,
            sampling=sampling,
            min_token_text_ratio=min_token_text_ratio,
            max_token_text_ratio=max_token_text_ratio,
            max_speech_tokens=max_speech_tokens,
            return_generation_info=return_generation_info,
            bypass_text_condition_bridge=bypass_text_condition_bridge,
        )

    @staticmethod
    def _save_speech_tokens(speech_tokens: list[torch.Tensor], output_path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        serializable = [tokens.detach().cpu().long().tolist() for tokens in speech_tokens]
        if output_path.endswith(".pt"):
            torch.save(serializable, output_path)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
                f.write("\n")
        return output_path

    @classmethod
    def _count_text_units(cls, text: str) -> int:
        """Count CJK characters individually and Latin text by words."""
        return len(cls._text_unit_pattern.findall(text))

    def _decode_answer_tokens(self, token_ids: torch.Tensor) -> str:
        return self.processor.tokenizer.decode(
            token_ids.detach().cpu().long().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _split_speech_chunks(
        self,
        generated_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        chunk_size: int,
    ) -> list[_SpeechChunk]:
        if generated_token_ids.dim() != 2 or generated_token_ids.shape[0] != 1:
            raise ValueError(
                "Chunked speech inference currently requires one generated answer; "
                f"got token shape {tuple(generated_token_ids.shape)}."
            )
        if hidden_states.dim() != 3 or hidden_states.shape[:2] != generated_token_ids.shape:
            raise ValueError(
                "Generated token IDs and hidden states must have matching batch/sequence dimensions: "
                f"{tuple(generated_token_ids.shape)} vs {tuple(hidden_states.shape)}."
            )

        token_count = generated_token_ids.shape[1]
        if token_count == 0:
            return []
        if chunk_size <= 0:
            text = self._decode_answer_tokens(generated_token_ids[0])
            return [
                _SpeechChunk(
                    index=0,
                    text=text,
                    text_unit_count=self._count_text_units(text),
                    token_start=0,
                    token_end=token_count,
                    token_ids=generated_token_ids,
                    hidden_states=hidden_states,
                )
            ]

        chunks = []
        start = 0
        minimum_boundary_units = max(1, int(chunk_size * 0.6))
        maximum_units = max(chunk_size, int(chunk_size * 1.25))
        while start < token_count:
            split = token_count
            last_sentence_boundary = None
            end = start
            while end < token_count:
                end += 1
                candidate_text = self._decode_answer_tokens(generated_token_ids[0, start:end])
                unit_count = self._count_text_units(candidate_text)
                if unit_count >= minimum_boundary_units and self._sentence_end_pattern.search(candidate_text):
                    last_sentence_boundary = end
                if unit_count < chunk_size:
                    continue
                if last_sentence_boundary is not None:
                    split = last_sentence_boundary
                    break
                if unit_count >= maximum_units or end == token_count:
                    split = end
                    break

            if split <= start:
                split = min(start + 1, token_count)
            chunk_text = self._decode_answer_tokens(generated_token_ids[0, start:split])
            if not chunk_text:
                raise ValueError(f"Speech chunk {len(chunks) + 1} decoded to empty text.")
            chunks.append(
                _SpeechChunk(
                    index=len(chunks),
                    text=chunk_text,
                    text_unit_count=self._count_text_units(chunk_text),
                    token_start=start,
                    token_end=split,
                    token_ids=generated_token_ids[:, start:split],
                    hidden_states=hidden_states[:, start:split],
                )
            )
            start = split
        return chunks

    @staticmethod
    def _validate_speech_chunks(generated_token_ids, hidden_states, chunks):
        """Verify that each CosyVoice chunk is the exact LLM output slice."""
        if generated_token_ids.dim() != 2 or hidden_states.dim() != 3:
            raise RuntimeError("LLM output must be [batch, tokens] and hidden states [batch, tokens, hidden].")
        if generated_token_ids.shape[0] != 1 or hidden_states.shape[:2] != generated_token_ids.shape:
            raise RuntimeError(
                f"LLM token/hidden shape mismatch: tokens={tuple(generated_token_ids.shape)}, "
                f"hidden={tuple(hidden_states.shape)}"
            )
        if not torch.isfinite(hidden_states).all():
            raise RuntimeError("LLM hidden states contain NaN/Inf before CosyVoice chunking.")
        expected_start = 0
        for chunk in chunks:
            if chunk.token_start != expected_start or chunk.token_end <= chunk.token_start:
                raise RuntimeError(
                    f"Invalid chunk {chunk.index} range [{chunk.token_start}, {chunk.token_end}), "
                    f"expected start {expected_start}."
                )
            expected_tokens = generated_token_ids[:, chunk.token_start:chunk.token_end]
            expected_hidden = hidden_states[:, chunk.token_start:chunk.token_end]
            if not torch.equal(chunk.token_ids, expected_tokens):
                raise RuntimeError(f"Chunk {chunk.index} token IDs are not the original LLM slice.")
            if not torch.equal(chunk.hidden_states, expected_hidden):
                raise RuntimeError(f"Chunk {chunk.index} hidden states are not the original LLM slice.")
            if chunk.hidden_states.shape[:2] != chunk.token_ids.shape:
                raise RuntimeError(
                    f"Chunk {chunk.index} token/hidden shape mismatch: "
                    f"tokens={tuple(chunk.token_ids.shape)}, hidden={tuple(chunk.hidden_states.shape)}"
                )
            logger.info(
                f"CosyVoice chunk {chunk.index + 1}/{len(chunks)} LLM condition verified: "
                f"tokens=[{chunk.token_start}:{chunk.token_end}] hidden={tuple(chunk.hidden_states.shape)}"
            )
            expected_start = chunk.token_end
        if expected_start != generated_token_ids.shape[1]:
            raise RuntimeError(
                f"Chunk ranges cover {expected_start} LLM tokens, "
                f"but generated output has {generated_token_ids.shape[1]}."
            )

    @staticmethod
    def _save_speech_chunk_manifest(
        chunks: list[_SpeechChunk],
        speech_generations: list,
        output_path: str,
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        manifest = []
        for chunk, generation in zip(chunks, speech_generations, strict=True):
            manifest.append(
                {
                    "index": chunk.index,
                    "text": chunk.text,
                    "speech_text": chunk.speech_text,
                    "text_unit_count": chunk.text_unit_count,
                    "foundation_token_start": chunk.token_start,
                    "foundation_token_end": chunk.token_end,
                    "foundation_token_count": chunk.token_end - chunk.token_start,
                    "speech_token_count": int(generation.token_ids.numel()),
                    "stop_token_id": generation.stop_token_id,
                    "stopped_by_eos": generation.stopped_by_eos,
                    "hit_max_length": generation.hit_max_length,
                }
            )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return output_path

    def _try_synthesize_wav(
        self,
        speech_tokens: list[torch.Tensor],
        output_audio_path: Optional[str],
        prompt_wav: Optional[str],
        prompt_text: str = "",
        speaker_profile: Optional[str] = None,
        speaker_id: Optional[str] = None,
        speed: float = 1.0,
        chunk_silence_ms: int = 200,
    ) -> tuple[Optional[str], str]:
        if not output_audio_path:
            return None, "Audio waveform synthesis skipped because --output-audio was not provided."
        if prompt_wav and speaker_profile:
            return None, (
                "Audio waveform synthesis skipped because --prompt-wav and --speaker-profile are mutually exclusive."
            )
        if speaker_profile and not speaker_id:
            return None, "Audio waveform synthesis skipped because --speaker-id is required with --speaker-profile."
        if speaker_profile and not os.path.isfile(speaker_profile):
            return None, (
                "Audio waveform synthesis skipped because the speaker profile does not exist: "
                f"{speaker_profile!r}. Pass --speaker-profile with a valid embedding-only spk2info.pt, "
                "or use --prompt-wav/--prompt-text for zero-shot conditioning."
            )
        if not prompt_wav and not speaker_profile:
            return None, (
                "Audio waveform synthesis skipped because CosyVoice3 token2wav needs a voice condition. "
                "Pass either --speaker-profile/--speaker-id for SFT conditioning or --prompt-wav for zero-shot conditioning."
            )
        if chunk_silence_ms < 0:
            return None, "Audio waveform synthesis skipped because chunk silence must be non-negative."
        if not speech_tokens:
            return None, "Audio waveform synthesis skipped because no speech-token chunks were generated."

        missing = []
        for module_name in ("onnxruntime", "whisper", "hyperpyyaml"):
            try:
                __import__(module_name)
            except ImportError:
                missing.append(module_name)
        if missing:
            return None, (
                "Audio waveform synthesis skipped because the current venv is missing CosyVoice runtime "
                f"dependencies: {', '.join(missing)}. Speech tokens were generated successfully."
            )

        try:
            if os.path.isdir(COSYVOICE_ROOT) and COSYVOICE_ROOT not in sys.path:
                sys.path.insert(0, COSYVOICE_ROOT)
            if os.path.isdir(MATCHA_TTS_ROOT) and MATCHA_TTS_ROOT not in sys.path:
                sys.path.insert(0, MATCHA_TTS_ROOT)

            # Some CosyVoice checkouts patch wetext during import, while newer
            # wetext versions no longer expose snapshot_download themselves.
            import wetext.wetext as wetext_impl
            if not hasattr(wetext_impl, "snapshot_download"):
                from modelscope import snapshot_download as modelscope_snapshot_download

                wetext_impl.snapshot_download = modelscope_snapshot_download
            import torchaudio
            from cosyvoice.cli.cosyvoice import CosyVoice3
        except Exception as exc:
            return None, f"Audio waveform synthesis skipped; failed to import CosyVoice3 runtime: {exc!r}"

        try:
            model_dir = self._get_audio_decoder().config.cosyvoice3_path
            if not os.path.isdir(model_dir):
                return None, (
                    f"Audio waveform synthesis skipped because local CosyVoice3 model directory does not exist: "
                    f"{model_dir!r}. Set --model-path to a checkpoint whose audio decoder config points to a local "
                    "CosyVoice3 directory, or update cosyvoice3_path in config.json."
                )
            if self._cosyvoice_runtime is None or self._cosyvoice_runtime_model_dir != model_dir:
                self._cosyvoice_runtime = CosyVoice3(model_dir=model_dir)
                self._cosyvoice_runtime_model_dir = model_dir
            cosyvoice = self._cosyvoice_runtime
            if speaker_profile:
                prompt_token, prompt_feat, embedding = self._load_sft_waveform_condition(
                    speaker_profile,
                    speaker_id,
                )
            else:
                frontend_input = cosyvoice.frontend.frontend_zero_shot(
                    "",
                    prompt_text,
                    prompt_wav,
                    cosyvoice.sample_rate,
                    "",
                )
                prompt_token = frontend_input["flow_prompt_speech_token"]
                prompt_feat = frontend_input["prompt_speech_feat"]
                embedding = frontend_input["flow_embedding"]
            wav_segments = []
            silence_samples = 0
            stream_uuid = f"veomni-cosyvoice3-infer-{id(self)}"
            cosyvoice.model.hift_cache_dict[stream_uuid] = None
            cumulative_tokens = []
            if speed != 1.0:
                logger.warning("Segmented CosyVoice decoding uses native speed to preserve cross-segment continuity.")
            for index, speech_token_chunk in enumerate(speech_tokens):
                cumulative_tokens.extend(speech_token_chunk.detach().cpu().to(torch.int32).tolist())
                token = torch.tensor(cumulative_tokens, dtype=torch.int32).unsqueeze(0)
                is_final = index == len(speech_tokens) - 1
                try:
                    wav_chunk = cosyvoice.model.token2wav(
                        token=token,
                        prompt_token=prompt_token,
                        prompt_feat=prompt_feat,
                        embedding=embedding,
                        token_offset=len(cumulative_tokens) - len(speech_token_chunk),
                        uuid=stream_uuid,
                        stream=True,
                        finalize=is_final,
                        # A cached streaming decode cannot apply speed on the
                        # final call; keep all chunks at native speed so the
                        # acoustic state remains continuous.
                        speed=1.0,
                    ).cpu()
                finally:
                    if is_final:
                        cosyvoice.model.hift_cache_dict.pop(stream_uuid, None)
                if wav_segments and silence_samples:
                    wav_segments.append(torch.zeros(wav_chunk.shape[0], silence_samples, dtype=wav_chunk.dtype))
                wav_segments.append(wav_chunk)
            wav = torch.cat(wav_segments, dim=-1)
            os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)) or ".", exist_ok=True)
            torchaudio.save(output_audio_path, wav, cosyvoice.sample_rate)
            return output_audio_path, ""
        except Exception as exc:
            if "stream_uuid" in locals():
                cosyvoice.model.hift_cache_dict.pop(stream_uuid, None)
            return None, f"Audio waveform synthesis failed: {exc!r}. Speech tokens were generated successfully."

    def generate_with_audio(
        self,
        prompt: str = "",
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        use_audio_in_video: Optional[bool] = None,
        enable_audio_output: bool = False,
        speech_token_output: Optional[str] = None,
        speech_chunk_manifest: Optional[str] = None,
        output_audio: Optional[str] = None,
        prompt_wav: Optional[str] = None,
        prompt_text: str = "",
        speaker_profile: Optional[str] = None,
        speaker_id: Optional[str] = None,
        speech_sampling: int = 25,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
        max_speech_tokens: Optional[int] = None,
        speech_chunk_size: int = 100,
        speech_chunk_silence_ms: int = 200,
        audio_speed: float = 1.0,
        bypass_text_condition_bridge: bool = False,
    ) -> AudioGenerationResult:
        text, generated_token_ids, hidden_states = self._generate_text_with_hidden_states(
            prompt=prompt,
            image_path=image_path,
            video_path=video_path,
            audio_path=audio_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            use_audio_in_video=use_audio_in_video,
        )

        # An output path is also an explicit request to run speech generation.
        if not enable_audio_output and speech_token_output is None and output_audio is None:
            return AudioGenerationResult(text=text, speech_tokens=[])

        if hidden_states.shape[1] == 0:
            return AudioGenerationResult(
                text=text,
                speech_tokens=[],
                audio_status="Audio generation skipped because no text tokens were generated.",
            )

        try:
            chunks = self._split_speech_chunks(generated_token_ids, hidden_states, speech_chunk_size)
            self._validate_speech_chunks(generated_token_ids, hidden_states, chunks)
            for chunk in chunks:
                chunk.speech_text = self._normalize_speech_text(chunk.text)
        except ValueError as exc:
            return AudioGenerationResult(
                text=text,
                speech_tokens=[],
                audio_status=f"Audio generation skipped: {exc}",
            )

        # Keep the original segmented speech-token generation contract: each
        # text chunk is submitted independently to the CosyVoice LM.  Waveform
        # synthesis below still shares the HIFT stream state across chunks.
        speech_generations = []
        for chunk in chunks:
            logger.info(
                f"Generating speech chunk {chunk.index + 1}/{len(chunks)}: "
                f"text_units={chunk.text_unit_count}, foundation_tokens={chunk.token_end - chunk.token_start}"
            )
            generated = self.generate_speech_tokens(
                hidden_states=chunk.hidden_states, answer_token_ids=chunk.token_ids,
                speech_text=chunk.speech_text, sampling=speech_sampling,
                min_token_text_ratio=min_token_text_ratio, max_token_text_ratio=max_token_text_ratio,
                max_speech_tokens=max_speech_tokens, return_generation_info=True,
                bypass_text_condition_bridge=bypass_text_condition_bridge,
            )
            if len(generated) != 1:
                raise RuntimeError(f"Speech chunk {chunk.index + 1} produced {len(generated)} items; expected one.")
            speech_generations.append(generated[0])
        speech_token_tensors = [item.token_ids for item in speech_generations]
        speech_token_path = (
            self._save_speech_tokens(speech_token_tensors, speech_token_output) if speech_token_output else None
        )
        if speech_chunk_manifest is None and speech_token_output and len(chunks) > 1:
            speech_chunk_manifest = os.path.splitext(speech_token_output)[0] + "_chunks.json"
        speech_chunk_manifest_path = (
            self._save_speech_chunk_manifest(chunks, speech_generations, speech_chunk_manifest)
            if speech_chunk_manifest
            else None
        )
        speech_text = "\n".join(chunk.speech_text for chunk in chunks)
        stop_token_ids = [item.stop_token_id for item in speech_generations]
        stopped_by_eos = [item.stopped_by_eos for item in speech_generations]
        hit_max_length = [item.hit_max_length for item in speech_generations]

        failed_items = [
            index
            for index, item in enumerate(speech_generations)
            if not item.stopped_by_eos or item.hit_max_length or item.token_ids.numel() == 0
        ]
        if failed_items:
            reasons = []
            for index in failed_items:
                item = speech_generations[index]
                if item.hit_max_length:
                    reason = "reached max length without EOS"
                elif not item.stopped_by_eos:
                    reason = f"stopped on non-EOS special token {item.stop_token_id}"
                else:
                    reason = "generated no speech tokens"
                reasons.append(f"chunk {index + 1}: {reason}")
            return AudioGenerationResult(
                text=text,
                speech_tokens=[item.detach().cpu().long().tolist() for item in speech_token_tensors],
                speech_text=speech_text,
                speech_token_path=speech_token_path,
                speech_chunk_manifest_path=speech_chunk_manifest_path,
                audio_status="Audio waveform synthesis skipped because speech generation failed: "
                + "; ".join(reasons),
                speech_stop_token_ids=stop_token_ids,
                speech_stopped_by_eos=stopped_by_eos,
                speech_hit_max_length=hit_max_length,
            )

        audio_path_out, audio_status = self._try_synthesize_wav(
            speech_token_tensors,
            output_audio_path=output_audio,
            prompt_wav=prompt_wav,
            prompt_text=prompt_text,
            speaker_profile=speaker_profile,
            speaker_id=speaker_id,
            speed=audio_speed,
            chunk_silence_ms=speech_chunk_silence_ms,
        )
        return AudioGenerationResult(
            text=text,
            speech_tokens=[item.detach().cpu().long().tolist() for item in speech_token_tensors],
            speech_text=speech_text,
            speech_token_path=speech_token_path,
            speech_chunk_manifest_path=speech_chunk_manifest_path,
            audio_path=audio_path_out,
            audio_status=audio_status,
            speech_stop_token_ids=stop_token_ids,
            speech_stopped_by_eos=stopped_by_eos,
            speech_hit_max_length=hit_max_length,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Qwen3.5 + Whisper + CosyVoice3 SeedOmni inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--load-mode", default=os.environ.get("LOAD_MODE", "full"))
    parser.add_argument("--processor-path", default=os.environ.get("PROCESSOR_PATH"))
    parser.add_argument("--chat-template", default=os.environ.get("CHAT_TEMPLATE", "qwen3_5omni"))
    parser.add_argument(
        "--thinking-prompt",
        choices=("disabled", "enabled", "none"),
        default=os.environ.get("THINKING_PROMPT", "none"),
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--max-input-frames", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=602112)
    parser.add_argument("--video-max-pixels", type=int, default=602112)
    audio_video_group = parser.add_mutually_exclusive_group()
    audio_video_group.add_argument(
        "--audio-in-video",
        dest="audio_in_video",
        action="store_true",
        help="Include the video's audio track in multimodal encoding (experimental).",
    )
    audio_video_group.add_argument(
        "--no-audio-in-video",
        dest="audio_in_video",
        action="store_false",
        help="Ignore the video's audio track during encoding (default).",
    )
    parser.set_defaults(audio_in_video=False)
    parser.add_argument("--enable-audio-output", "--generate-audio", action="store_true")
    parser.add_argument(
        "--speech-token-output", default=None, help="Optional .json or .pt path for generated speech tokens."
    )
    parser.add_argument(
        "--speech-chunk-manifest",
        default=None,
        help="Optional JSON path for per-chunk text, token ranges, token counts, and stop status.",
    )
    parser.add_argument(
        "--output-audio", default=None, help="Optional wav output path. Requires a voice condition and CosyVoice deps."
    )
    parser.add_argument("--prompt-wav", default=None, help="Reference voice wav for CosyVoice3 token2wav.")
    parser.add_argument(
        "--prompt-text", default="", help="Transcript of --prompt-wav for CosyVoice3 zero-shot voice prompt."
    )
    parser.add_argument(
        "--speaker-profile",
        default=DEFAULT_SPEAKER_PROFILE,
        help=f"Embedding-only spk2info.pt for inference_sft-compatible waveform conditioning (default: {DEFAULT_SPEAKER_PROFILE}).",
    )
    parser.add_argument("--speaker-id", default="fysics", help="Speaker ID in --speaker-profile (default: fysics).")
    parser.add_argument("--speech-sampling", type=int, default=25)
    parser.add_argument("--min-token-text-ratio", type=float, default=2.0)
    parser.add_argument("--max-token-text-ratio", type=float, default=20.0)
    parser.add_argument(
        "--max-speech-tokens",
        type=int,
        default=None,
        help="Maximum generated speech tokens per chunk. Defaults to the configured text-token ratio.",
    )
    parser.add_argument(
        "--speech-chunk-size",
        type=int,
        default=100,
        help="Target natural-language units per speech chunk; CJK characters and Latin words count as one. Use 0 to disable.",
    )
    parser.add_argument(
        "--speech-chunk-silence-ms",
        type=int,
        default=200,
        help="Legacy gap setting; streaming chunk synthesis keeps segments continuous and does not insert a gap.",
    )
    parser.add_argument(
        "--bypass-text-condition-bridge",
        action="store_true",
        help="Diagnostic mode: use native CosyVoice text embeddings without projector, gate, or cross-attention.",
    )
    parser.add_argument("--audio-speed", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true", help="Print input/audio encoding diagnostics.")
    parser.add_argument("--debug-generation-steps", type=int, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.debug:
        os.environ["INFER_DEBUG"] = "1"
    if args.debug_generation_steps is not None:
        os.environ["INFER_DEBUG_GENERATION_STEPS"] = str(args.debug_generation_steps)

    config = InferenceConfig(
        model_path=args.model_path,
        load_mode=args.load_mode,
        processor_path=args.processor_path,
        chat_template=args.chat_template,
        thinking_prompt=args.thinking_prompt,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        fps=args.fps,
        max_frames=args.max_frames,
        max_input_frames=args.max_input_frames,
        image_max_pixels=args.image_max_pixels,
        video_max_pixels=args.video_max_pixels,
        use_audio_in_video=args.audio_in_video,
    )
    infer = Qwen35CosyVoice3OmniInfer(config)
    result = infer.generate_with_audio(
        prompt=args.prompt,
        image_path=args.image,
        video_path=args.video,
        audio_path=args.audio,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        min_new_tokens=args.min_new_tokens,
        enable_audio_output=args.enable_audio_output,
        speech_token_output=args.speech_token_output,
        speech_chunk_manifest=args.speech_chunk_manifest,
        output_audio=args.output_audio,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        speaker_profile=args.speaker_profile,
        speaker_id=args.speaker_id,
        speech_sampling=args.speech_sampling,
        min_token_text_ratio=args.min_token_text_ratio,
        max_token_text_ratio=args.max_token_text_ratio,
        max_speech_tokens=args.max_speech_tokens,
        speech_chunk_size=args.speech_chunk_size,
        speech_chunk_silence_ms=args.speech_chunk_silence_ms,
        audio_speed=args.audio_speed,
        bypass_text_condition_bridge=args.bypass_text_condition_bridge,
    )
    print(result.text)
    if result.speech_text is not None and result.speech_text != result.text:
        print(f"[SPEECH_TEXT] {result.speech_text}")
    if result.speech_tokens:
        speech_token_counts = [len(tokens) for tokens in result.speech_tokens]
        print(
            f"[SPEECH_TOKENS] generated {sum(speech_token_counts)} tokens across "
            f"{len(speech_token_counts)} chunks; per_chunk={speech_token_counts}"
        )
    if result.speech_stopped_by_eos is not None:
        print(
            "[SPEECH_STATUS] "
            f"stop_token_ids={result.speech_stop_token_ids}, "
            f"stopped_by_eos={result.speech_stopped_by_eos}, "
            f"hit_max_length={result.speech_hit_max_length}"
        )
    if result.speech_token_path:
        print(f"[SPEECH_TOKEN_PATH] {result.speech_token_path}")
    if result.speech_chunk_manifest_path:
        print(f"[SPEECH_CHUNK_MANIFEST] {result.speech_chunk_manifest_path}")
    if result.audio_path:
        print(f"[AUDIO] {result.audio_path}")
    if result.audio_status:
        print(f"[AUDIO_STATUS] {result.audio_status}")


if __name__ == "__main__":
    main()
