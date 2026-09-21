# ruff: noqa: E402
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
from safetensors import safe_open


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from veomni.data import build_multimodal_chat_template
from veomni.data.multimodal.multimodal_transform import encode_multimodal_sample_inference
from veomni.models import build_foundation_model, build_processor
from veomni.models.seed_omni import SeedOmniModel, SeedOmniProcessor
from veomni.utils import helper
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids


logger = helper.create_logger(__name__)


DEFAULT_MODEL_PATH = "hf_ckpt"


@dataclass
class InferenceConfig:
    model_path: str = DEFAULT_MODEL_PATH
    load_mode: str = "full"
    processor_path: Optional[str] = None
    chat_template: str = "qwen3_5omni"
    # enable_thinking: bool = False
    thinking_prompt: str = "none"
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    sample_rate: int = 16000
    fps: float = 2.0
    max_frames: int = 128
    max_input_frames: int = 128
    image_max_pixels: int = 602112
    video_max_pixels: int = 602112
    use_audio_in_video: bool = True

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


class CosyVoice3AudioBackend:
    """Placeholder for text-to-speech/audio generation integration."""

    sample_rate = 24000

    def synthesize(self, text: str, **kwargs) -> str:
        raise NotImplementedError("CosyVoice3 audio output is not integrated yet.")


class Qwen35WhisperOmniInfer:
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.model: Optional[SeedOmniModel] = None
        self.processor: Optional[SeedOmniProcessor] = None
        self.chat_template = None
        self.position_id_func = None
        self.modality_info = None
        self.audio_backend = CosyVoice3AudioBackend()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    @property
    def device(self) -> str:
        return self.config.resolved_device()

    @property
    def compute_dtype(self):
        if self.model is not None:
            return next(self.model.parameters()).dtype
        if self.config.torch_dtype == "bfloat16":
            return torch.bfloat16
        if self.config.torch_dtype == "float16":
            return torch.float16
        return torch.float32

    @staticmethod
    def _debug_enabled() -> bool:
        return os.environ.get("INFER_DEBUG", "").lower() not in {"", "0", "false", "no", "off"}

    @staticmethod
    def _debug_generation_limit() -> int:
        try:
            return max(0, int(os.environ.get("INFER_DEBUG_GENERATION_STEPS", "8")))
        except ValueError:
            return 8

    @staticmethod
    def _debug_generation_top_k() -> int:
        try:
            return max(1, int(os.environ.get("INFER_DEBUG_TOP_K", "5")))
        except ValueError:
            return 5

    @staticmethod
    def _summarize_tensor(value: torch.Tensor) -> str:
        summary = f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
        if value.numel() == 0:
            return f"{summary} numel=0"
        if value.dtype == torch.bool:
            return f"{summary} true={int(value.sum().item())}"
        if value.numel() <= 16 and not torch.is_floating_point(value):
            return f"{summary} values={value.detach().cpu().reshape(-1).tolist()}"
        if torch.is_floating_point(value):
            finite = value.detach().float()
            return (
                f"{summary} mean={float(finite.mean().item()):.6g} "
                f"std={float(finite.std().item()):.6g}"
            )
        return summary

    def _debug_media_paths(
        self,
        image_path: Optional[str],
        video_path: Optional[str],
        audio_path: Optional[str],
    ) -> None:
        if not self._debug_enabled():
            return
        for name, path in (("image", image_path), ("video", video_path), ("audio", audio_path)):
            if not path:
                logger.info(f"[INFER_DEBUG] {name}_path=None")
                continue
            exists = os.path.exists(path) if isinstance(path, str) else False
            size = os.path.getsize(path) if exists and os.path.isfile(path) else None
            logger.info(f"[INFER_DEBUG] {name}_path={path!r} exists={exists} size={size}")

    def _debug_model_inputs(self, inputs: dict, stage: str) -> None:
        if not self._debug_enabled():
            return
        keys = sorted(inputs.keys())
        audio_keys = [key for key in keys if key.startswith("audio_")]
        logger.info(f"[INFER_DEBUG] {stage} keys={keys}")
        logger.info(f"[INFER_DEBUG] {stage} audio_keys={audio_keys}")
        for key in (
            "input_ids",
            "attention_mask",
            "position_ids",
            "audio_input_features",
            "audio_input_feature_lengths",
            "audio_input_num_tokens",
            "audio_input_mask",
        ):
            value = inputs.get(key)
            if isinstance(value, torch.Tensor):
                logger.info(f"[INFER_DEBUG] {stage} {key}: {self._summarize_tensor(value)}")
            elif value is not None:
                logger.info(f"[INFER_DEBUG] {stage} {key}: {value!r}")

    def _debug_prompt_tail(self, inputs: dict, stage: str) -> None:
        if not self._debug_enabled() or self.processor is None:
            return

        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return

        ids = input_ids.detach().cpu()
        if ids.dim() == 2:
            ids = ids[0]

        modal_masks = []
        for key, value in sorted(inputs.items()):
            if key == "attention_mask" or not key.endswith("_mask") or not isinstance(value, torch.Tensor):
                continue
            mask = value.detach().cpu()
            if mask.dim() == 2:
                mask = mask[0]
            if mask.shape[-1] == ids.shape[-1]:
                modal_masks.append((key.removesuffix("_mask"), mask.bool()))

        def modal_label_at(position: int) -> Optional[str]:
            for label, mask in modal_masks:
                if bool(mask[position].item()):
                    return label
            return None

        tail_ids = ids[-160:].tolist()
        tail_start = ids.shape[-1] - len(tail_ids)
        rendered_parts = []
        text_buffer = []
        modal_token_counts = {}

        def flush_text_buffer():
            if text_buffer:
                rendered_parts.append(
                    self.processor.tokenizer.decode(
                        text_buffer,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                )
                text_buffer.clear()

        def append_modal(label: str):
            flush_text_buffer()
            modal_token_counts[label] = modal_token_counts.get(label, 0) + 1
            if rendered_parts and isinstance(rendered_parts[-1], tuple) and rendered_parts[-1][0] == label:
                rendered_parts[-1] = (label, rendered_parts[-1][1] + 1)
            else:
                rendered_parts.append((label, 1))

        for offset, token_id in enumerate(tail_ids):
            position = tail_start + offset
            label = modal_label_at(position)
            if label is not None:
                append_modal(label)
                continue

            token_id = int(token_id)
            if token_id >= 0:
                text_buffer.append(token_id)
                continue

            flush_text_buffer()
            modal_token_counts[token_id] = modal_token_counts.get(token_id, 0) + 1
            if not rendered_parts or rendered_parts[-1] != f"<MM:{token_id}>":
                rendered_parts.append(f"<MM:{token_id}>")

        flush_text_buffer()
        rendered = "".join(
            f"<{part[0]}*{part[1]}>" if isinstance(part, tuple) else part for part in rendered_parts
        )
        logger.info(f"[INFER_DEBUG] {stage} prompt_tail={rendered!r}")
        if modal_token_counts:
            logger.info(f"[INFER_DEBUG] {stage} multimodal_token_counts={modal_token_counts}")

    def _debug_generation_step(self, tokenizer, step: int, logits: torch.Tensor, next_token_id: int, eos_token_ids: set):
        if not self._debug_enabled() or step >= self._debug_generation_limit():
            return

        top_k = min(self._debug_generation_top_k(), logits.numel())
        top_values, top_indices = torch.topk(logits, k=top_k)
        top_items = []
        for score, token_id in zip(top_values.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
            token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            top_items.append(f"{int(token_id)}:{token_text!r}:{float(score):.4g}")

        next_text = tokenizer.decode([next_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        logger.info(
            f"[INFER_DEBUG] generation step={step} next_token_id={next_token_id} "
            f"token={next_text!r} eos={next_token_id in eos_token_ids} top{top_k}={top_items}"
        )

    def _install_audio_debug_wrapper(self):
        if not self._debug_enabled() or self.model is None:
            return None
        audio_encoder = getattr(getattr(self.model, "encoder", None), "audio_encoder", None)
        if audio_encoder is None or not hasattr(audio_encoder, "lm_encode"):
            logger.info("[INFER_DEBUG] audio_encoder.lm_encode is not available.")
            return None

        original_lm_encode = audio_encoder.lm_encode
        logged = False

        def wrapped_lm_encode(*args, **kwargs):
            nonlocal logged
            if not logged:
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"[INFER_DEBUG] audio_encoder.lm_encode input {key}: {self._summarize_tensor(value)}")
                    else:
                        logger.info(f"[INFER_DEBUG] audio_encoder.lm_encode input {key}: {value!r}")
            output = original_lm_encode(*args, **kwargs)
            if not logged:
                if isinstance(output, torch.Tensor):
                    logger.info(f"[INFER_DEBUG] audio_encoder.lm_encode output: {self._summarize_tensor(output)}")
                else:
                    logger.info(f"[INFER_DEBUG] audio_encoder.lm_encode output: {output!r}")
                logged = True
            return output

        audio_encoder.lm_encode = wrapped_lm_encode
        return original_lm_encode

    def load(self):
        if self.loaded:
            return self

        if not os.path.isdir(self.config.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.config.model_path}")

        device = self.device
        dtype_name = self.config.torch_dtype
        if device == "cpu" and dtype_name in {"float16", "bfloat16"}:
            logger.warning("CPU inference requested; using float32 for better operator coverage.")
            dtype_name = "float32"

        if self.config.load_mode != "full":
            raise ValueError("This release supports only load_mode='full' for HF checkpoints.")
        self._load_full_model(device=device, dtype_name=dtype_name)

        if self.processor is None:
            processor_path = self._resolve_processor_path()
            logger.info(f"Loading SeedOmni processor from {processor_path}")
            self.processor = build_processor(processor_path)
        self.chat_template = build_multimodal_chat_template(
            self.config.chat_template,
            self.processor.tokenizer,
        )
        self.position_id_func = self.model.get_position_id_func()
        self.modality_info = self.model.get_modality()
        logger.info(f"Model loaded. Modalities: {self.modality_info}")
        return self

    def _resolve_processor_path(self) -> str:
        if self.config.processor_path:
            return self.config.processor_path
        return self.config.model_path

    def _load_full_model(self, device: str, dtype_name: str):
        logger.info(f"Loading full SeedOmni model from {self.config.model_path}")
        self.model = build_foundation_model(
            config_path=self.config.model_path,
            weights_path=self.config.model_path,
            torch_dtype=dtype_name,
            attn_implementation=self.config.attn_implementation,
            init_device=device,
        ).eval()
        # ``build_foundation_model`` loads the main foundation weights, while
        # the audio decoder is normally initialized through its CosyVoice
        # base checkpoint. The stage-3 checkpoint also contains the trained
        # projector and text-condition bridge (and the matching CosyVoice LM),
        # so load that prefixed submodule explicitly from the same checkpoint.
        audio_decoder = getattr(getattr(self.model, "decoder", None), "audio_decoder", None)
        if audio_decoder is not None:
            self._load_prefixed_module_weights(
                audio_decoder,
                self.config.model_path,
                "decoder.audio_decoder.",
            )
        self.model.to(device)

    def _tie_seed_omni_embeddings(self):
        self.model.get_input_embeddings()._parameters["weight"] = (
            self.model.foundation.get_input_embeddings()._parameters["weight"]
        )
        if getattr(self.model.foundation.config, "tie_word_embeddings", True):
            input_embeddings = self.model.get_input_embeddings()
            output_embeddings = self.model.get_output_embeddings()
            output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]

    def _resize_to_processor_vocab(self):
        tokenizer_size = len(self.processor.tokenizer)
        current_size = self.model.get_input_embeddings().num_embeddings
        if tokenizer_size != current_size:
            logger.info(f"Resize token embeddings from {current_size} to {tokenizer_size}.")
            self.model.resize_token_embeddings(tokenizer_size)
            self._tie_seed_omni_embeddings()

    def _adapter_index(self):
        index_path = os.path.join(self.config.model_path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing adapter safetensors index: {index_path}")
        with open(index_path, encoding="utf-8") as f:
            return json.load(f)["weight_map"]

    def _load_adapter_tensor(self, key: str, weight_map: dict[str, str]):
        filename = weight_map.get(key)
        if filename is None:
            raise KeyError(f"Adapter tensor not found: {key}")
        with safe_open(os.path.join(self.config.model_path, filename), framework="pt", device="cpu") as shard:
            return shard.get_tensor(key)

    @staticmethod
    def _load_hf_tensor(model_path: str, key: str, weight_map: dict[str, str]):
        filename = weight_map.get(key)
        if filename is None:
            raise KeyError(key)
        with safe_open(os.path.join(model_path, filename), framework="pt", device="cpu") as shard:
            return shard.get_tensor(key)

    @staticmethod
    def _hf_weight_map(model_path: str):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing safetensors index: {index_path}")
        with open(index_path, encoding="utf-8") as f:
            return json.load(f)["weight_map"]

    def _load_prefixed_module_weights(self, module: torch.nn.Module, model_path: str, prefix: str):
        weight_map = self._hf_weight_map(model_path)
        loaded, missing = 0, []
        if any(param.device.type == "meta" for param in module.parameters()) or any(
            buffer.device.type == "meta" for buffer in module.buffers()
        ):
            module.to_empty(device=self.device)
            init_weights = getattr(module, "_init_weights", None)
            if init_weights is not None:
                module.apply(init_weights)
        tensors = dict(module.named_parameters())
        tensors.update(dict(module.named_buffers()))
        with torch.no_grad():
            for name, target in tensors.items():
                key = f"{prefix}{name}"
                try:
                    source = self._load_hf_tensor(model_path, key, weight_map)
                except KeyError:
                    missing.append(name)
                    continue
                if source.shape != target.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: source={tuple(source.shape)} target={tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded += 1
        missing_to_warn = [name for name in missing if name != "rotary_pos_emb.inv_freq"]
        if missing_to_warn:
            logger.warning(
                f"Missing {len(missing_to_warn)} tensors while loading prefix {prefix}: {missing_to_warn[:8]}"
            )
        logger.info(f"Loaded {loaded} tensors from {model_path} with prefix {prefix}")

    def _load_qwen35_vision_weights(self):
        if hasattr(self.model.encoder, "image_encoder"):
            self._load_prefixed_module_weights(
                self.model.encoder.image_encoder,
                self.config.base_model_path,
                "model.visual.",
            )
        if hasattr(self.model.encoder, "video_encoder"):
            self._load_prefixed_module_weights(
                self.model.encoder.video_encoder,
                self.config.base_model_path,
                "model.visual.",
            )

    def _load_stage1_adapter_weights(self):
        weight_map = self._adapter_index()
        named_params = dict(self.model.named_parameters())
        loaded = []

        with torch.no_grad():
            for name, param in named_params.items():
                if not name.startswith("encoder.audio_encoder.projector."):
                    continue
                tensor = self._load_adapter_tensor(name, weight_map).to(device=param.device, dtype=param.dtype)
                if tensor.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: adapter={tuple(tensor.shape)} model={tuple(param.shape)}"
                    )
                param.copy_(tensor)
                loaded.append(name)
                if self._debug_enabled():
                    logger.info(f"[INFER_DEBUG] loaded {name}: {self._summarize_tensor(param.detach())}")

            embedding_ids = sorted(set(getattr(self.chat_template, "trained_embedding", [])))
            if embedding_ids:
                source_embedding = self._load_adapter_tensor("encoder.text_encoder.weight", weight_map)
                input_embedding = self.model.get_input_embeddings().weight
                output_embedding = self.model.get_output_embeddings().weight
                for token_id in embedding_ids:
                    if token_id >= source_embedding.shape[0] or token_id >= input_embedding.shape[0]:
                        logger.warning(f"Skip adapter embedding row out of range: {token_id}")
                        continue
                    row = source_embedding[token_id].to(device=input_embedding.device, dtype=input_embedding.dtype)
                    input_embedding[token_id].copy_(row)
                    if token_id < output_embedding.shape[0]:
                        output_embedding[token_id].copy_(
                            row.to(device=output_embedding.device, dtype=output_embedding.dtype)
                        )
                    if self._debug_enabled():
                        logger.info(
                            f"[INFER_DEBUG] loaded template embedding token_id={token_id}: "
                            f"{self._summarize_tensor(input_embedding[token_id].detach())}"
                        )
                loaded.append(f"template_embeddings={embedding_ids}")

        logger.info(f"Loaded stage-1 adapter weights: {loaded}")

    def _build_conversations(
        self,
        prompt: str = "",
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ):
        user_turn = ["user"]
        if image_path:
            user_turn.append(("image", None))
        if video_path:
            user_turn.append(("video", None))
        if audio_path:
            user_turn.append(("audio", None))
        if prompt and prompt.strip():
            user_turn.append(("text", prompt.strip()))

        if len(user_turn) == 1:
            raise ValueError("Please provide at least one input: text, image, video, or audio.")

        return [user_turn, ["assistant"]]

    def _processor_kwargs(self, use_audio_in_video: Optional[bool] = None):
        return {
            "sample_rate": self.config.sample_rate,
            "fps": self.config.fps,
            "max_frames": self.config.max_frames,
            "max_input_frames": self.config.max_input_frames,
            "image_max_pixels": self.config.image_max_pixels,
            "video_max_pixels": self.config.video_max_pixels,
            "use_audio_in_video": self.config.use_audio_in_video if use_audio_in_video is None else use_audio_in_video,
        }

    def _prepare_inputs(
        self,
        prompt: str = "",
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        use_audio_in_video: Optional[bool] = None,
    ):
        if not self.loaded:
            self.load()

        sample = {
            "conversations": self._build_conversations(prompt, image_path, video_path, audio_path),
            "images": [image_path] if image_path else [],
            "videos": [video_path] if video_path else [],
            "audios": [audio_path] if audio_path else [],
        }
        if self._debug_enabled():
            logger.info(f"[INFER_DEBUG] sample.conversations={sample['conversations']!r}")
            logger.info(
                f"[INFER_DEBUG] sample media counts: images={len(sample['images'])} "
                f"videos={len(sample['videos'])} audios={len(sample['audios'])}"
            )
        self._debug_media_paths(image_path=image_path, video_path=video_path, audio_path=audio_path)
        encoded = encode_multimodal_sample_inference(
            sample=sample,
            processor=self.processor,
            chat_template=self.chat_template,
            position_id_func=self.position_id_func,
            modality_info=self.modality_info,
            force_image_gen=False,
            # enable_thinking=self.config.enable_thinking,
            thinking_prompt=self.config.thinking_prompt,
            **self._processor_kwargs(use_audio_in_video),
        )[0]
        self._debug_model_inputs(encoded, "encoded")
        self._debug_prompt_tail(encoded, "encoded")

        inputs = {}
        for key, value in encoded.items():
            if key == "labels":
                continue
            if isinstance(value, torch.Tensor):
                if key == "position_ids" and value.dim() in (1, 2):
                    value = value.unsqueeze(0)
                elif key in {"input_ids", "attention_mask"} or key.endswith("_mask"):
                    value = value.unsqueeze(0) if value.dim() == 1 else value
                value = value.to(self.device)
                if torch.is_floating_point(value):
                    value = value.to(self.compute_dtype)
                inputs[key] = value
            else:
                inputs[key] = value
        self._debug_model_inputs(inputs, "prepared")
        return inputs

    def _add_varlen_attention_kwargs(self, inputs):
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                return inputs
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            inputs["position_ids"] = position_ids

        if position_ids.dim() == 3 and position_ids.shape[1] in (3, 4):
            position_ids_for_fa = position_ids[:, 0, :]
        elif position_ids.dim() == 3 and position_ids.shape[0] in (3, 4):
            position_ids_for_fa = position_ids[0]
        elif position_ids.dim() == 2:
            position_ids_for_fa = position_ids[0:1] if position_ids.shape[0] in (3, 4) else position_ids
        else:
            position_ids_for_fa = position_ids

        (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
            position_ids_for_fa
        )
        inputs["cu_seq_lens_q"] = cu_seq_lens_q
        inputs["cu_seq_lens_k"] = cu_seq_lens_k
        inputs["max_length_q"] = max_length_q
        inputs["max_length_k"] = max_length_k
        return inputs

    def _append_generated_token(self, inputs, next_token):
        next_token = next_token.view(1, 1).to(inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)

        if "attention_mask" in inputs:
            next_mask = torch.ones(
                (inputs["attention_mask"].shape[0], 1),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            )
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], next_mask], dim=-1)

        for key, value in list(inputs.items()):
            if key == "attention_mask" or not key.endswith("_mask") or not isinstance(value, torch.Tensor):
                continue
            if value.dim() == 2 and value.shape[-1] + 1 == inputs["input_ids"].shape[-1]:
                next_modal_mask = torch.zeros(
                    (value.shape[0], 1),
                    dtype=value.dtype,
                    device=value.device,
                )
                inputs[key] = torch.cat([value, next_modal_mask], dim=-1)

        if "position_ids" in inputs:
            position_ids = inputs["position_ids"]
            if position_ids.dim() == 3 and position_ids.shape[1] in (3, 4):
                if position_ids.shape[1] == 4:
                    text_next = position_ids[:, 0:1, -1:] + 1
                    mrope_next = position_ids[:, 1:, -1:].amax(dim=1, keepdim=True) + 1
                    next_position = torch.cat([text_next, mrope_next.expand(-1, 3, -1)], dim=1)
                else:
                    next_position = position_ids[..., -1:].amax(dim=1, keepdim=True) + 1
                    next_position = next_position.expand_as(position_ids[..., -1:])
            elif position_ids.dim() == 3 and position_ids.shape[0] in (3, 4):
                if position_ids.shape[0] == 4:
                    text_next = position_ids[0:1, :, -1:] + 1
                    mrope_next = position_ids[1:, :, -1:].amax(dim=0, keepdim=True) + 1
                    next_position = torch.cat([text_next, mrope_next.expand(3, -1, -1)], dim=0)
                else:
                    next_position = position_ids[..., -1:].amax(dim=0, keepdim=True) + 1
                    next_position = next_position.expand_as(position_ids[..., -1:])
            elif position_ids.dim() == 2 and position_ids.shape[0] in (3, 4):
                if position_ids.shape[0] == 4:
                    text_next = position_ids[0:1, -1:] + 1
                    mrope_next = position_ids[1:, -1:].amax(dim=0, keepdim=True) + 1
                    next_position = torch.cat([text_next, mrope_next.expand(3, -1)], dim=0)
                else:
                    next_position = position_ids[..., -1:].amax(dim=0, keepdim=True) + 1
                    next_position = next_position.expand_as(position_ids[..., -1:])
            elif position_ids.dim() == 3:
                next_position = position_ids[..., -1:] + 1
            elif position_ids.dim() == 2:
                next_position = position_ids[:, -1:] + 1
            else:
                next_position = None
            if next_position is not None:
                inputs["position_ids"] = torch.cat([position_ids, next_position], dim=-1)
        return inputs

    @staticmethod
    def _apply_repetition_penalty(logits, input_ids, repetition_penalty: float):
        if repetition_penalty == 1.0:
            return logits
        for token_id in set(input_ids[0].tolist()):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty
        return logits

    @staticmethod
    def _sample_next_token(logits, temperature: float, top_p: float):
        if temperature is None or temperature <= 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature
        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, -float("inf"))
            logits = torch.full_like(logits, -float("inf")).scatter(0, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate_text(
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
    ) -> str:
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
        generated_ids = []

        try:
            with torch.no_grad():
                for step in range(max_new_tokens):
                    step_inputs = dict(inputs)
                    self._add_varlen_attention_kwargs(step_inputs)
                    outputs = self.model(
                        **step_inputs,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                    next_logits = outputs.logits[0, -1].float()
                    next_logits = self._apply_repetition_penalty(next_logits, inputs["input_ids"], repetition_penalty)
                    if step < min_new_tokens:
                        for eos_token_id in eos_token_ids:
                            next_logits[eos_token_id] = -float("inf")
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
        finally:
            if restore_lm_encode is not None:
                self.model.encoder.audio_encoder.lm_encode = restore_lm_encode

        return tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def generate(
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
    ):
        text = self.generate_text(
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
        audio_path_out = None
        audio_status = ""
        if enable_audio_output:
            try:
                audio_path_out = self.audio_backend.synthesize(text)
            except NotImplementedError as exc:
                audio_status = str(exc)
        return text, audio_path_out, audio_status


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Qwen3.5 + Whisper SeedOmni inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--load-mode", default="full", choices=("full",))
    parser.add_argument("--processor-path", default=os.environ.get("PROCESSOR_PATH"))
    parser.add_argument("--chat-template", default=os.environ.get("CHAT_TEMPLATE", "qwen3_5omni"))
    parser.add_argument(
        "--thinking-prompt",
        choices=("disabled", "enabled", "none"),
        default=os.environ.get("THINKING_PROMPT", "none"),
        help=(
            "Assistant prefix for empty inference turns: disabled inserts an empty <think></think> block, "
            "enabled inserts <think>, none inserts no thinking marker."
        ),
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
    parser.add_argument("--no-audio-in-video", action="store_true")
    parser.add_argument(
        "--generate-audio",
        "--enable-audio-output",
        action="store_true",
        help="Generate CosyVoice3 speech tokens after text generation. Disabled by default.",
    )
    parser.add_argument("--speech-token-output", default=None, help="Optional .json or .pt path for speech tokens.")
    parser.add_argument("--speech-chunk-manifest", default=None, help="Optional JSON path for speech chunk metadata.")
    parser.add_argument("--output-audio", default=None, help="Optional .wav output path; requires a voice condition.")
    parser.add_argument("--prompt-wav", default=None, help="Reference wav for zero-shot voice conditioning.")
    parser.add_argument("--prompt-text", default="", help="Transcript of --prompt-wav.")
    parser.add_argument("--speaker-profile", default=None, help="Embedding-only spk2info.pt for preset-speaker synthesis.")
    parser.add_argument("--speaker-id", default="fysics", help="Speaker ID in --speaker-profile.")
    parser.add_argument("--speech-sampling", type=int, default=25)
    parser.add_argument("--min-token-text-ratio", type=float, default=2.0)
    parser.add_argument("--max-token-text-ratio", type=float, default=20.0)
    parser.add_argument("--max-speech-tokens", type=int, default=None)
    parser.add_argument("--speech-chunk-size", type=int, default=100)
    parser.add_argument("--speech-chunk-silence-ms", type=int, default=200)
    parser.add_argument("--audio-speed", type=float, default=1.0)
    parser.add_argument("--bypass-text-condition-bridge", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print input/audio encoding diagnostics.")
    parser.add_argument(
        "--debug-generation-steps",
        type=int,
        default=None,
        help="When --debug is set, log selected token/top-k logits for the first N generation steps.",
    )
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
        use_audio_in_video=not args.no_audio_in_video,
    )
    enable_audio = args.generate_audio or args.speech_token_output is not None or args.output_audio is not None
    if not enable_audio:
        infer = Qwen35WhisperOmniInfer(config)
        print(infer.generate_text(
            prompt=args.prompt, image_path=args.image, video_path=args.video, audio_path=args.audio,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, min_new_tokens=args.min_new_tokens,
        ))
        return

    # Avoid importing the CosyVoice runtime for text-only inference.
    from scripts.Inference_AudioGen import DEFAULT_SPEAKER_PROFILE, Qwen35CosyVoice3OmniInfer

    infer = Qwen35CosyVoice3OmniInfer(config)
    result = infer.generate_with_audio(
        prompt=args.prompt, image_path=args.image, video_path=args.video, audio_path=args.audio,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, min_new_tokens=args.min_new_tokens,
        enable_audio_output=enable_audio, speech_token_output=args.speech_token_output,
        speech_chunk_manifest=args.speech_chunk_manifest, output_audio=args.output_audio,
        prompt_wav=args.prompt_wav, prompt_text=args.prompt_text,
        speaker_profile=args.speaker_profile or DEFAULT_SPEAKER_PROFILE, speaker_id=args.speaker_id,
        speech_sampling=args.speech_sampling, min_token_text_ratio=args.min_token_text_ratio,
        max_token_text_ratio=args.max_token_text_ratio, max_speech_tokens=args.max_speech_tokens,
        speech_chunk_size=args.speech_chunk_size, speech_chunk_silence_ms=args.speech_chunk_silence_ms,
        audio_speed=args.audio_speed, bypass_text_condition_bridge=args.bypass_text_condition_bridge,
    )
    print(result.text)
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
