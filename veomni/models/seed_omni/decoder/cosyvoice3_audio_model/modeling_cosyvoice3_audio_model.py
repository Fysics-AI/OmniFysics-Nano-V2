import os
import sys
import types
import uuid
import inspect
from dataclasses import dataclass
from functools import partial
from importlib.machinery import ModuleSpec
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from ..base import BaseDecoderModelMixin, BaseDecoderOutput
from .configuration_cosyvoice3_audio_model import CosyVoice3AudioDecoderConfig
from .text_normalization import CosyVoiceTextNormalizer


COSY_IGNORE_ID = -1


@dataclass
class CosyVoice3AudioDecoderOutput(BaseDecoderOutput):
    speech_acc: Optional[torch.FloatTensor] = None
    speech_token_ids: Optional[List[torch.LongTensor]] = None


@dataclass
class SpeechGenerationItem:
    token_ids: torch.LongTensor
    stop_token_id: Optional[int]
    stopped_by_eos: bool
    hit_max_length: bool


class Qwen2EncoderAdapter(nn.Module):
    """CosyVoice Qwen2Encoder-compatible wrapper built from a config."""

    def __init__(self, config):
        super().__init__()
        self.model = Qwen2ForCausalLM(config)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        seq_len = xs.size(1)
        positions = torch.arange(seq_len, device=xs.device).unsqueeze(0)
        masks = positions < xs_lens.to(xs.device).unsqueeze(1)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        outs = self.model(
            inputs_embeds=xs,
            # The autoregressive path is unpadded. Transformers can derive
            # the causal mask from the KV-cache length; passing a [B, 1]
            # mask after the prefill is interpreted differently across
            # Transformers versions.
            attention_mask=None,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        return outs.hidden_states[-1], outs.past_key_values


def _ensure_cosyvoice_importable(cosyvoice_repo_path: str) -> None:
    if cosyvoice_repo_path and os.path.isdir(cosyvoice_repo_path) and cosyvoice_repo_path not in sys.path:
        sys.path.insert(0, cosyvoice_repo_path)
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        os.environ.pop("onnx_path", None)
        dummy_onnxruntime = types.ModuleType("onnxruntime")
        dummy_onnxruntime.__spec__ = ModuleSpec("onnxruntime", loader=None)
        sys.modules.setdefault("onnxruntime", dummy_onnxruntime)


def _build_audio_output_projector(config: CosyVoice3AudioDecoderConfig) -> nn.Module:
    input_size = config.output_size
    output_size = config.llm_input_size
    if input_size is None:
        raise ValueError("CosyVoice3AudioDecoderConfig.output_size must be set to the foundation hidden size.")

    if config.projector_type == "linear":
        return nn.Linear(input_size, output_size)
    if config.projector_type != "mlp":
        raise ValueError(f"Unsupported audio output projector type: {config.projector_type}")

    hidden_size = config.projector_hidden_size
    layers: List[nn.Module] = [nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size), nn.GELU()]
    for _ in range(max(0, config.projector_depth - 2)):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
    layers.extend([nn.Linear(hidden_size, output_size), nn.LayerNorm(output_size)])
    return nn.Sequential(*layers)


class TextQueryCrossAttentionBridge(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        query_position_type: str,
        query_position_base: float,
        query_position_scale: float | None,
        gate_init: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"Cross-attention hidden size {hidden_size} must be divisible by num_heads {num_heads}.")
        if hidden_size % 2 != 0:
            raise ValueError(f"Sinusoidal query position encoding requires an even hidden size, got {hidden_size}.")
        if query_position_type != "sinusoidal":
            raise ValueError(f"Unsupported cross-attention query position type: {query_position_type!r}.")
        if query_position_base <= 0:
            raise ValueError(f"Cross-attention query position base must be positive, got {query_position_base}.")

        self.gate_init = gate_init
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.query_position_type = query_position_type
        self.query_position_base = query_position_base
        if query_position_scale is None:
            query_position_scale = hidden_size**-0.5
        if query_position_scale <= 0:
            raise ValueError(f"Cross-attention query position scale must be positive, got {query_position_scale}.")
        self.query_position_scale = query_position_scale
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # FSDP1 cannot shard scalar parameters; keep one element while preserving broadcast semantics.
        self.gate = nn.Parameter(torch.empty(1))

    def _query_position_encoding(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        dimensions = torch.arange(0, self.hidden_size, 2, device=device, dtype=torch.float32)
        inverse_frequency = 1.0 / (self.query_position_base ** (dimensions / self.hidden_size))
        angles = torch.outer(positions, inverse_frequency)
        position_encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        return (self.query_position_scale * position_encoding).to(dtype=dtype)

    def forward(self, text_query: torch.Tensor, projected_hidden: torch.Tensor) -> torch.Tensor:
        if text_query.dim() != 2 or projected_hidden.dim() != 2:
            raise ValueError(
                "Cross-attention bridge expects unbatched [sequence, hidden] tensors, got "
                f"{tuple(text_query.shape)} and {tuple(projected_hidden.shape)}."
            )
        if text_query.shape[0] == 0 or projected_hidden.shape[0] == 0:
            raise ValueError("Cross-attention bridge received an empty text or hidden-state sequence.")

        query_position = self._query_position_encoding(
            length=text_query.shape[0],
            device=text_query.device,
            dtype=text_query.dtype,
        )
        projected_hidden = F.normalize(projected_hidden.float(), p=2, dim=-1).to(text_query.dtype)
        query = self.query_norm(text_query + query_position)
        key = self.key_norm(projected_hidden)
        query = self.q_proj(query).view(-1, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        key = self.k_proj(key).view(-1, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        value = self.v_proj(projected_hidden).view(-1, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        attended_hidden = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended_hidden = attended_hidden.transpose(1, 2).reshape(text_query.shape[0], self.hidden_size)
        attended_hidden = self.out_proj(attended_hidden)
        gate = self.gate.tanh().to(text_query.dtype)
        return text_query + gate * attended_hidden


class CosyVoice3AudioDecoder(BaseDecoderModelMixin):
    config_class = CosyVoice3AudioDecoderConfig
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen2DecoderLayer"]

    def __init__(self, config: CosyVoice3AudioDecoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        _ensure_cosyvoice_importable(config.cosyvoice_repo_path)

        self.audio_output_projector = _build_audio_output_projector(config)
        self.text_condition_bridge = TextQueryCrossAttentionBridge(
            hidden_size=config.llm_input_size,
            num_heads=config.cross_attention_num_heads,
            dropout=config.cross_attention_dropout,
            query_position_type=config.cross_attention_query_position_type,
            query_position_base=config.cross_attention_query_position_base,
            query_position_scale=config.cross_attention_query_position_scale,
            gate_init=config.cross_attention_gate_init,
        )
        self.cosy_lm = self._build_cosy_lm(config)
        self._pretrained_loaded = False
        self._foundation_tokenizer = None
        self._cosy_text_tokenizer = None
        self._speech_text_normalizer = None
        self._initialize_text_condition_bridge()
        self._init_control_token_ids()
        self.apply_default_trainable_config()

        if config.load_pretrained and not config.use_dummy_lm and not any(p.is_meta for p in self.parameters()):
            self.load_pretrained_weights()

    def _build_cosy_lm(self, config: CosyVoice3AudioDecoderConfig) -> nn.Module:
        from cosyvoice.llm.llm import CosyVoice3LM
        from cosyvoice.utils.common import ras_sampling

        if config.use_dummy_lm:
            qwen_config = Qwen2Config(
                vocab_size=config.dummy_vocab_size,
                hidden_size=config.llm_input_size,
                intermediate_size=config.dummy_intermediate_size,
                num_hidden_layers=config.dummy_num_hidden_layers,
                num_attention_heads=config.dummy_num_attention_heads,
                num_key_value_heads=config.dummy_num_key_value_heads,
                max_position_embeddings=512,
            )
        else:
            qwen_config = AutoConfig.from_pretrained(self._blanken_path)

        llm = Qwen2EncoderAdapter(qwen_config)
        sampling = partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1)
        return CosyVoice3LM(
            llm_input_size=config.llm_input_size,
            llm_output_size=config.llm_output_size,
            speech_token_size=config.speech_token_size,
            llm=llm,
            sampling=sampling,
            length_normalized_loss=True,
            lsm_weight=0.0,
            mix_ratio=[5, 15],
        )

    @property
    def _blanken_path(self) -> str:
        return os.path.join(self.config.cosyvoice3_path, "CosyVoice-BlankEN")

    def _init_control_token_ids(self) -> None:
        if self.config.use_dummy_lm or not self.config.control_text:
            control_ids = torch.empty(0, dtype=torch.long)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self._blanken_path)
            if "<|endofprompt|>" in self.config.control_text:
                tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofprompt|>"]})
            control_ids = torch.tensor(
                tokenizer.encode(self.config.control_text, allowed_special="all"),
                dtype=torch.long,
            )
        self.register_buffer("control_token_ids", control_ids, persistent=False)

    @property
    def foundation_tokenizer(self):
        if self._foundation_tokenizer is None:
            if not self.config.foundation_tokenizer_path:
                raise ValueError(
                    "CosyVoice3 text-query bridge requires `foundation_tokenizer_path` to decode answer token IDs."
                )
            self._foundation_tokenizer = AutoTokenizer.from_pretrained(self.config.foundation_tokenizer_path)
        return self._foundation_tokenizer

    @property
    def cosy_text_tokenizer(self):
        if self._cosy_text_tokenizer is None:
            self._cosy_text_tokenizer = AutoTokenizer.from_pretrained(self._blanken_path)
        return self._cosy_text_tokenizer

    @property
    def speech_text_normalizer(self):
        if self._speech_text_normalizer is None:
            self._speech_text_normalizer = CosyVoiceTextNormalizer(self.config.cosyvoice_repo_path)
        return self._speech_text_normalizer

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def _initialize_text_condition_bridge(self) -> None:
        self.text_condition_bridge.apply(self._init_weights)
        self.text_condition_bridge.gate.data.fill_(self.config.cross_attention_gate_init)

    @staticmethod
    def _parameter_names(module: nn.Module) -> set[str]:
        return {name for name, _ in module.named_parameters()}

    def initialize_missing_weights(self, missing_keys: set[str]) -> None:
        """Restore custom decoder initialization after the outer model was materialized with ``to_empty``."""
        groups = (
            (
                "audio_output_projector",
                self.audio_output_projector,
                lambda: self.audio_output_projector.apply(self._init_weights),
            ),
            ("text_condition_bridge", self.text_condition_bridge, self._initialize_text_condition_bridge),
        )
        for prefix, module, initializer in groups:
            expected = {f"{prefix}.{name}" for name in self._parameter_names(module)}
            missing = expected.intersection(missing_keys)
            if missing and missing != expected:
                present = sorted(expected - missing)
                raise RuntimeError(
                    f"CosyVoice3 `{prefix}` is only partially present in the checkpoint. "
                    f"Missing keys: {sorted(missing)}; present keys: {present}."
                )
            if missing:
                initializer()

        self.validate_bridge_initialization()

    def validate_bridge_initialization(self) -> None:
        bridge = self.text_condition_bridge
        if not torch.isfinite(bridge.gate).all() or torch.count_nonzero(bridge.gate).item() == 0:
            raise RuntimeError(
                "CosyVoice3 text-condition bridge gate is zero or non-finite after initialization; "
                "the cross-attention branch would receive no gradient."
            )
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            weight = getattr(bridge, name).weight
            if not torch.isfinite(weight).all() or torch.count_nonzero(weight).item() == 0:
                raise RuntimeError(f"CosyVoice3 text-condition bridge `{name}.weight` is zero or non-finite.")

    def load_pretrained_weights(self, init_device: str = "cuda") -> None:
        if self._pretrained_loaded or self.config.use_dummy_lm or not self.config.load_pretrained:
            return

        llm_path = os.path.join(self.config.cosyvoice3_path, "llm.pt")
        if not os.path.exists(llm_path):
            raise FileNotFoundError(f"CosyVoice3 llm.pt not found: {llm_path}")

        if any(p.is_meta for p in self.parameters()):
            self.to_empty(device=init_device)
            self.audio_output_projector.apply(self._init_weights)
            self._initialize_text_condition_bridge()

        state_dict = torch.load(llm_path, map_location="cpu", weights_only=True)
        self.cosy_lm.load_state_dict(state_dict, strict=True)
        self._pretrained_loaded = True
        self.apply_default_trainable_config()

    def apply_default_trainable_config(self) -> None:
        self.requires_grad_(False)
        if self.config.train_audio_output_projector:
            self.audio_output_projector.requires_grad_(True)
            self.text_condition_bridge.requires_grad_(True)
        if self.config.train_cosyvoice3_lm:
            self.cosy_lm.requires_grad_(True)

    def set_projector_trainable_only(self) -> None:
        self.requires_grad_(False)
        self.audio_output_projector.requires_grad_(True)
        self.text_condition_bridge.requires_grad_(True)

    def set_lm_trainable_only(self) -> None:
        self.requires_grad_(False)
        self.cosy_lm.requires_grad_(True)

    def set_projector_and_lm_trainable(self) -> None:
        self.requires_grad_(False)
        self.audio_output_projector.requires_grad_(True)
        self.text_condition_bridge.requires_grad_(True)
        self.cosy_lm.requires_grad_(True)

    @staticmethod
    def _enabled(value) -> bool:
        if value is None:
            return False
        if isinstance(value, torch.Tensor):
            return bool(value.detach().bool().any().item())
        return bool(value)

    @staticmethod
    def _lengths_to_list(lengths: Optional[torch.Tensor], fallback: Sequence[int]) -> List[int]:
        if lengths is None:
            return [int(item) for item in fallback]
        if isinstance(lengths, torch.Tensor):
            return [int(item) for item in lengths.detach().cpu().reshape(-1).tolist()]
        return [int(item) for item in lengths]

    @staticmethod
    def _split_flat_tensor(flat: torch.Tensor, lengths: Sequence[int]) -> List[torch.Tensor]:
        pieces = []
        offset = 0
        for length in lengths:
            pieces.append(flat.narrow(0, offset, length))
            offset += length
        if offset != flat.shape[0]:
            raise ValueError(f"Length sum {offset} does not match flat tensor length {flat.shape[0]}.")
        return pieces

    def _collect_answer_hidden(
        self,
        hidden_states: torch.Tensor,
        answer_text_mask: Optional[torch.Tensor],
        answer_text_len: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        if answer_text_mask is None:
            lengths = [hidden_states.shape[1]] * hidden_states.shape[0]
            return [hidden_states[i, : lengths[i]] for i in range(hidden_states.shape[0])]

        answer_text_mask = answer_text_mask.to(hidden_states.device).bool()
        if answer_text_mask.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "answer_text_mask shape must match hidden_states batch/sequence dimensions: "
                f"{tuple(answer_text_mask.shape)} vs {tuple(hidden_states.shape)}"
            )

        selected = hidden_states[answer_text_mask]
        fallback_lengths = answer_text_mask.long().sum(dim=1).tolist()
        lengths = self._lengths_to_list(answer_text_len, fallback_lengths)
        return self._split_flat_tensor(selected, lengths)

    def _collect_answer_token_ids(
        self,
        answer_token_ids: Optional[torch.Tensor],
        answer_text_mask: Optional[torch.Tensor],
        answer_text_len: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        if answer_token_ids is None:
            raise ValueError(
                "CosyVoice3 text-query bridge requires answer token IDs so they can be decoded and retokenized."
            )
        answer_token_ids = answer_token_ids.to(dtype=torch.long)
        if answer_text_mask is not None:
            answer_text_mask = answer_text_mask.to(answer_token_ids.device).bool()
            if answer_text_mask.shape != answer_token_ids.shape:
                raise ValueError(
                    "answer_text_mask shape must match answer_token_ids: "
                    f"{tuple(answer_text_mask.shape)} vs {tuple(answer_token_ids.shape)}"
                )
            selected = answer_token_ids[answer_text_mask]
            fallback_lengths = answer_text_mask.long().sum(dim=1).tolist()
            lengths = self._lengths_to_list(answer_text_len, fallback_lengths)
            return self._split_flat_tensor(selected, lengths)

        if answer_token_ids.dim() == 1:
            lengths = self._lengths_to_list(answer_text_len, [answer_token_ids.numel()])
            return self._split_flat_tensor(answer_token_ids, lengths)
        if answer_token_ids.dim() == 2:
            lengths = self._lengths_to_list(
                answer_text_len,
                [answer_token_ids.shape[1]] * answer_token_ids.shape[0],
            )
            return [answer_token_ids[i, :length] for i, length in enumerate(lengths)]
        raise ValueError(f"answer_token_ids must be 1D or 2D, got shape {tuple(answer_token_ids.shape)}")

    def _collect_cosy_text_tokens(
        self,
        cosy_text_token: torch.Tensor,
        cosy_text_token_len: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        cosy_text_token = cosy_text_token.to(dtype=torch.long)
        if cosy_text_token.dim() == 1:
            lengths = self._lengths_to_list(cosy_text_token_len, [cosy_text_token.numel()])
            return self._split_flat_tensor(cosy_text_token, lengths)
        if cosy_text_token.dim() == 2:
            lengths = self._lengths_to_list(
                cosy_text_token_len,
                [cosy_text_token.shape[1]] * cosy_text_token.shape[0],
            )
            return [cosy_text_token[i, :length] for i, length in enumerate(lengths)]
        raise ValueError(f"cosy_text_token must be 1D or 2D, got shape {tuple(cosy_text_token.shape)}")

    def _retokenize_answer_tokens(self, answer_token_ids: List[torch.Tensor]) -> List[torch.Tensor]:
        cosy_token_ids = []
        for token_ids in answer_token_ids:
            text = self.foundation_tokenizer.decode(
                token_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not text:
                raise ValueError("Answer token IDs decoded to empty text; cannot build CosyVoice3 text queries.")
            speech_text = self.speech_text_normalizer.normalize(text)
            ids = self.cosy_text_tokenizer.encode(speech_text, add_special_tokens=False)
            if not ids:
                raise ValueError("CosyVoice-BlankEN tokenizer produced an empty answer token sequence.")
            cosy_token_ids.append(torch.tensor(ids, dtype=torch.long, device=self.device))
        return cosy_token_ids

    def _build_answer_conditions(
        self,
        hidden_states: torch.Tensor,
        answer_text_mask: Optional[torch.Tensor] = None,
        answer_text_len: Optional[torch.Tensor] = None,
        answer_token_ids: Optional[torch.Tensor] = None,
        cosy_text_token: Optional[torch.Tensor] = None,
        cosy_text_token_len: Optional[torch.Tensor] = None,
        bypass_text_condition_bridge: bool = False,
    ) -> List[torch.Tensor]:
        answer_hidden = self._collect_answer_hidden(hidden_states, answer_text_mask, answer_text_len)
        if not answer_hidden or any(item.shape[0] == 0 for item in answer_hidden):
            raise ValueError("CosyVoice3 text-query bridge received an empty answer hidden-state sequence.")

        if cosy_text_token is None:
            foundation_token_ids = self._collect_answer_token_ids(
                answer_token_ids,
                answer_text_mask,
                answer_text_len,
            )
            cosy_text_tokens = self._retokenize_answer_tokens(foundation_token_ids)
        else:
            cosy_text_tokens = self._collect_cosy_text_tokens(cosy_text_token, cosy_text_token_len)
        if len(cosy_text_tokens) != len(answer_hidden):
            raise ValueError(
                f"Cosy text batch size {len(cosy_text_tokens)} does not match "
                f"answer hidden batch size {len(answer_hidden)}."
            )

        projected_answer_hidden = None
        if not bypass_text_condition_bridge:
            projected_flat = self.audio_output_projector(torch.cat(answer_hidden, dim=0))
            projected_answer_hidden = self._split_flat_tensor(
                projected_flat, [item.shape[0] for item in answer_hidden]
            )

        conditions = []
        text_embedding = self.cosy_lm.llm.model.model.embed_tokens
        for index, cosy_ids in enumerate(cosy_text_tokens):
            cosy_ids = cosy_ids.to(device=answer_hidden[index].device, dtype=torch.long)
            if (cosy_ids < 0).any() or (cosy_ids >= text_embedding.num_embeddings).any():
                raise ValueError(f"Cosy text token ids must be in [0, {text_embedding.num_embeddings - 1}].")
            text_query = text_embedding(cosy_ids).to(dtype=answer_hidden[index].dtype)
            if bypass_text_condition_bridge:
                conditions.append(text_query)
            else:
                projected_hidden = projected_answer_hidden[index]
                conditions.append(self.text_condition_bridge(text_query, projected_hidden))
        return conditions

    def _collect_target_tokens(
        self,
        target_speech_token: Optional[torch.Tensor],
        target_speech_token_len: Optional[torch.Tensor],
    ) -> Optional[List[torch.Tensor]]:
        if target_speech_token is None:
            return None

        target_speech_token = target_speech_token.to(device=self.device, dtype=torch.long)
        if target_speech_token_len is None:
            if target_speech_token.dim() == 1:
                target_speech_token_len = target_speech_token.new_tensor([target_speech_token.numel()])
            else:
                target_speech_token_len = target_speech_token.new_full(
                    (target_speech_token.shape[0],), target_speech_token.shape[1]
                )
        lengths = self._lengths_to_list(target_speech_token_len, [])

        if target_speech_token.dim() == 1:
            return self._split_flat_tensor(target_speech_token, lengths)
        if target_speech_token.dim() == 2:
            return [target_speech_token[i, :length] for i, length in enumerate(lengths)]
        raise ValueError(f"target_speech_token must be 1D or 2D, got shape {tuple(target_speech_token.shape)}")

    def _control_prefix_emb(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.control_token_ids.numel() == 0:
            return torch.zeros(0, self.config.llm_input_size, device=device, dtype=dtype)
        control_ids = self.control_token_ids.to(device=device)
        return self.cosy_lm.llm.model.model.embed_tokens(control_ids).to(dtype=dtype)

    def _prepare_lm_batch(
        self,
        answer_conditions: List[torch.Tensor],
        target_tokens: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(answer_conditions) != len(target_tokens):
            raise ValueError(
                f"answer condition batch size {len(answer_conditions)} does not match "
                f"target speech token batch size {len(target_tokens)}."
            )

        lm_inputs, lm_targets = [], []
        device = answer_conditions[0].device
        dtype = answer_conditions[0].dtype
        control_emb = self._control_prefix_emb(device=device, dtype=dtype)
        sos_emb = self.cosy_lm.speech_embedding.weight[self.cosy_lm.sos].reshape(1, -1).to(device=device, dtype=dtype)
        task_id_emb = (
            self.cosy_lm.speech_embedding.weight[self.cosy_lm.task_id].reshape(1, -1).to(device=device, dtype=dtype)
        )

        for answer_condition, speech_ids in zip(answer_conditions, target_tokens, strict=True):
            answer_condition = answer_condition.to(device=device, dtype=dtype)
            speech_ids = speech_ids.to(device=device, dtype=torch.long)
            if speech_ids.numel() == 0:
                raise ValueError("target_speech_token contains an empty sequence.")
            if (speech_ids < 0).any() or (speech_ids >= self.cosy_lm.speech_token_size).any():
                raise ValueError(f"target_speech_token ids must be in [0, {self.cosy_lm.speech_token_size - 1}].")

            speech_emb = self.cosy_lm.speech_embedding(speech_ids).to(dtype=dtype)
            lm_input = torch.cat([sos_emb, control_emb, answer_condition, task_id_emb, speech_emb], dim=0)
            ignore_prefix_len = 1 + control_emb.shape[0] + answer_condition.shape[0]
            lm_target = torch.cat(
                [
                    torch.full((ignore_prefix_len,), COSY_IGNORE_ID, dtype=torch.long, device=device),
                    speech_ids,
                    torch.tensor([self.cosy_lm.eos_token], dtype=torch.long, device=device),
                ],
                dim=0,
            )
            if lm_input.shape[0] != lm_target.shape[0]:
                raise ValueError(f"CosyVoice3 LM input/target length mismatch: {lm_input.shape} vs {lm_target.shape}")
            lm_inputs.append(lm_input)
            lm_targets.append(lm_target)

        lm_input = pad_sequence(lm_inputs, batch_first=True, padding_value=0.0)
        lm_input_len = torch.tensor([item.shape[0] for item in lm_inputs], dtype=torch.int32, device=device)
        lm_target = pad_sequence(lm_targets, batch_first=True, padding_value=COSY_IGNORE_ID)
        return lm_input, lm_input_len, lm_target

    def _run_cosy_lm(
        self,
        answer_conditions: List[torch.Tensor],
        target_tokens: List[torch.Tensor],
    ) -> CosyVoice3AudioDecoderOutput:
        from cosyvoice.utils.common import th_accuracy

        lm_input, lm_input_len, lm_target = self._prepare_lm_batch(answer_conditions, target_tokens)
        lm_output, _ = self.cosy_lm.llm(lm_input, lm_input_len)
        logits = self.cosy_lm.llm_decoder(lm_output)
        loss = self.cosy_lm.criterion_ce(logits, lm_target) * self.config.audio_loss_weight
        acc = th_accuracy(logits.view(-1, logits.shape[-1]), lm_target, ignore_label=COSY_IGNORE_ID)
        return CosyVoice3AudioDecoderOutput(loss=loss, logits=logits, speech_acc=acc)

    def forward(
        self,
        hidden_states: torch.Tensor,
        answer_text_mask: Optional[torch.Tensor] = None,
        answer_text_len: Optional[torch.Tensor] = None,
        answer_token_ids: Optional[torch.Tensor] = None,
        cosy_text_token: Optional[torch.Tensor] = None,
        cosy_text_token_len: Optional[torch.Tensor] = None,
        target_speech_token: Optional[torch.Tensor] = None,
        target_speech_token_len: Optional[torch.Tensor] = None,
        speech_token_labels: Optional[torch.Tensor] = None,
        speech_token_label_lens: Optional[torch.Tensor] = None,
        enable_audio_output=None,
        **kwargs,
    ) -> CosyVoice3AudioDecoderOutput:
        if not self._enabled(enable_audio_output) and target_speech_token is None and speech_token_labels is None:
            return CosyVoice3AudioDecoderOutput()

        if target_speech_token is None:
            target_speech_token = speech_token_labels
            target_speech_token_len = speech_token_label_lens

        answer_conditions = self._build_answer_conditions(
            hidden_states=hidden_states,
            answer_text_mask=answer_text_mask,
            answer_text_len=answer_text_len,
            answer_token_ids=answer_token_ids,
            cosy_text_token=cosy_text_token,
            cosy_text_token_len=cosy_text_token_len,
        )
        target_tokens = self._collect_target_tokens(target_speech_token, target_speech_token_len)
        if target_tokens is None:
            return CosyVoice3AudioDecoderOutput()
        return self._run_cosy_lm(answer_conditions, target_tokens)

    @torch.no_grad()
    def lm_generate(
        self,
        hidden_states: torch.Tensor,
        answer_text_mask: Optional[torch.Tensor] = None,
        answer_text_len: Optional[torch.Tensor] = None,
        answer_token_ids: Optional[torch.Tensor] = None,
        cosy_text_token: Optional[torch.Tensor] = None,
        cosy_text_token_len: Optional[torch.Tensor] = None,
        sampling: int = 25,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
        max_speech_tokens: Optional[int] = None,
        return_generation_info: bool = False,
        bypass_text_condition_bridge: bool = False,
        **kwargs,
    ) -> List[torch.LongTensor] | List[SpeechGenerationItem]:
        answer_conditions = self._build_answer_conditions(
            hidden_states=hidden_states,
            answer_text_mask=answer_text_mask,
            answer_text_len=answer_text_len,
            answer_token_ids=answer_token_ids,
            cosy_text_token=cosy_text_token,
            cosy_text_token_len=cosy_text_token_len,
            bypass_text_condition_bridge=bypass_text_condition_bridge,
        )

        generated = []
        for answer_condition in answer_conditions:
            device, dtype = answer_condition.device, answer_condition.dtype
            control_emb = self._control_prefix_emb(device=device, dtype=dtype).unsqueeze(0)
            sos_emb = (
                self.cosy_lm.speech_embedding.weight[self.cosy_lm.sos].reshape(1, 1, -1).to(device=device, dtype=dtype)
            )
            task_id_emb = (
                self.cosy_lm.speech_embedding.weight[self.cosy_lm.task_id]
                .reshape(1, 1, -1)
                .to(device=device, dtype=dtype)
            )
            lm_input = torch.cat([sos_emb, control_emb, answer_condition.unsqueeze(0), task_id_emb], dim=1)
            max_len = (
                int(answer_condition.shape[0] * max_token_text_ratio)
                if max_speech_tokens is None
                else max_speech_tokens
            )
            if max_len <= 0:
                raise ValueError(f"CosyVoice3 speech generation max length must be positive, got {max_len}.")
            min_len = min(int(answer_condition.shape[0] * min_token_text_ratio), max(0, max_len - 1))
            generation_state = {}
            generation_kwargs = {
                "lm_input": lm_input,
                "sampling": sampling,
                "min_len": min_len,
                "max_len": max_len,
                "uuid": str(uuid.uuid4()),
            }
            supports_generation_state = "generation_state" in inspect.signature(
                self.cosy_lm.inference_wrapper
            ).parameters
            if supports_generation_state:
                generation_kwargs["generation_state"] = generation_state

            tokens = []
            for token in self.cosy_lm.inference_wrapper(**generation_kwargs):
                token = int(token)
                if token < self.cosy_lm.speech_token_size:
                    tokens.append(token)
            if not supports_generation_state:
                generation_state["stop_token_id"] = next(
                    iter(getattr(self.cosy_lm, "stop_token_ids", ())), None
                )
                generation_state["stopped_by_eos"] = len(tokens) < max_len
                generation_state["hit_max_length"] = len(tokens) >= max_len
            token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
            if return_generation_info:
                generated.append(
                    SpeechGenerationItem(
                        token_ids=token_ids,
                        stop_token_id=generation_state.get("stop_token_id"),
                        stopped_by_eos=bool(generation_state.get("stopped_by_eos", False)),
                        hit_max_length=bool(generation_state.get("hit_max_length", len(tokens) >= max_len)),
                    )
                )
            else:
                generated.append(token_ids)
        return generated

    def lm_head(self, hidden_states: torch.Tensor, labels: torch.Tensor = None, **kwargs) -> BaseDecoderOutput:
        return self.forward(hidden_states=hidden_states, target_speech_token=labels, **kwargs)

    def lm_embed(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        conditions = self._build_answer_conditions(hidden_states=hidden_states, **kwargs)
        return pad_sequence(conditions, batch_first=True, padding_value=0.0)

    def lm_encode(self, features: torch.Tensor, **kwargs):
        raise NotImplementedError("CosyVoice3AudioDecoder does not encode output audio into Seed-Omni input embeds.")

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        return {"features": torch.zeros(1, 1, self.config.output_size, dtype=self.dtype, device=self.device)}
