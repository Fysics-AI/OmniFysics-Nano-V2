# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ...utils.single_process import get_runtime_state, identity_layout
from ...utils.single_process import get_runtime_state
from ...utils import logging
from ...utils.constants import IGNORE_INDEX
from ..loader import get_model_class
from .configuration_seed_omni import SeedOmniConfig, SeedOmniDecoderConfig, SeedOmniEncoderConfig
from .decoder import BaseDecoderModelMixin, BaseDecoderOutput
from .encoder import BaseEncoderModelMixin
from .foundation import BaseFoundationModelMixin


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers import Cache


def extract_model_inputs(prefix: str, kwargs: Dict[str, "torch.Tensor"]):
    model_inputs = {}
    for key, value in kwargs.items():
        if key.startswith(prefix):
            model_inputs[key[len(prefix) :]] = value
    return model_inputs


def _has_trainable_parameters(module: torch.nn.Module) -> bool:
    return any(param.requires_grad for param in module.parameters())


@dataclass
class SeedOmniOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    losses: Optional[Dict[str, torch.FloatTensor]] = None
    speech_token_logits: Optional[torch.FloatTensor] = None
    speech_token_ids: Optional[List[torch.LongTensor]] = None


class SeedOmniPreTrainedModel(PreTrainedModel):
    config_class = SeedOmniConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    @property
    def _no_split_modules(self):
        no_split_modules = []
        for module in self.children():
            if isinstance(module, PreTrainedModel) and module._no_split_modules:
                no_split_modules.extend(module._no_split_modules)
            elif isinstance(module, nn.ModuleDict):
                for sub_module in module.children():
                    if isinstance(sub_module, PreTrainedModel) and sub_module._no_split_modules:
                        no_split_modules.extend(sub_module._no_split_modules)

        return no_split_modules

    @_no_split_modules.setter
    def _no_split_modules(self, value):
        pass

    @property
    def all_tied_weights_keys(self):
        expanded_tied_weights = self.get_expanded_tied_weights_keys(all_submodels=False)
        dynamic_tied_weights = getattr(self, "_dynamic_tied_weights_keys", None) or []
        for key in dynamic_tied_weights:
            expanded_tied_weights.setdefault(key, key)
        return expanded_tied_weights

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SeedOmniEncoderModel(SeedOmniPreTrainedModel):
    config_class = SeedOmniEncoderConfig

    def __init__(self, config: SeedOmniEncoderConfig):
        super().__init__(config)
        torch_dtype = torch.get_default_dtype()
        self.text_encoder = nn.Embedding(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            padding_idx=config.text_config.pad_token_id,
            dtype=torch_dtype,
        )
        self.modality = []
        if config.image_config.model_type:
            model_cls = get_model_class(config.image_config)
            self.image_encoder: BaseEncoderModelMixin = model_cls._from_config(
                config.image_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("image")
            self.modality.append("video")

        if config.video_config.model_type:
            model_cls = get_model_class(config.video_config)
            self.video_encoder: BaseEncoderModelMixin = model_cls._from_config(
                config.video_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("video") if "video" not in self.modality else None

        if config.audio_config.model_type:
            model_cls = get_model_class(config.audio_config)
            self.audio_encoder: BaseEncoderModelMixin = model_cls._from_config(
                config.audio_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("audio")

        self.encode_input = config.encode_input
        self.encode_output = config.encode_output

    def set_projector_trainable_only(self):
        for module in self.children():
            if isinstance(module, BaseEncoderModelMixin):
                module.set_projector_trainable_only()

    def image_forward(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        if self.encode_input:
            input_image_inputs = extract_model_inputs("image_input_", kwargs)
            input_image_mask: torch.Tensor = input_image_inputs.pop("mask", None)
            if input_image_inputs:
                input_image_features: torch.Tensor = self.image_encoder.lm_encode(**input_image_inputs).to(
                    inputs_embeds
                )
                if get_runtime_state().sp_enabled:
                    input_image_features = identity_layout(
                        input_image_features, seq_dim=0, head_dim=1, group=get_runtime_state().sp_group
                    )
                input_image_features = input_image_features[: input_image_mask.sum()]
                image_mask = input_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, input_image_features)
            elif self.training and _has_trainable_parameters(self.image_encoder):
                dummy_embeds: torch.Tensor = self.image_encoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0

        if self.encode_output:
            output_image_inputs = extract_model_inputs("image_output_", kwargs)
            output_image_mask: torch.Tensor = output_image_inputs.pop("mask", None)
            if output_image_inputs:
                output_image_features: torch.Tensor = self.image_encoder.lm_encode(**output_image_inputs).to(
                    inputs_embeds
                )
                if get_runtime_state().sp_enabled:
                    output_image_features = identity_layout(
                        output_image_features, seq_dim=0, head_dim=1, group=get_runtime_state().sp_group
                    )
                output_image_features = output_image_features[: output_image_mask.sum()]
                image_mask = output_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, output_image_features)
                decoder_inputs["image_output_labels"] = output_image_features
            elif self.training and _has_trainable_parameters(self.image_encoder):
                dummy_embeds: torch.Tensor = self.image_encoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0
        return inputs_embeds

    def video_forward(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        if self.encode_input:
            input_video_inputs = extract_model_inputs("video_input_", kwargs)
            input_video_mask: torch.Tensor = input_video_inputs.pop("mask", None)
            if input_video_inputs:
                if getattr(self, "video_encoder", None) is not None:
                    input_video_features: torch.Tensor = self.video_encoder.lm_encode(**input_video_inputs).to(
                        inputs_embeds
                    )
                else:
                    input_video_features: torch.Tensor = self.image_encoder.lm_encode(**input_video_inputs).to(
                        inputs_embeds
                    )
                if get_runtime_state().sp_enabled:
                    input_video_features = identity_layout(
                        input_video_features, seq_dim=0, head_dim=1, group=get_runtime_state().sp_group
                    )
                input_video_features = input_video_features[: input_video_mask.sum()]
                video_mask = input_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, input_video_features)
            elif self.training:
                if getattr(self, "video_encoder", None) is not None:
                    if _has_trainable_parameters(self.video_encoder):
                        dummy_embeds: torch.Tensor = self.video_encoder.lm_dummy_encode()
                        inputs_embeds += dummy_embeds.mean() * 0.0
                elif _has_trainable_parameters(self.image_encoder):
                    dummy_embeds: torch.Tensor = self.image_encoder.lm_dummy_encode()
                    inputs_embeds += dummy_embeds.mean() * 0.0

        return inputs_embeds

    def audio_forward(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        if self.encode_input:
            input_audio_inputs = extract_model_inputs("audio_input_", kwargs)
            input_audio_mask: torch.Tensor = input_audio_inputs.pop("mask", None)
            if input_audio_inputs and input_audio_mask.sum() > 0:
                input_audio_features: torch.Tensor = self.audio_encoder.lm_encode(**input_audio_inputs).to(
                    inputs_embeds
                )
                if get_runtime_state().sp_enabled:
                    input_audio_features = identity_layout(
                        input_audio_features, seq_dim=0, head_dim=1, group=get_runtime_state().sp_group
                    )
                input_audio_features = input_audio_features[: input_audio_mask.sum()]
                audio_mask = input_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, input_audio_features)
            elif self.training and _has_trainable_parameters(self.audio_encoder):
                dummy_embeds: torch.Tensor = self.audio_encoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0

        return inputs_embeds

    def forward(self, input_ids: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        inputs_embeds: torch.Tensor = self.text_encoder(input_ids)
        decoder_inputs = {}

        if get_runtime_state().sp_enabled:
            inputs_embeds = identity_layout(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_runtime_state().sp_group
            )

        if "image" in self.modality:
            inputs_embeds = self.image_forward(inputs_embeds, decoder_inputs, **kwargs)

        if "video" in self.modality:
            inputs_embeds = self.video_forward(inputs_embeds, decoder_inputs, **kwargs)

        if "audio" in self.modality:
            inputs_embeds = self.audio_forward(inputs_embeds, decoder_inputs, **kwargs)

        if get_runtime_state().sp_enabled:
            inputs_embeds = identity_layout(
                inputs_embeds, head_dim=2, seq_dim=1, group=get_runtime_state().sp_group
            )
        return {"inputs_embeds": inputs_embeds, "decoder_inputs": decoder_inputs}


class SeedOmniDecoderModel(SeedOmniPreTrainedModel):
    config_class = SeedOmniDecoderConfig

    def __init__(self, config: SeedOmniDecoderConfig):
        self.config = config
        super().__init__(config)
        torch_dtype = torch.get_default_dtype()
        self.modality = []
        if config.image_config.model_type:
            model_cls = get_model_class(config.image_config)
            self.image_decoder: BaseDecoderModelMixin = model_cls._from_config(
                config.image_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("image")
        if config.video_config.model_type:
            model_cls = get_model_class(config.video_config)
            self.video_decoder: BaseDecoderModelMixin = model_cls._from_config(
                config.video_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("video")
        if config.audio_config.model_type:
            model_cls = get_model_class(config.audio_config)
            self.audio_decoder: BaseDecoderModelMixin = model_cls._from_config(
                config.audio_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("audio")

        self.encode_input = config.encode_input
        self.encode_output = config.encode_output

    def set_projector_trainable_only(self):
        for module in self.children():
            if isinstance(module, BaseDecoderModelMixin):
                module.set_projector_trainable_only()

    def image_encode(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        if self.encode_input:
            input_image_inputs = extract_model_inputs("image_input_", kwargs)
            input_image_mask: torch.Tensor = input_image_inputs.pop("mask", None)
            if input_image_inputs:
                input_image_features, _ = self.image_decoder.lm_encode(**input_image_inputs)
                if get_runtime_state().sp_enabled:
                    input_image_features = identity_layout(
                        input_image_features, seq_dim=0, head_dim=1, group=get_runtime_state().sp_group
                    )
                input_image_features = input_image_features[: input_image_mask.sum()]
                image_mask = input_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                image_features = input_image_features.to(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            elif self.training and _has_trainable_parameters(self.image_decoder):
                dummy_embeds, _ = self.image_decoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0

        if self.encode_output:
            output_image_inputs = extract_model_inputs("image_output_", kwargs)
            output_image_mask: torch.Tensor = output_image_inputs.pop("mask", None)
            if output_image_inputs:
                output_image_features, output_image_indices = self.image_decoder.lm_encode(**output_image_inputs)
                if get_runtime_state().sp_enabled:
                    output_image_features = identity_layout(
                        output_image_features, seq_dim=0, head_dim=1, group=get_runtime_state().sp_group
                    )
                output_image_features = output_image_features[: output_image_mask.sum()]
                image_mask = output_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, output_image_features)
                decoder_inputs["image_output_labels"] = output_image_indices
            elif self.training and _has_trainable_parameters(self.image_decoder):
                dummy_embeds, _ = self.image_decoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0
        return inputs_embeds

    def encode(self, inputs_embeds: torch.Tensor, **kwargs):
        """Encodes the output images to foundation input embeds and image indices labels.
        Only support descrete tokenizer encode.
        Returns of decoder['mm_type'].encode:
            - embeds (torch.Tensor(dtype=float32, shape=(batch_size, seq_len, hidden_size)): input_embeds
            - indices (torch.Tensor(dtype=int64, shape=(batch_size, seq_len))): feature code
        """
        decoder_inputs = kwargs.pop("decoder_inputs", {})

        if get_runtime_state().sp_enabled:
            inputs_embeds = identity_layout(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_runtime_state().sp_group
            )

        if "image" in self.modality:
            inputs_embeds = self.image_encode(inputs_embeds, decoder_inputs, **kwargs)

        if get_runtime_state().sp_enabled:
            inputs_embeds = identity_layout(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_runtime_state().sp_group
            )

        return {"inputs_embeds": inputs_embeds, "decoder_inputs": decoder_inputs}

    def image_decode(self, hidden_states: torch.Tensor, decoder_inputs, loss, **kwargs):
        image_output_labels = decoder_inputs.get("image_output_labels", None)
        target_inputs = extract_model_inputs("image_target", kwargs)
        if target_inputs or image_output_labels is not None:
            output_image_inputs = extract_model_inputs("image_output_", kwargs)
            output_image_mask: torch.Tensor = output_image_inputs.pop("mask", None)
            if get_runtime_state().sp_enabled:
                bs = output_image_mask.size(0)
                sp_size = get_runtime_state().sp_size
                sp_rank = get_runtime_state().sp_rank
                sp_chunk_size = output_image_mask.size(-1) // sp_size
                sp_slice_dim = 1

                output_image_mask = F.pad(output_image_mask[..., 1:], (0, 1), value=0)

                labels_shape = [bs] + list(image_output_labels.shape)
                labels_shape[1] = output_image_mask.shape[1]
                labels = torch.zeros(labels_shape).to(image_output_labels)

                gathered_image_output_labels = distF.all_gather(
                    image_output_labels, group=get_runtime_state().sp_group
                )
                gathered_image_output_labels = torch.cat(gathered_image_output_labels, dim=0)
                gathered_image_output_labels = gathered_image_output_labels[: output_image_mask.sum()]

                if len(labels_shape) == 3:
                    output_image_mask = output_image_mask.unsqueeze(-1).expand_as(labels)
                labels[~output_image_mask] = IGNORE_INDEX
                labels = labels.masked_scatter(output_image_mask, gathered_image_output_labels)
                labels = labels.narrow(sp_slice_dim, sp_rank * sp_chunk_size, sp_chunk_size)

                outputs: BaseDecoderOutput = self.image_decoder.lm_head(
                    hidden_states,
                    labels=labels,
                    **target_inputs,
                )
                loss["image_decoder_loss"] = outputs.loss
            else:
                output_image_hs = hidden_states[..., :-1, :][output_image_mask[..., 1:]]
                labels = image_output_labels
                outputs: BaseDecoderOutput = self.image_decoder.lm_head(
                    output_image_hs,
                    labels=labels,
                    **target_inputs,
                )
                loss["image_decoder_loss"] = outputs.loss
        elif self.training:
            dummy_hidden_states = hidden_states[..., -1, :]
            outputs: BaseDecoderOutput = self.image_decoder.lm_head(dummy_hidden_states)
            loss["image_decoder_loss"] = outputs.logits.mean() * 0.0

    @staticmethod
    def _enabled(value) -> bool:
        if value is None:
            return False
        if isinstance(value, torch.Tensor):
            return bool(value.detach().bool().any().item())
        return bool(value)

    def audio_decode(self, hidden_states: torch.Tensor, loss, **kwargs):
        target_speech_token = kwargs.get("target_speech_token")
        speech_token_labels = kwargs.get("speech_token_labels")
        enable_audio_output = self._enabled(kwargs.get("enable_audio_output"))
        if not enable_audio_output and target_speech_token is None and speech_token_labels is None:
            return None
        if get_runtime_state().sp_enabled:
            raise NotImplementedError("CosyVoice3 audio output training does not support sequence parallel yet.")

        outputs: BaseDecoderOutput = self.audio_decoder(
            hidden_states=hidden_states,
            answer_text_mask=kwargs.get("answer_text_mask"),
            answer_text_len=kwargs.get("answer_text_len"),
            answer_token_ids=kwargs.get("input_ids"),
            cosy_text_token=kwargs.get("cosy_text_token"),
            cosy_text_token_len=kwargs.get("cosy_text_token_len"),
            target_speech_token=target_speech_token,
            target_speech_token_len=kwargs.get("target_speech_token_len"),
            speech_token_labels=speech_token_labels,
            speech_token_label_lens=kwargs.get("speech_token_label_lens"),
            enable_audio_output=enable_audio_output,
        )
        if outputs.loss is not None:
            loss["audio_decoder_loss"] = outputs.loss
        return outputs

    def decode(self, hidden_states: torch.Tensor, decoder_inputs: dict = {}, **kwargs):
        loss = {}
        decoder_outputs = {}
        if "image" in self.modality:
            self.image_decode(hidden_states, decoder_inputs, loss, **kwargs)
        if "audio" in self.modality:
            audio_outputs = self.audio_decode(hidden_states, loss, **kwargs)
            if audio_outputs is not None:
                decoder_outputs["audio"] = audio_outputs

        return loss, decoder_outputs

    def lm_embed(self, hidden_states: torch.Tensor, model_type: str = "image", **kwargs):
        if model_type == "image":
            outputs = self.image_decoder.lm_embed(hidden_states, **kwargs)
        else:
            raise NotImplementedError
        return outputs

    def generate(self, hidden_states: torch.Tensor, modal_type: str = "image", **kwargs):
        if modal_type == "image":
            outputs = self.image_decoder.lm_generate(hidden_states, **kwargs)
        elif modal_type == "audio":
            outputs = self.audio_decoder.lm_generate(hidden_states, **kwargs)
        else:
            raise NotImplementedError
        return outputs


class SeedOmniModel(SeedOmniPreTrainedModel, GenerationMixin):
    def __init__(self, config: SeedOmniConfig):
        super().__init__(config)
        self.config = config
        torch_dtype = torch.get_default_dtype()
        model_cls = get_model_class(config.foundation_config)
        self.foundation: BaseFoundationModelMixin = model_cls._from_config(
            config.foundation_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
        )
        self.encoder = SeedOmniEncoderModel._from_config(
            config.encoder_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
        )
        self.decoder = SeedOmniDecoderModel._from_config(
            config.decoder_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
        )

    def get_input_embeddings(self):
        return self.encoder.text_encoder

    def set_input_embeddings(self, value):
        self.encoder.text_encoder = value

    def get_output_embeddings(self):
        return self.foundation.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.foundation.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ):
        model_embeds = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        vocab_size = self.vocab_size
        self.foundation.vocab_size = vocab_size
        self.foundation.config.get_text_config().vocab_size = vocab_size
        self.config.encoder_config.text_config.vocab_size = vocab_size
        return model_embeds

    def get_modality(self):
        input_modality = self.encoder.modality
        output_modality = self.decoder.modality
        return {"input": input_modality, "output": output_modality}

    def get_position_id_func(self):
        """
        func(input_ids=input_ids, **kwargs) -> dict(position_ids=position_ids, **kwargs)
        """
        return self.foundation.position_id_func

    def _get_foundation_forward_keys(self) -> set[str]:
        forward_keys = {
            key
            for key in inspect.signature(self.foundation.forward).parameters.keys()
            if key not in {"self", "kwargs"}
        }
        forward_keys.update(getattr(self.foundation, "forward_extra_keys", ()))
        return forward_keys

    def _build_foundation_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        foundation_forward_keys = self._get_foundation_forward_keys()
        foundation_inputs = {key: value for key, value in inputs.items() if key in foundation_forward_keys}
        position_ids = foundation_inputs.get("position_ids")
        if isinstance(position_ids, torch.Tensor) and torch.is_floating_point(position_ids):
            foundation_inputs["position_ids"] = position_ids.float()

        foundation_inputs["return_dict"] = True
        foundation_inputs["output_hidden_states"] = True
        return foundation_inputs

    def _get_foundation_autocast_dtype(self) -> Optional[torch.dtype]:
        text_config = getattr(getattr(self.foundation, "config", None), "text_config", None)
        dtype = getattr(text_config, "torch_dtype", None) or getattr(text_config, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype if dtype in (torch.float16, torch.bfloat16) else None
        if isinstance(dtype, str):
            return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype)
        return None

    def _foundation_forward(self, foundation_inputs: Dict[str, torch.Tensor]):
        autocast_dtype = self._get_foundation_autocast_dtype()
        if autocast_dtype is None:
            return self.foundation(**foundation_inputs)

        device_type = None
        for value in foundation_inputs.values():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                device_type = value.device.type
                break
        if device_type is None:
            try:
                device_type = next(self.foundation.parameters()).device.type
            except StopIteration:
                return self.foundation(**foundation_inputs)
        if not torch.amp.is_autocast_available(device_type):
            return self.foundation(**foundation_inputs)

        with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
            return self.foundation(**foundation_inputs)

    def get_preserve_fp32_forward_input_keys(self) -> tuple[str, ...]:
        return ()

    def _cast_forward_inputs_for_mixed_precision(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        try:
            compute_dtype = next(self.parameters()).dtype
        except StopIteration:
            return inputs

        if compute_dtype not in (torch.float16, torch.bfloat16):
            return inputs

        preserve_keys = set(self.get_preserve_fp32_forward_input_keys())
        for key, value in list(inputs.items()):
            if key in preserve_keys:
                continue
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value) and value.dtype != compute_dtype:
                inputs[key] = value.to(dtype=compute_dtype)
        return inputs

    def forward(self, **inputs: torch.Tensor):
        inputs = self._cast_forward_inputs_for_mixed_precision(inputs)
        decoder_inputs = {}

        if "inputs_embeds" not in inputs:
            encoder_encodes = self.encoder.forward(**inputs)
            decoder_encodes = self.decoder.encode(**inputs, **encoder_encodes)
            inputs["inputs_embeds"] = decoder_encodes["inputs_embeds"]
            decoder_inputs = decoder_encodes["decoder_inputs"]

        foundation_inputs = self._build_foundation_inputs(inputs)
        outputs = self._foundation_forward(foundation_inputs)

        losses = {}
        if outputs.loss is not None:
            if torch.isnan(outputs.loss):
                outputs.loss = torch.nan_to_num(outputs.loss)
            losses["foundation_loss"] = outputs.loss

        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else None
        decoder_extra_outputs = {}
        if hidden_states is not None:
            decoder_returns = self.decoder.decode(hidden_states=hidden_states, decoder_inputs=decoder_inputs, **inputs)
            if isinstance(decoder_returns, tuple):
                decoder_loss, decoder_extra_outputs = decoder_returns
            else:
                decoder_loss = decoder_returns
            for key, v in decoder_loss.items():
                losses[key] = v

        if losses:
            audio_outputs = decoder_extra_outputs.get("audio") if decoder_extra_outputs else None
            return SeedOmniOutput(
                losses=losses,
                logits=outputs.logits,
                hidden_states=outputs.hidden_states,
                speech_token_logits=getattr(audio_outputs, "logits", None),
                speech_token_ids=getattr(audio_outputs, "speech_token_ids", None),
            )
        return outputs

    def _prepare_image_generation_config(
        self,
        image_start_token: int = None,
        image_end_token: int = None,
        image_token_num: List = None,
        image_parallel_size: int = 16,
        image_classifier_free_guidance: bool = True,
        image_generation_config: dict = {},
        **kwargs,
    ):
        self.image_start_token = image_start_token
        self.image_token_num = image_token_num
        self.image_parallel_size = image_parallel_size
        self.image_classifier_free_guidance = image_classifier_free_guidance
        self.image_generation_config = image_generation_config
        self.image_end_token = image_end_token
        return kwargs

    def _prepare_generation_config(self, *args, force_image_gen: bool = False, **kwargs):
        kwargs = self._prepare_image_generation_config(**kwargs)

        self.force_image_gen = force_image_gen
        if self.force_image_gen:
            self.parallel_size = self.image_parallel_size
            kwargs = self.setup_image_generation(**kwargs)
        else:
            kwargs = self.setup_text_generation(**kwargs)

        return super()._prepare_generation_config(*args, **kwargs)

    def setup_image_generation(self, **kwargs):
        self.gen_type = "image"
        self.tmp_image = []
        self.generated_images_sequence = []
        if hasattr(self.foundation, "get_generation_position_id"):
            self.generation_position_id_map = None
        else:
            self.generation_position_id_map = None

        kwargs["input_ids"] = kwargs["input_ids"].repeat(self.image_parallel_size, 1)
        kwargs["attention_mask"] = kwargs["attention_mask"].repeat(self.image_parallel_size, 1)
        kwargs["position_ids"] = kwargs["position_ids"].repeat_interleave(self.image_parallel_size, dim=0)
        if self.image_classifier_free_guidance:
            if kwargs.get("bos_token_id", None) and kwargs["input_ids"][0][0] == kwargs["bos_token_id"]:
                start_id = 1
            else:
                start_id = 0
            if kwargs["input_ids"][0][-1] == self.image_start_token:
                end_id = -1
            else:
                end_id = 0

            kwargs["input_ids"][1::2, start_id:end_id] = kwargs.get("pad_token_id", 0)
        return kwargs

    def setup_text_generation(self, **kwargs):
        self.gen_type = "text"
        self.generation_position_id_map = None
        return kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional["Cache"] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        foundation_args = set(inspect.signature(self.foundation.prepare_inputs_for_generation).parameters)
        encoder_decoder_inputs = {}

        for key in list(kwargs.keys()):
            encoder_decoder_inputs[key] = kwargs.pop(key)
            if key in foundation_args:
                kwargs[key] = encoder_decoder_inputs.pop(key)

        model_inputs = self.foundation.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )
        if cache_position[0] == 0:
            encoder_encodes = self.encoder(input_ids=input_ids, **encoder_decoder_inputs)
            model_inputs["inputs_embeds"] = encoder_encodes["inputs_embeds"]
            return model_inputs

        model_inputs.pop("position_ids", None)
        if self.gen_type == "text":
            if input_ids[0][-1] == self.image_start_token:
                self.setup_image_generation()
        elif self.gen_type == "image":
            hidden_states = encoder_decoder_inputs["hidden_states"][-1]
            input_embeds, next_tokens = self.decoder.lm_embed(
                hidden_states[:, -1:], model_type="image", **self.image_generation_config
            )
            self.tmp_image.append(next_tokens)
            generated_tokens = len(self.tmp_image)
            model_inputs["inputs_embeds"] = input_embeds
            if self.generation_position_id_map is not None:
                position_ids = self.generation_position_id_map[..., generated_tokens : generated_tokens + 1]
                model_inputs["position_ids"] = position_ids + cache_position[0] - generated_tokens
            if generated_tokens == self.image_token_num:
                tmp_image = torch.cat(self.tmp_image, dim=-1)
                self.generated_images_sequence.append(tmp_image)
                self.tmp_image = []
                self.setup_text_generation()
        else:
            raise NotImplementedError
        return model_inputs

    def generate_multimodal(self, hidden_states, modal_type="image"):
        return self.decoder.generate(hidden_states, modal_type=modal_type, **self.image_generation_config)

    def _validate_model_kwargs(self, model_kwargs):
        pass

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        model_kwargs = self.foundation._update_model_kwargs_for_generation(
            outputs=outputs, model_kwargs=model_kwargs, **kwargs
        )
        model_kwargs["hidden_states"] = outputs.hidden_states
        return model_kwargs

    def _has_unfinished_sequences(self, *args, **kwargs) -> bool:
        if self.gen_type != "text":
            return True
        return super()._has_unfinished_sequences(*args, **kwargs)
