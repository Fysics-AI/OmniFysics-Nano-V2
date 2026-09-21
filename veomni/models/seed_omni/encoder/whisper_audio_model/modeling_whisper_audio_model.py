from typing import Dict

import torch
import torch.nn as nn
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from ...projector import build_feature_projector
from ..base import BaseEncoderModelMixin
from .configuration_whisper_audio_model import WhisperAudioModelConfig


class WhisperAudioModel(BaseEncoderModelMixin, WhisperEncoder):
    config_class = WhisperAudioModelConfig
    _no_split_modules = ["WhisperEncoderLayer"]
    _checkpoint_conversion_mapping = {"^model\\.encoder\\.": ""}

    def __init__(self, config: WhisperAudioModelConfig):
        super().__init__(config)
        self.config = config
        if config.add_projector and config.output_size is not None:
            self.projector = build_feature_projector(config.d_model, config.output_size)
        else:
            if config.output_size is not None and config.output_size != config.d_model:
                raise ValueError(
                    "`output_size` does not match Whisper hidden size. "
                    "Set `add_projector: true` (or leave it unset to auto-enable)."
                )
            self.projector = nn.Identity()

    @property
    def output_dim(self) -> int:
        if self.config.return_hidden_states:
            return self.config.d_model
        if isinstance(self.projector, nn.Sequential):
            return self.projector[-1].out_features
        return self.config.d_model

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        if self.config.add_projector and self.config.output_size is not None:
            self.projector.requires_grad_(True)
        else:
            raise ValueError(
                "WhisperAudioModel has no native alignment head to keep trainable. "
                "Use `add_projector: true` when `freeze_encoder` is enabled."
            )

    @staticmethod
    def _get_output_lengths(feature_lengths: torch.Tensor) -> torch.Tensor:
        return (feature_lengths - 1) // 2 + 1

    def lm_encode(self, features: torch.Tensor, feature_lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        if feature_lengths.ndim > 1:
            feature_lengths = feature_lengths.squeeze(-1)

        valid_mask = feature_lengths > 0
        if not torch.any(valid_mask):
            return features.new_empty((0, self.output_dim))

        features = features[valid_mask]
        feature_lengths = feature_lengths[valid_mask]

        hidden_states = super().forward(input_features=features, return_dict=True, **kwargs).last_hidden_state
        hidden_states = hidden_states if self.config.return_hidden_states else self.projector(hidden_states)

        output_lengths = self._get_output_lengths(feature_lengths).clamp(max=hidden_states.shape[1]).tolist()
        valid_hidden_states = [hidden_states[i, :seq_len] for i, seq_len in enumerate(output_lengths) if seq_len > 0]
        if not valid_hidden_states:
            return hidden_states.new_empty((0, self.output_dim))
        return torch.cat(valid_hidden_states, dim=0)

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        features = torch.zeros((1, self.config.num_mel_bins, self.config.max_source_positions * 2), dtype=self.dtype)
        features = features.to(self.device)
        feature_lengths = torch.tensor([self.config.max_source_positions * 2], dtype=torch.int64, device=self.device)
        return {"features": features, "feature_lengths": feature_lengths}
