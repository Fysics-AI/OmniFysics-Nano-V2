from typing import Dict

import torch
import torch.nn as nn

from ....transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import Qwen3_5VisionModel
from ...projector import build_feature_projector
from ..base import BaseEncoderModelMixin
from .configuration_qwen3_5_vision_model import Qwen35VisionModelConfig


class Qwen35VisionModel(BaseEncoderModelMixin, Qwen3_5VisionModel):
    config_class = Qwen35VisionModelConfig
    _no_split_modules = ["Qwen3_5VisionBlock"]
    _checkpoint_conversion_mapping = {"^model\\.visual\\.": ""}

    def __init__(self, config: Qwen35VisionModelConfig):
        super().__init__(config)
        self.config = config
        if config.add_projector and config.output_size is not None:
            self.projector = build_feature_projector(config.out_hidden_size, config.output_size)
        else:
            if config.output_size and config.output_size != config.out_hidden_size:
                raise ValueError("`output_size` should be same as `out_hidden_size`.")

            self.projector = nn.Identity()

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        if self.config.add_projector and self.config.output_size is not None:
            self.projector.requires_grad_(True)
            if self.config.train_origin_projector:
                self.merger.requires_grad_(True)
        else:
            self.merger.requires_grad_(True)

    def lm_encode(self, features: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        vision_outputs = super().forward(features, grid_thw, return_dict=True, **kwargs)
        if self.config.return_hidden_states:
            return vision_outputs.last_hidden_state
        return self.projector(vision_outputs.pooler_output)

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        pixel_values = torch.randn((4, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
        return {"features": pixel_values, "grid_thw": grid_thw}
