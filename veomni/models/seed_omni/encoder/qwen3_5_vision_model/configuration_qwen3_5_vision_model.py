from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from ..base import BaseEncoderConfigMixin


class Qwen35VisionModelConfig(BaseEncoderConfigMixin, Qwen3_5VisionConfig):
    model_type = "qwen3_5_vision_model"

    def __init__(
        self,
        return_hidden_states=False,
        train_origin_projector=False,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.return_hidden_states = return_hidden_states
        self.train_origin_projector = train_origin_projector
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
