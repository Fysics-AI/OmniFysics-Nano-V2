from typing import Optional

from transformers.models.whisper.configuration_whisper import WhisperConfig

from ..base import BaseEncoderConfigMixin


class WhisperAudioModelConfig(BaseEncoderConfigMixin, WhisperConfig):
    model_type = "whisper"

    def __init__(
        self,
        add_projector: Optional[bool] = None,
        return_hidden_states: bool = False,
        **kwargs,
    ):
        self.return_hidden_states = return_hidden_states
        super().__init__(add_projector=False if add_projector is None else add_projector, **kwargs)
        if add_projector is None and self.output_size is not None:
            self.add_projector = self.output_size != self.d_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        add_projector = kwargs.get("add_projector")
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if add_projector is None and config.output_size is not None:
            config.add_projector = config.output_size != config.d_model
        return config
