import inspect
from typing import List, Optional

import numpy as np
import torch
from transformers import BatchFeature, WhisperFeatureExtractor


try:
    from transformers.audio_utils import AudioInput
except Exception:
    from transformers.tokenization_utils_base import AudioInput

from ..base import BaseEncoderProcessorMixin


class WhisperAudioModelProcessor(BaseEncoderProcessorMixin, WhisperFeatureExtractor):
    valid_kwargs = BaseEncoderProcessorMixin.valid_kwargs + list(
        inspect.signature(WhisperFeatureExtractor.__init__).parameters.keys()
    )

    def __init__(
        self,
        token_num: int = None,
        token_size: List = None,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        dither: float = 0.0,
        return_attention_mask: bool = False,
        **kwargs,
    ) -> None:
        BaseEncoderProcessorMixin.__init__(self, token_num=token_num, token_size=token_size, **kwargs)
        WhisperFeatureExtractor.__init__(
            self,
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            chunk_length=chunk_length,
            n_fft=n_fft,
            padding_value=padding_value,
            dither=dither,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

    @property
    def model_input_names(self):
        return WhisperFeatureExtractor.model_input_names

    def process(
        self,
        audios: Optional[AudioInput] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        if audios is None:
            raise ValueError("`audios` must be provided.")

        audios = [audio if audio is not None else np.zeros((0,), dtype=np.float32) for audio in audios]
        output = WhisperFeatureExtractor.__call__(
            self,
            audios,
            return_tensors=return_tensors,
            return_attention_mask=True,
            padding=kwargs.pop("padding", "max_length"),
            sampling_rate=kwargs.pop("sampling_rate", self.sampling_rate),
            **kwargs,
        )
        features = output["input_features"]
        attention_mask = output["attention_mask"]
        feature_lengths = attention_mask.sum(-1).to(torch.int32)
        num_tokens = torch.where(
            feature_lengths > 0,
            ((feature_lengths - 1) // 2 + 1).to(torch.int32),
            torch.zeros_like(feature_lengths),
        )
        return BatchFeature(
            data={"features": features, "num_tokens": num_tokens, "feature_lengths": feature_lengths},
            tensor_type=return_tensors,
        )
