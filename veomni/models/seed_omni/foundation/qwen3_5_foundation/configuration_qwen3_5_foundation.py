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
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from ..base import BaseFoundationConfigMixin


def _get_subconfig_attr(config, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class Qwen35FoundationConfig(BaseFoundationConfigMixin, Qwen3_5Config):
    model_type = "qwen3_5_foundation"

    def __init__(
        self,
        hidden_size=None,
        vocab_size=None,
        tie_word_embeddings=None,
        **kwargs,
    ):
        text_config = kwargs.get("text_config")
        if hidden_size is None:
            hidden_size = _get_subconfig_attr(text_config, "hidden_size")
        if vocab_size is None:
            vocab_size = _get_subconfig_attr(text_config, "vocab_size")
        if tie_word_embeddings is None:
            tie_word_embeddings = kwargs.get("tie_word_embeddings", False)

        super().__init__(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
