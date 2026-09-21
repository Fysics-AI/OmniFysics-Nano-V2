# Copyright 2024-2025 The Black-forest-labs Authors. All rights reserved.
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
from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen3_5_foundation")
def register_qwen3_5_foundation_config():
    from transformers import AutoTokenizer, Qwen2TokenizerFast

    from .configuration_qwen3_5_foundation import Qwen35FoundationConfig

    # HF Qwen3.x continues to reuse the Qwen2 tokenizer classes.
    # Multimodal processor registration still needs to be added separately.
    AutoTokenizer.register(Qwen35FoundationConfig, fast_tokenizer_class=Qwen2TokenizerFast)
    return Qwen35FoundationConfig


@MODELING_REGISTRY.register("qwen3_5_foundation")
def register_qwen3_5_foundation_modeling(architecture: str):
    from .modeling_qwen3_5_foundation import Qwen35FoundationModel

    return Qwen35FoundationModel
