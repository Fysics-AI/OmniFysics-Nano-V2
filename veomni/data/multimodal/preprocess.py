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

"""
Dataset Preprocessors

This module contains both built-in and custom dataset preprocessors.
All preprocessors are registered using the @register_preprocessor decorator.

To add custom preprocessors, simply define a function and decorate it with @register_preprocessor.
"""

import json
import os
import random
import re
from typing import List

from ...utils.registry import Registry


PREPROCESSOR_REGISTRY = Registry("preprocessor")


def conv_preprocess(source: str, conversations, **kwargs):
    if source is None:
        raise ValueError("source must not be None when preprocessing conversations.")

    if source.startswith("honey_") or source.startswith("finevision__") or source == "bytedance_data":
        return honey_preprocess(conversations, **kwargs)

    if source in DATASETS:
        return DATASETS[source](conversations, **kwargs)

    return PREPROCESSOR_REGISTRY[source](conversations, **kwargs)


# ============================================================================
# Built-in Dataset Preprocessors
# ============================================================================


@PREPROCESSOR_REGISTRY.register("sharegpt4v_pretrain")
@PREPROCESSOR_REGISTRY.register("sharegpt4v_captioner")
def sharegpt4v_pretrain_preprocess(conversations, generation_ratio=0.0, **kwargs):
    constructed_conversation = []
    if conversations[0]["from"] != "human":  # Skip the first one if it is not from human
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    for message in conversations:
        role = message["from"]
        value = message["value"]
        if role == "human":
            value = value.replace("<image>", "")
            constructed_conversation.append(["user", ("image", None)])
        else:
            constructed_conversation.append(["assistant", ("text", value)])
    generate_sample = random.random() < generation_ratio
    if generate_sample:
        instruction = f"Generate an image based on the following caption: {constructed_conversation[-1][0][1]}"
        constructed_conversation = [["user", ("text", instruction)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("sharegpt4v_captioner_sft")
@PREPROCESSOR_REGISTRY.register("sharegpt4v_sft")
def sharegpt4v_sft_preprocess(conversations, **kwargs):
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed_conversation = []
    if conversations[0]["from"] != "human":  # Skip the first one if it is not from human
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    for message in conversations:
        value = message["value"]
        role = role_mapping[message["from"]]
        if "<image>" in value:
            value = value.replace("<image>", "")
            constructed_conversation.append([role, ("image", None), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("doom")
def doom_preprocess(conversations, max_image_nums=None, **kwargs):
    """
    merge the assistant output in a single message
    """
    constructed_conversation = []
    image_count = 0
    role_mapping = {"human": "user", "gpt": "assistant"}
    prev_conversation = []
    prev_role = "user"
    for i, message in enumerate(conversations):
        role = role_mapping[message["from"]]
        value = message["value"]
        if i == 0:
            value = value.strip()
        if value == "<image>":
            cur_message = [("image", None)]
            image_count += 1
        else:
            cur_message = [("text", value)]
        if role == prev_role == "assistant":
            cur_message = [("text", "\n\n")] + cur_message
            prev_conversation += cur_message
        elif role == prev_role:
            prev_conversation += cur_message
        else:
            constructed_conversation.append([prev_role] + prev_conversation)
            prev_role = role
            prev_conversation = cur_message
        if max_image_nums is not None and image_count >= max_image_nums:
            break
    if len(prev_conversation) != 0:
        constructed_conversation.append([prev_role] + prev_conversation)
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("seed_edit")
def seed_edit_preprocess(conversations, **kwargs):
    constructed_conversation = []
    for message in conversations:
        value = message["value"]
        parts = value.split("<image>")
        if parts == ["", ""]:  # "<image>"
            cur_message = ["assistant", ("image", None)]
        else:
            cur_message = ["user"]
            for part in parts:
                if part == "":
                    cur_message += [("image", None)]
                else:
                    cur_message += [("text", part), ("image", None)]
            cur_message = cur_message[:-1]
        constructed_conversation.append(cur_message)
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("imagenet1k")
def imagenet1k_preprocess(conversations, **kwargs):
    class_labels = [item.strip() for item in conversations.split(",")]
    class_label = random.choice(class_labels)
    constructed_conversation = [
        ["user", ("text", class_label)],
        ["assistant", ("image", None)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("imagenet1k_caption")
def imagenet1k_caption_preprocess(conversations, **kwargs):
    class_labels = [item.strip() for item in conversations.split(",")]
    class_label = random.choice(class_labels)
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image.")],
        ["assistant", ("text", class_label)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("fineweb_100BT")
def fineweb_preprocess(conversations, **kwargs):
    conversations = conversations["text"]
    constructed_conversation = [
        ["assistant", ("text", conversations)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("wikihow_ct_0904")
def wikihow_preprocess(conversations, stage="pretrain", **kwargs):
    constructed_conversation = []
    role_mapping = {"human": "user", "gpt": "assistant"}
    for conv in conversations:
        role = role_mapping[conv["from"]]
        value = conv["value"]
        cur_message = [role]
        if "<image>" in value:
            value = value.replace("<image>", "").strip()
            cur_message.append(("image", None))
            if value != "":
                cur_message.append(("text", value))
        else:
            cur_message.append(("text", value))
        constructed_conversation.append(cur_message)
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("Detailed_Caption")
def detailed_caption_preprocess(conversations, **kwargs):
    constructed_conversation = []
    assert conversations[-1]["from"] == "gpt"
    caption = conversations[-1]["value"][8:].strip()  # skip Answer:
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("ArxivQA")
def arxivqa_preprocess(conversations, **kwargs):
    question = conversations[0]["value"].replace("<image>\n", "").strip()
    answer = conversations[1]["value"].strip()
    constructed_conversation = [["user", ("image", None), ("text", question)], ["assistant", ("text", answer)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("pixelprose")
def pixelprose_preprocess(conversations, **kwargs):
    caption = conversations
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("DenseFusion-1M")
@PREPROCESSOR_REGISTRY.register("DenseFusion-4V-100k")
def densefusion_preprocess(conversations, **kwargs):
    caption = conversations[0]["value"]
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("sam")
def sam_preprocess(conversations, **kwargs):
    caption = conversations
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("sam_gen")
def sam_gen_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("pixelprose_gen")
def pixelprose_gen_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("chart_to_table")
def chart_to_table_preprocess(conversations, **kwargs):
    caption = conversations
    constructed_conversation = [
        ["user", ("image", None), ("text", "Convert the image to a table.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("CHartQA")
def chartqa_preprocess(conversations, **kwargs):
    question = conversations[0]["value"].replace("<image>\n", "").strip()
    answer = conversations[1]["value"].strip()
    constructed_conversation = [["user", ("image", None), ("text", question)], ["assistant", ("text", answer)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("megalith")
def megalith_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("journeydb")
def journeydb_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("dalle3_1m")
def dalle3_1m_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("wit")
def wit_preprocess(conversations, **kwargs):
    text_content_1, text_content_2, text_content_3 = "", "", ""
    if conversations["page_title"]:
        text_content_1 += conversations["page_title"] + "\n"
    if conversations["context_page_description"]:
        text_content_2 += conversations["context_page_description"] + "\n"
    if conversations["caption_reference_description"]:
        text_content_3 += conversations["caption_reference_description"]

    constructed_conversation = [
        ["user", ("text", text_content_1)],
        ["assistant", ("text", text_content_2)],
        ["user", ("image", None)],
        ["assistant", ("text", text_content_3)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("mmsci")
def mmsci_preprocess(conversations, **kwargs):
    caption = conversations[0]["value"]

    def replace_figure_number(text):
        return re.sub(r"^(Figure|Fig\.) \d+[:]*", "", text)

    caption = replace_figure_number(caption).strip()
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("LLaVA-Video-178K")
def llava_video_preprocess(conversations, **kwargs):
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed_conversation = []
    if conversations[0]["from"] != "human":  # Skip the first one if it is not from human
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    for message in conversations:
        value = message["value"]
        role = role_mapping[message["from"]]
        if "<image>" in value:
            value = value.replace("<image>\n", "")
            constructed_conversation.append([role, ("video", None), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("VoiceAssistant")
def voice_assistant_preprocess(conversations, **kwargs):
    constructed_conversation = [
        ["user", ("audio", None)],
        ["assistant", ("text", conversations[1]["value"])],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("tulu-3-sft-mixture")
def tulu_3_sft_mixture_preprocess(conversations, **kwargs):
    text_example = conversations["messages"]
    constructed_conversation = []
    for conversation in text_example:
        constructed_conversation.append([conversation["role"], ("text", conversation["content"])])
    return constructed_conversation


def text_chat_sft_preprocess(conversations, merge_consecutive_assistant=True, **kwargs):
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()
    if not conversations:
        return []

    constructed_conversation = []
    role_mapping = {"human": "user", "gpt": "assistant"}

    for message in conversations:
        role = message.get("from", "")
        value = message.get("value", "")
        if not value or not role or role not in role_mapping:
            continue

        mapped_role = role_mapping[role]
        if isinstance(value, str):
            value = value.strip()

        cur_message = [mapped_role, ("text", value)]
        if merge_consecutive_assistant and constructed_conversation:
            prev_message = constructed_conversation[-1]
            if prev_message[0] == "assistant" and cur_message[0] == "assistant":
                prev_text = prev_message[1][1]
                current_text = cur_message[1][1]
                constructed_conversation[-1][1] = ("text", f"{prev_text}\n\n{current_text}")
                continue

        constructed_conversation.append(cur_message)

    return constructed_conversation


def audio_preprocess(conversations, **kwargs):
    if not conversations:
        return []

    transformed_conversations = []
    audio_inserted = False

    for message in conversations:
        role = message.get("from")
        content = message.get("value", "")

        if role == "human":
            cur_message = ["user"]
            if not audio_inserted:
                cur_message.append(("audio", None))
                cur_message.append(("text", content))
                audio_inserted = True
            else:
                cur_message.append(("text", content))
            transformed_conversations.append(cur_message)
        elif role == "gpt":
            transformed_conversations.append(["assistant", ("text", content)])

    return transformed_conversations


def honey_preprocess(conversations, **kwargs):
    role_mapping = {"human": "user", "gpt": "assistant"}
    if not conversations:
        return []

    constructed_conversation = []

    start_idx = 0
    for i, msg in enumerate(conversations):
        if msg.get("from") == "human":
            start_idx = i
            break
    else:
        return []

    for message in conversations[start_idx:]:
        if message.get("from") not in role_mapping or "from" not in message:
            continue

        value = message.get("value", "")
        role = role_mapping[message["from"]]

        if value is None or value == "":
            constructed_conversation.append([role, ("text", "")])
            continue

        if not isinstance(value, str):
            value = str(value)

        value = value.replace("<image>\n", "<image>")
        parts = value.split("<image>")

        cur_message = [role]
        if len(parts) == 1:
            cur_message.append(("text", value.strip()))
        else:
            for idx, part in enumerate(parts):
                cleaned_part = part.strip()
                if cleaned_part:
                    cur_message.append(("text", cleaned_part))
                if idx < len(parts) - 1:
                    cur_message.append(("image", None))
        constructed_conversation.append(cur_message)

    return constructed_conversation


def omnimodal_preprocess(conversations, **kwargs):
    role_mapping = {"human": "user", "gpt": "assistant"}
    if not conversations:
        return []

    constructed_conversation = []

    start_idx = 0
    for i, msg in enumerate(conversations):
        if msg.get("from") == "human":
            start_idx = i
            break
    else:
        return []

    for message in conversations[start_idx:]:
        if message.get("from") not in role_mapping or "from" not in message:
            continue

        value = message.get("value", "")
        role = role_mapping[message["from"]]

        if value is None or value == "":
            constructed_conversation.append([role, ("text", "")])
            continue

        if not isinstance(value, str):
            value = str(value)

        value = value.replace("<image>\n", "<image>")
        value = value.replace("<audio>\n", "<audio>")
        value = value.replace("<video>\n", "<video>")

        cur_message = [role]
        pattern = r"(<image>|<audio>|<video>)"
        parts = re.split(pattern, value)
        current_text = ""

        for part in parts:
            if part == "<image>":
                if current_text.strip():
                    cur_message.append(("text", current_text.strip()))
                    current_text = ""
                cur_message.append(("image", None))
            elif part == "<audio>":
                if current_text.strip():
                    cur_message.append(("text", current_text.strip()))
                    current_text = ""
                cur_message.append(("audio", None))
            elif part == "<video>":
                if current_text.strip():
                    cur_message.append(("text", current_text.strip()))
                    current_text = ""
                cur_message.append(("video", None))
            else:
                current_text += part

        if current_text.strip():
            cur_message.append(("text", current_text.strip()))

        if len(cur_message) == 1:
            cur_message.append(("text", value.strip()))

        constructed_conversation.append(cur_message)

    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("lccc_cosyvoice3")
@PREPROCESSOR_REGISTRY.register("LCCC_CosyVoice3")
def lccc_cosyvoice3_preprocess(conversations, **kwargs):
    if isinstance(conversations, bytes):
        conversations = conversations.decode("utf-8")
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()

    constructed_conversation = []
    role_mapping = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

    for message in conversations:
        if isinstance(message, dict):
            role = role_mapping.get(message.get("from") or message.get("role"))
            content = message.get("value", message.get("content", ""))
            if role is None:
                continue
            constructed_conversation.append([role, ("text", "" if content is None else str(content))])
            continue

        if not isinstance(message, (list, tuple)) or not message:
            continue

        role = role_mapping.get(message[0])
        if role is None:
            continue

        cur_message = [role]
        for value in message[1:]:
            if isinstance(value, dict):
                value_type = value.get("type") or value.get("modality", "text")
                value_content = value.get("value", value.get("text", value.get("content", "")))
            elif isinstance(value, (list, tuple)) and value:
                value_type = value[0]
                value_content = value[1] if len(value) > 1 else ""
            else:
                value_type = "text"
                value_content = value

            if value_type == "text":
                cur_message.append(("text", "" if value_content is None else str(value_content)))
            elif value_type in {"image", "video", "audio"}:
                cur_message.append((value_type, value_content))
            else:
                raise ValueError(f"Unknown LCCC CosyVoice3 message value type: {value_type}")

        if len(cur_message) == 1:
            cur_message.append(("text", ""))
        constructed_conversation.append(cur_message)

    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("mminfinity_cosyvoice3")
@PREPROCESSOR_REGISTRY.register("mminfinity_cosyvoice3_image")
def mminfinity_cosyvoice3_preprocess(conversations, **kwargs):
    if isinstance(conversations, bytes):
        conversations = conversations.decode("utf-8")
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()

    constructed_conversation = []
    role_mapping = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

    for message in conversations:
        if isinstance(message, dict):
            role = role_mapping.get(message.get("from") or message.get("role"))
            content = message.get("value", message.get("content", ""))
            if role is None:
                continue
            value = "" if content is None else str(content)
            value = value.replace("<image>\n", "<image>")
            value = value.replace("<audio>\n", "<audio>")
            value = value.replace("<video>\n", "<video>")

            cur_message = [role]
            parts = re.split(r"(<image>|<audio>|<video>)", value)
            current_text = ""
            for part in parts:
                if part == "<image>":
                    if current_text.strip():
                        cur_message.append(("text", current_text.strip()))
                        current_text = ""
                    cur_message.append(("image", None))
                elif part == "<audio>":
                    if current_text.strip():
                        cur_message.append(("text", current_text.strip()))
                        current_text = ""
                    cur_message.append(("audio", None))
                elif part == "<video>":
                    if current_text.strip():
                        cur_message.append(("text", current_text.strip()))
                        current_text = ""
                    cur_message.append(("video", None))
                else:
                    current_text += part

            if current_text.strip():
                cur_message.append(("text", current_text.strip()))
            if len(cur_message) == 1:
                cur_message.append(("text", ""))
            constructed_conversation.append(cur_message)
            continue

        if not isinstance(message, (list, tuple)) or not message:
            continue

        role = role_mapping.get(message[0])
        if role is None:
            continue

        cur_message = [role]
        for value in message[1:]:
            if isinstance(value, dict):
                value_type = value.get("type") or value.get("modality", "text")
                value_content = value.get("value", value.get("text", value.get("content", "")))
            elif isinstance(value, (list, tuple)) and value:
                value_type = value[0]
                value_content = value[1] if len(value) > 1 else ""
            else:
                value_type = "text"
                value_content = value

            if value_type == "text":
                cur_message.append(("text", "" if value_content is None else str(value_content)))
            elif value_type in {"image", "video", "audio"}:
                cur_message.append((value_type, value_content))
            else:
                raise ValueError(f"Unknown MMinfinity CosyVoice3 image message value type: {value_type}")

        if len(cur_message) == 1:
            cur_message.append(("text", ""))
        constructed_conversation.append(cur_message)

    return constructed_conversation


LEGACY_DATASETS = {
    "OpenHermes-2.5": text_chat_sft_preprocess,
    "OpenHermes-2.5-zh": text_chat_sft_preprocess,
    "OmniFysics-Identity": text_chat_sft_preprocess,
    "LibriSpeech": audio_preprocess,
    "AudioUnderstanding": audio_preprocess,
    "PeoplesSpeech": audio_preprocess,
    "PeoplesSpeech_SA": audio_preprocess,
    "GigaSpeech": audio_preprocess,
    "CommonVoice": audio_preprocess,
    "MMAU": audio_preprocess,
    "MMAR": audio_preprocess,
    "AI2D": honey_preprocess,
    "hallusionbench": honey_preprocess,
    "mathvista": honey_preprocess,
    "MMBench_v1.1": honey_preprocess,
    "MMMU": honey_preprocess,
    "MMStar": honey_preprocess,
    "mm-vet": honey_preprocess,
    "ocrbench": honey_preprocess,
    "openimage-physical": honey_preprocess,
    "mixed_tts": omnimodal_preprocess,
    "MusicAQA": omnimodal_preprocess,
    "MusicQA": omnimodal_preprocess,
    "audio_openhermes": omnimodal_preprocess,
    "audio_openhermes_zh": omnimodal_preprocess,
    "voiceassistant-400k": omnimodal_preprocess,
    "AV-Odyssey-Bench": omnimodal_preprocess,
    "Daily-Omni": omnimodal_preprocess,
    "OmniBench": omnimodal_preprocess,
    "WorldSense": omnimodal_preprocess,
    "MME": omnimodal_preprocess,
    "Video-MME": omnimodal_preprocess,
    "qkx_video_clip_train_caption": omnimodal_preprocess,
    "PhysBench": omnimodal_preprocess,
    "PAI-Bench-understanding": omnimodal_preprocess,
    "QuantiPhy": omnimodal_preprocess,
    "PhysUniBench": omnimodal_preprocess,
    "FysicsEval": omnimodal_preprocess,
    "FysicsWorld": omnimodal_preprocess,
}


for _source_name, _preprocessor in LEGACY_DATASETS.items():
    if _source_name not in PREPROCESSOR_REGISTRY.valid_keys():
        PREPROCESSOR_REGISTRY.register(_source_name, _preprocessor)


DATASETS = {name: PREPROCESSOR_REGISTRY[name] for name in PREPROCESSOR_REGISTRY.valid_keys()}
DATASETS.update(LEGACY_DATASETS)


DATASET_BASE_PATHS = {
    "LibriSpeech": "/share/Audio/LibriSpeech/Ready",
    "PeoplesSpeech": "/share/Audio/peoples_speech/Ready/clean",
    "PeoplesSpeech_SA": "/share/Audio/peoples_speech/Ready/clean_sa",
    "GigaSpeech": "/share/Audio/GigaSpeech/Ready",
    "CommonVoice": "/share/Audio/common_voice_15_0/Ready",
    "MMAU": "/share/Audio/MMAU/Ready",
    "MMAR": "/share/Audio/MMAR/Ready",
    "mixed_tts": "/share/Audio/processed_parquet/mixed_tts",
    "MusicAQA": "/share/Audio/processed_parquet/MUSIC-AVQA",
    "MusicQA": "/share/Audio/processed_parquet/MusicQA",
    "audio_openhermes": "/share/Audio/processed_parquet/openhermes",
    "audio_openhermes_zh": "/share/Audio/processed_parquet/openhermes-zh",
    "voiceassistant-400k": "/share/Audio/processed_parquet/voiceassistant",
    "qkx_video_clip_train_caption": "/share/Video/qkx_video_clip/clip_train",
    "AV-Odyssey-Bench": "/share/Omni-Data/Omni-Benchmark/AV-Odyssey-Bench",
    "Daily-Omni": "/share/Omni-Data/Omni-Benchmark/Daily-Omni",
    "OmniBench": "/share/Omni-Data/Omni-Benchmark/OmniBench",
    "WorldSense": "/share/Omni-Data/Omni-Benchmark/WorldSense",
    "OpenHermes-2.5": "/share/T2I/processed_parquet/OpenHermes-2.5",
    "OpenHermes-2.5-zh": "/share/T2I/processed_parquet/OpenHermes-2.5-zh",
    "OmniFysics-Identity": "/share/T2I/processed_parquet/OmniFysics-Identity",
    "AI2D": "/share/T2I/processed_parquet/ai2d",
    "hallusionbench": "/share/T2I/processed_parquet/hallusionbench",
    "mathvista": "/share/T2I/processed_parquet/mathvista",
    "MMBench_v1.1": "/share/T2I/processed_parquet/mmbench",
    "MMMU": "/share/T2I/processed_parquet/mmmu",
    "MMStar": "/share/T2I/processed_parquet/mmstar",
    "mm-vet": "/share/T2I/processed_parquet/mmvet",
    "ocrbench": "/share/T2I/processed_parquet/ocrbench",
    "PhysBench": "/share/Physics-Benchmark/PhysBench/Ready",
    "PAI-Bench-understanding": "/share/Physics-Benchmark/PAI-Bench-understanding/Ready",
    "QuantiPhy": "/share/Physics-Benchmark/QuantiPhy/Ready",
    "PhysUniBench": "/share/Physics-Benchmark/PhysUniBench/Ready",
    "FysicsEval": "/share/Physics-Benchmark/FysicsEval/Ready",
    "FysicsWorld": "/share/jy/FysicsWorld_data/media_merge",
}


def get_dataset_base_path(source: str) -> str:
    return DATASET_BASE_PATHS.get(source, "")


def get_full_data_path(source: str, relative_paths: List[str]) -> List[str]:
    base_path = get_dataset_base_path(source)
    if base_path:
        resolved_paths = []
        for path in relative_paths:
            if not isinstance(path, str) or os.path.isabs(path):
                resolved_paths.append(path)
                continue
            candidate_path = os.path.join(base_path, path)
            resolved_paths.append(candidate_path if os.path.exists(candidate_path) else path)
        return resolved_paths
    return relative_paths


# @PREPROCESSOR_REGISTRY.register("your_dataset_name")
# def your_dataset_preprocess(conversations, **kwargs):
#     ...
