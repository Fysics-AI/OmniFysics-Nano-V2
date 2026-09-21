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


import glob
import json
import os
from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch

from ...utils.constants import IGNORE_INDEX, TYPE2INDEX
from ...utils.import_utils import is_video_audio_available
from .image_utils import fetch_images
from .preprocess import PREPROCESSOR_REGISTRY, conv_preprocess, get_full_data_path, omnimodal_preprocess


if is_video_audio_available():
    from .audio_utils import fetch_audios
    from .video_utils import fetch_videos
else:

    def fetch_videos(*args, **kwargs):
        return [], []

    def fetch_audios(*args, **kwargs):
        return []


if TYPE_CHECKING:
    from ...models.seed_omni import SeedOmniProcessor
    from .multimodal_chat_template import MultimodalChatTemplate


INVALID_MEDIA_PLACEHOLDERS = {
    "",
    ".",
    "./",
    "images",
    "images/",
    "audios",
    "audios/",
    "audio",
    "audio/",
    "videos",
    "videos/",
}

EARLY_TEXT_LENGTH_FILTER_SOURCES = {
    "finevision__tablet_small",
    "finevision__doclaynet_v1_2",
}


class RawTextOverlongSampleError(ValueError):
    def __init__(self, source: str, sample_length: int, max_seq_len: int):
        super().__init__(
            f"{source} raw text token length {sample_length} exceeds max_seq_len {max_seq_len}, "
            "skipping before media decode."
        )
        self.source = source
        self.sample_length = sample_length
        self.max_seq_len = max_seq_len
        self.skip_reason = "raw_text_overlong_sample"


@lru_cache(maxsize=16)
def _load_speech_token_map(path: str):
    if not path:
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


@lru_cache(maxsize=200000)
def _resolve_gigaspeech_audio_path(path: str) -> str:
    """Resolve GigaSpeech basename-only audio paths to their chunked wav files.

    GigaSpeech parquet rows store values like ``POD0000006459_S0000400.wav``.
    In some environments ``Ready/audios`` does not contain a flat copy of every
    segment wav, while the original chunked files still exist under
    ``data/audio/<subset>/<chunk_dir>/<segment>.wav``. A two-level glob is fast
    enough here and avoids a full recursive scan of the 7M+ audio tree.
    """
    if not isinstance(path, str) or os.path.isabs(path):
        return path

    hits = glob.glob(os.path.join("/share/Audio/GigaSpeech/data/audio", "*", "*", path))
    return hits[0] if hits else path


def _to_list(value):
    normalized_values = []

    def _append(item):
        if item is None:
            return

        if isinstance(item, (list, tuple)):
            for sub_item in item:
                _append(sub_item)
            return

        if isinstance(item, str):
            stripped_item = item.strip()
            if not stripped_item:
                return

            if stripped_item.startswith("[") and stripped_item.endswith("]"):
                try:
                    parsed_item = json.loads(stripped_item)
                except (json.JSONDecodeError, TypeError):
                    parsed_item = None
                if isinstance(parsed_item, list):
                    _append(parsed_item)
                    return

            if stripped_item in INVALID_MEDIA_PLACEHOLDERS:
                return

            normalized_values.append(stripped_item)
            return

        normalized_values.append(item)

    _append(value)
    return normalized_values


def _filter_invalid_media_paths(paths: List[Any]) -> List[Any]:
    filtered_paths = []
    for path in paths:
        if not isinstance(path, str):
            filtered_paths.append(path)
            continue

        stripped_path = path.strip()
        if not stripped_path or stripped_path in INVALID_MEDIA_PLACEHOLDERS:
            continue

        if stripped_path.startswith("http://") or stripped_path.startswith("https://"):
            filtered_paths.append(stripped_path)
            continue

        if os.path.isdir(stripped_path):
            continue

        filtered_paths.append(stripped_path)

    return filtered_paths


def _is_video_frame_bytes_sequence(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(item, bytes) for item in value)


def _get_images_binary_video_inputs(sample: Dict[str, Any]) -> List[Any]:
    images_binary = sample.get("images_binary")
    if _is_video_frame_bytes_sequence(images_binary):
        return [images_binary]
    return []


def _count_multimodal_placeholders(conversations: List[Any]) -> Dict[str, int]:
    counts = {"image": 0, "video": 0, "audio": 0}
    for conversation in conversations:
        for message in conversation[1:]:
            if message and message[0] in counts:
                counts[message[0]] += 1
    return counts


def _collapse_image_placeholders_to_video(conversations: List[Any]) -> List[Any]:
    collapsed_conversations = []
    video_inserted = False
    for conversation in conversations:
        collapsed = [conversation[0]]
        for message in conversation[1:]:
            if message[0] == "image":
                if not video_inserted:
                    collapsed.append(("video", None))
                    video_inserted = True
                continue
            collapsed.append(message)
        if len(collapsed) == 1:
            collapsed.append(("text", ""))
        collapsed_conversations.append(collapsed)
    return collapsed_conversations


def _preprocess_conversations(source: str, conversations: Any, sample: Dict[str, Any], **kwargs) -> List[Any]:
    if source in PREPROCESSOR_REGISTRY.valid_keys():
        processed_conversations = conv_preprocess(source, conversations, **kwargs)
    else:
        processed_conversations = omnimodal_preprocess(conversations, **kwargs)

    if _get_images_binary_video_inputs(sample):
        placeholder_counts = _count_multimodal_placeholders(processed_conversations)
        if placeholder_counts["video"] == 0 and placeholder_counts["image"] > 0:
            processed_conversations = _collapse_image_placeholders_to_video(processed_conversations)

    return processed_conversations


def _token_length(tokenizer: Any, text: Any) -> int:
    if text is None:
        return 0
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    return len(token_ids)


def _conversation_text_content(conversation: Any) -> str:
    if not isinstance(conversation, (list, tuple)) or len(conversation) <= 1:
        return ""

    text_parts = []
    for message in conversation[1:]:
        if not isinstance(message, (list, tuple)) or len(message) < 2:
            continue
        if message[0] == "text" and message[1] is not None:
            text_parts.append(str(message[1]))
    return "".join(text_parts).strip()


def _get_system_message(chat_template: "MultimodalChatTemplate") -> Dict[str, Any] | None:
    get_system_message = getattr(chat_template, "_get_system_mesage", None)
    if not callable(get_system_message):
        return None
    system_message = get_system_message()
    return system_message if isinstance(system_message, dict) else None


def _text_only_chat_token_length(
    conversations: List[Any],
    chat_template: "MultimodalChatTemplate",
) -> int | None:
    tokenizer = getattr(chat_template, "tokenizer", None)
    if tokenizer is None:
        return None

    start_token_cache = {}
    end_token_len = _token_length(tokenizer, "<|im_end|>\n")

    def message_length(role: str, content: str) -> int:
        if role not in start_token_cache:
            start_token_cache[role] = _token_length(tokenizer, f"<|im_start|>{role}\n")

        length = start_token_cache[role]
        if content:
            length += _token_length(tokenizer, content)
            length += end_token_len
        return length

    total_length = 0
    system_message = _get_system_message(chat_template)
    if system_message is not None:
        total_length += message_length(
            str(system_message.get("role", "system")),
            str(system_message.get("content", "")).strip(),
        )

    for conversation in conversations:
        if not isinstance(conversation, (list, tuple)) or not conversation:
            continue
        total_length += message_length(str(conversation[0]), _conversation_text_content(conversation))

    return total_length


def _filter_early_text_overlong_sample_if_needed(
    source: str,
    conversations: List[Any],
    chat_template: "MultimodalChatTemplate",
    max_seq_len: Any,
) -> None:
    if source not in EARLY_TEXT_LENGTH_FILTER_SOURCES or max_seq_len is None:
        return

    try:
        max_seq_len = int(max_seq_len)
    except (TypeError, ValueError):
        return

    if max_seq_len <= 0:
        return

    sample_length = _text_only_chat_token_length(conversations, chat_template)
    if sample_length is not None and sample_length > max_seq_len:
        raise RawTextOverlongSampleError(source, sample_length, max_seq_len)


def _get_sample_utt(sample: Dict[str, Any]):
    for key in ("utt", "utt_id", "id", "sample_id"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return None


def _get_pre_tokenized_speech(sample: Dict[str, Any], **kwargs):
    for key in (
        "target_speech_token",
        "target_speech_tokens",
        "speech_token",
        "speech_tokens",
        "speech_token_ids",
    ):
        if key in sample and sample[key] is not None:
            return sample[key]

    token_map_path = (
        kwargs.get("speech_token_map_path") or kwargs.get("utt2speech_token_path") or kwargs.get("utt2speech_token")
    )
    token_map = _load_speech_token_map(token_map_path) if token_map_path else None
    utt = _get_sample_utt(sample)
    if token_map is not None and utt is not None:
        return token_map.get(utt)
    return None


def _normalize_speech_token_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().long().reshape(-1)
    else:
        tensor = torch.tensor(value, dtype=torch.long).reshape(-1)
    return tensor


def _resolve_with_source_path(paths: List[Any], source_path: str) -> List[Any]:
    if not source_path:
        return paths

    normalized_source_path = source_path.rstrip(os.sep)
    if os.path.isfile(normalized_source_path):
        normalized_source_path = os.path.dirname(normalized_source_path)

    source_parent = os.path.dirname(normalized_source_path)
    source_basename = os.path.basename(normalized_source_path)
    if source_basename.startswith("parquet") or source_basename.endswith(("parquet", "parquets")):
        base_candidates = [normalized_source_path, source_parent]
    else:
        base_candidates = [normalized_source_path]

    resolved = []
    for path in paths:
        if not isinstance(path, str) or os.path.isabs(path):
            resolved.append(path)
            continue

        candidate_paths = [path]
        for base_path in base_candidates:
            candidate_paths.extend(
                [
                    os.path.join(base_path, path),
                    os.path.join(base_path, "images", path),
                    os.path.join(base_path, "audios", path),
                    os.path.join(base_path, "audio", path),
                    os.path.join(base_path, "videos", path),
                ]
            )

        for candidate_path in candidate_paths:
            if os.path.isfile(candidate_path):
                resolved.append(candidate_path)
                break
        else:
            resolved.append(path)

    return resolved


def _resolve_media_paths(source: str, source_path: str, paths: List[Any]) -> List[Any]:
    if not paths:
        return paths

    normalized_paths = _to_list(paths)
    resolved_paths = get_full_data_path(source, normalized_paths)
    resolved_paths = _resolve_with_source_path(resolved_paths, source_path)
    if source == "GigaSpeech":
        resolved_paths = [_resolve_gigaspeech_audio_path(path) for path in resolved_paths]
    return _filter_invalid_media_paths(resolved_paths)


def mask_before_position_id_func(input_ids: torch.Tensor):
    """Mask special multimodal tokens in input_ids to input_mm_token for position_id.
    Only supports special image tokens now. (input_image_id=-200, output_image_id=-201->-200)
    Similar to veomni.module.seed_omni.modeling_seed_omni.mask_before_text_encoder

    Args:
        input_ids (torch.Tensor)

    Returns:
        input_ids (torch.Tensor)
    """
    for modality in ["image", "video", "audio"]:
        output_mask = input_ids == TYPE2INDEX["output"][modality]
        input_mask = input_ids == TYPE2INDEX["input"][modality]
        input_ids = torch.where(output_mask | input_mask, TYPE2INDEX["input"][modality], input_ids)
    return input_ids


def mask_input_ids(modality_info: Dict, input_ids: torch.Tensor):
    """Mask special multimodal tokens in input_ids to 0 for text_encoder.word_embedding.
    And return masks including: image_input_mask, image_output_mask, etc
    For example:
        input_ids:                  torch.tensor([-200, -200,   2,  -200,   -200,   4,  5,  6,  -201,   -201])
        Returns:
            input_ids:              torch.tensor([0,    0,      2,  0,      0,      4,  5,  6,  0,      0   ])
            image_input_mask:       torch.tensor([1,    1,      0,  1,      1,      0,  0,  0,  0,      0   ])
            image_output_mask:      torch.tensor([0,    0,      0,  0,      0,      0,  0,  0,  1,      1   ])

    Args:
        input_ids (torch.Tensor)

    Returns:
        input_ids (torch.Tensor)
        mask_dict (Dict) : {modal}_[input/output]_mask.
    """
    mask_dict = {}
    for data_type in modality_info.keys():
        for modal in modality_info[data_type]:
            mask = input_ids == TYPE2INDEX[data_type][modal]
            mask_dict[f"{modal}_{data_type}_mask"] = mask
            input_ids = torch.where(mask, 0, input_ids)
    return input_ids, mask_dict


def process_mm_data(
    conversations, images: List[Any], videos: List[Any], video_audios: List[Any], audio_audios: List[Any]
):
    """
    Processes multi-modal conversation data and aligns images, videos, and audio
    with a corresponding output mask indicating whether the data was produced by the assistant.

    Parameters:
    ----------
    conversations : List[List]
    images : List[Any], List of image data in order.
    videos : List[Any], List of video data in order.
    video_audios : List[Any], List of audio tracks corresponding to the videos.
    audio_audios : List[Any], List of standalone audio samples.

    Returns:
    -------
    conv_images : List[Any], List of images in the order they appeared in conversations.
    conv_videos : List[Any], List of videos in the order they appeared in conversations.
    conv_audios : List[Any], List of all audio data, including both video audio and standalone audio.

    mask : Dict[str, torch.BoolTensor]
        A dictionary with modality names as keys ("image", "video", "audio"), and boolean tensors
        indicating whether each sample was produced by the assistant (True) or the user (False).

    Example:
    --------
    Input:
        conversations = [
            ["user", ["video"], ["audio"], ["video"], ["text"]],
            ["assistant", ["audio"]]
        ]
        videos = ["video1", "video2"]
        video_audios = ["v_audio1", "v_audio2"]
        audio_audios = ["audio1", "audio2"]

    Output:
        conv_videos = ["video1", "video2"]
        conv_audios = ["v_audio1", "audio1", "v_audio2", "audio2"]
        mask["video"] = tensor([False, False])              # user videos
        mask["audio"] = tensor([False, False, False, True]) # user+assistant audios
    """
    images, videos, video_audios, audio_audios = iter(images), iter(videos), iter(video_audios), iter(audio_audios)
    conv_images, conv_videos, conv_audios = [], [], []
    mask = defaultdict(list)
    for conversation in conversations:
        role = conversation[0]
        is_output = (
            role == "assistant"
        )
        for message in conversation[1:]:
            data_type = message[0]
            if data_type == "text":
                continue
            elif data_type == "image":
                conv_images.append(next(images))
                mask["image"].append(is_output)
            elif data_type == "video":
                conv_videos.append(next(videos))
                conv_audios.append(next(video_audios))
                mask["video"].append(is_output)
                mask["audio"].append(is_output)
            elif data_type == "audio":
                conv_audios.append(next(audio_audios))
                mask["audio"].append(is_output)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
    mask = {key: torch.tensor(value).type(torch.bool) for key, value in mask.items()}
    return conv_images, conv_videos, conv_audios, mask


def get_multimodal_configs(modality_input: Dict, multimodal_output_mask: Dict):
    def _empty_tensor_config(reference: torch.Tensor, length: int) -> torch.Tensor:
        return reference.new_empty((length, *reference.shape[1:]))

    def _normalize_sequence_config(config: Any) -> List[Any]:
        if config is None:
            return []
        if isinstance(config, list):
            return config
        if isinstance(config, tuple):
            return list(config)
        return [config]

    def _merge_config_by_mask(
        reference_config: Any,
        input_config: Any,
        output_config: Any,
        mm_mask: torch.Tensor,
        config_key: str,
        modal: str,
    ):
        input_count = int((~mm_mask).sum().item())
        output_count = int(mm_mask.sum().item())
        total_count = int(mm_mask.shape[0])

        if isinstance(reference_config, torch.Tensor):
            input_config = (
                _empty_tensor_config(reference_config, input_count) if input_config is None else input_config
            )
            output_config = (
                _empty_tensor_config(reference_config, output_count) if output_config is None else output_config
            )

            if input_config.shape[0] != input_count:
                raise ValueError(
                    f"Mismatched {modal}_{config_key} input config length: expected {input_count}, "
                    f"got {input_config.shape[0]}."
                )
            if output_config.shape[0] != output_count:
                raise ValueError(
                    f"Mismatched {modal}_{config_key} output config length: expected {output_count}, "
                    f"got {output_config.shape[0]}."
                )

            merged_config = reference_config.new_empty((total_count, *reference_config.shape[1:]))
            if output_count:
                merged_config[mm_mask] = output_config
            if input_count:
                merged_config[~mm_mask] = input_config
            return merged_config

        input_items = _normalize_sequence_config(input_config)
        output_items = _normalize_sequence_config(output_config)

        if len(input_items) != input_count:
            raise ValueError(
                f"Mismatched {modal}_{config_key} input config length: expected {input_count}, got {len(input_items)}."
            )
        if len(output_items) != output_count:
            raise ValueError(
                f"Mismatched {modal}_{config_key} output config length: expected {output_count}, "
                f"got {len(output_items)}."
            )

        merged_config = []
        input_index, output_index = 0, 0
        for is_output in mm_mask.tolist():
            if is_output:
                merged_config.append(output_items[output_index])
                output_index += 1
            else:
                merged_config.append(input_items[input_index])
                input_index += 1
        return merged_config

    multimodal_configs, config_repr = {}, {}
    for key in modality_input.keys():
        config_key = key.split("_", 2)[-1]
        if config_key != "features":
            config_repr[config_key] = modality_input[key]
    for config_key, reference_config in config_repr.items():
        modal_configs = {}
        merged_sequence_config = None
        for modal, mm_mask in multimodal_output_mask.items():
            if (
                f"{modal}_input_{config_key}" not in modality_input
                and f"{modal}_output_{config_key}" not in modality_input
            ):
                continue
            input_config = modality_input.get(f"{modal}_input_{config_key}")
            output_config = modality_input.get(f"{modal}_output_{config_key}")
            current_reference = reference_config
            if input_config is not None:
                current_reference = input_config
            elif output_config is not None:
                current_reference = output_config

            config = _merge_config_by_mask(
                current_reference,
                input_config,
                output_config,
                mm_mask,
                config_key,
                modal,
            )

            if config_key == "video_metadata" and modal == "video":
                merged_sequence_config = config
            else:
                modal_configs[modal] = config
        if modal_configs:
            multimodal_configs[config_key] = modal_configs
        elif merged_sequence_config is not None:
            multimodal_configs[config_key] = merged_sequence_config
    return multimodal_configs


def keep_input_only(multimodal_config: Dict, multimodal_output_mask: Dict):
    """Only keep the input data in multimodal_config. Used when use_special_rope=False.
    When use_special_rope=False, only do special_rope on input_multimodal_data.
    For example: 2d_rope on input_image_token, but 1d_rope on output_image_token.
    """
    for config in multimodal_config.keys():
        if not isinstance(multimodal_config[config], dict):
            continue
        for modal in multimodal_config[config].keys():
            multimodal_config[config][modal] = multimodal_config[config][modal][~multimodal_output_mask[modal]]


def build_position_id_extra_kwargs(
    chat_template: "MultimodalChatTemplate", multimodal_config: Dict[str, Any]
) -> Dict[str, Any]:
    extra_kwargs = {}
    if getattr(chat_template, "audio_bos_id", None) is not None:
        extra_kwargs["audio_start_token_id"] = chat_template.audio_bos_id
    if getattr(chat_template, "audio_eos_id", None) is not None:
        extra_kwargs["audio_end_token_id"] = chat_template.audio_eos_id
    if getattr(chat_template, "position_id_per_seconds", None) is not None:
        extra_kwargs["position_id_per_seconds"] = chat_template.position_id_per_seconds
    if getattr(chat_template, "audio_position_scale", None) is not None:
        extra_kwargs["audio_position_scale"] = chat_template.audio_position_scale
    video_metadata = multimodal_config.get("video_metadata", [])
    if video_metadata:
        temporal_patch_size = getattr(chat_template, "video_temporal_patch_size", 2)
        second_per_grids = []
        for metadata in video_metadata:
            if isinstance(metadata, dict):
                fps = metadata.get("fps", 2.0)
            else:
                fps = metadata.fps if getattr(metadata, "fps", None) is not None else 2.0
            second_per_grids.append(temporal_patch_size / fps)
        extra_kwargs["video_second_per_grid"] = torch.tensor(second_per_grids, dtype=torch.float32)
    return extra_kwargs


def encode_multimodal_sample(
    sample: Dict[str, Any],
    processor: "SeedOmniProcessor",
    chat_template: "MultimodalChatTemplate",
    position_id_func: "Callable",
    modality_info: Dict,
    use_special_rope=False,  # 2d rope position id for image generation
    **kwargs,
) -> Dict[str, List[int]]:
    model_inputs = {}
    source = sample.get("source") or sample.get("source_name") or kwargs.get("source_name")
    sample_source_path = sample.get("source_path")
    modality = set(modality_info["input"] + modality_info["output"])
    if source == "fineweb_100BT" and "text" in sample:
        conversations = sample["text"]
    else:
        conversations = sample["conversations"] if ("conversations" in sample and sample["conversations"]) else sample
    if isinstance(conversations, bytes):
        conversations = json.loads(conversations.decode("utf-8"))
    conversations = _preprocess_conversations(source, conversations, sample, **kwargs)
    _filter_early_text_overlong_sample_if_needed(source, conversations, chat_template, kwargs.get("max_seq_len"))
    processor_input = {}

    image_paths = _to_list(sample.get("images")) or _to_list(sample.get("image")) or _to_list(sample.get("image_path"))
    image_paths = _resolve_media_paths(source, sample_source_path, image_paths)

    video_paths = _to_list(sample.get("videos")) or _to_list(sample.get("video")) or _to_list(sample.get("video_path"))
    video_paths = _resolve_media_paths(source, sample_source_path, video_paths)
    video_paths = video_paths or _get_images_binary_video_inputs(sample)

    audio_paths = _to_list(sample.get("audios")) or _to_list(sample.get("audio")) or _to_list(sample.get("audio_path"))
    audio_paths = _resolve_media_paths(source, sample_source_path, audio_paths)

    if "image" in modality:
        images = fetch_images(image_paths, **kwargs)
    else:
        images = []
    if "video" in modality:
        videos, video_audios = fetch_videos(video_paths, **kwargs)
        if "audio" not in modality:
            video_audios = [None] * len(videos)
    else:
        videos, video_audios = [], []
    if "audio" in modality:
        audio_audios = fetch_audios(audio_paths, **kwargs)
    else:
        audio_audios = []

    images, videos, audios, multimodal_output_mask = process_mm_data(
        conversations, images, videos, video_audios, audio_audios
    )

    if images:
        processor_input.update(
            {
                "input_images": [img for img, mask in zip(images, multimodal_output_mask["image"]) if not mask],
                "output_images": [img for img, mask in zip(images, multimodal_output_mask["image"]) if mask],
            }
        )
    if videos:
        processor_input.update(
            {
                "input_videos": [vid for vid, mask in zip(videos, multimodal_output_mask["video"]) if not mask],
                "output_videos": [img for img, mask in zip(videos, multimodal_output_mask["video"]) if mask],
            }
        )
    if audios and "audio" in modality:
        processor_input.update(
            {
                "input_audios": [aud for aud, mask in zip(audios, multimodal_output_mask["audio"]) if not mask],
                "output_audios": [aud for aud, mask in zip(audios, multimodal_output_mask["audio"]) if mask],
            }
        )

    processor_kwargs = {"return_tensors": "pt", **processor_input}
    if processor_input.get("input_videos"):
        processor_kwargs["return_metadata"] = True

    modality_input = processor(**processor_kwargs)
    multimodal_config = get_multimodal_configs(modality_input, multimodal_output_mask)
    enable_audio_output = bool(kwargs.get("enable_audio_output", False))
    speech_tokens = _get_pre_tokenized_speech(sample, **kwargs)
    if speech_tokens is not None:
        enable_audio_output = True

    chat_template_kwargs = dict(multimodal_config)
    if enable_audio_output:
        chat_template_kwargs["return_answer_text_mask"] = True
    text_inputs = chat_template.encode_messages(conversations, **chat_template_kwargs)
    model_inputs.update(modality_input)
    model_inputs.update(text_inputs)

    if enable_audio_output:
        answer_text_mask = model_inputs.get("answer_text_mask")
        if answer_text_mask is None:
            answer_text_mask = model_inputs["labels"] != IGNORE_INDEX
            model_inputs["answer_text_mask"] = answer_text_mask
        model_inputs["answer_text_len"] = answer_text_mask.long().sum().reshape(1)
        model_inputs["enable_audio_output"] = torch.tensor(True, dtype=torch.bool)

    if speech_tokens is not None:
        speech_token_tensor = _normalize_speech_token_tensor(speech_tokens)
        model_inputs["target_speech_token"] = speech_token_tensor
        model_inputs["target_speech_token_len"] = torch.tensor([speech_token_tensor.numel()], dtype=torch.long)
        utt = _get_sample_utt(sample)
        if utt is not None:
            model_inputs["utt"] = utt

    # position_ids (dim, len)
    if position_id_func is None:  # default position_ids
        position_ids = torch.arange(0, len(text_inputs["input_ids"])).unsqueeze(0)
    else:  # customized position_ids
        input_ids = text_inputs["input_ids"].clone()
        attention_mask = text_inputs["attention_mask"].clone()
        if use_special_rope:
            input_ids = mask_before_position_id_func(input_ids)
        else:
            keep_input_only(multimodal_config, multimodal_output_mask)
        position_id_extra_kwargs = build_position_id_extra_kwargs(chat_template, multimodal_config)
        position_ids = position_id_func(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            **multimodal_config,
            **position_id_extra_kwargs,
        )["position_ids"]
    model_inputs["position_ids"] = position_ids

    input_ids, mask_dict = mask_input_ids(modality_info, model_inputs["input_ids"])
    model_inputs["input_ids"] = input_ids
    model_inputs.update(mask_dict)
    return [model_inputs]


def encode_multimodal_sample_inference(
    sample: Dict[str, Any],
    processor: "SeedOmniProcessor",
    chat_template: "MultimodalChatTemplate",
    position_id_func: "Callable",
    modality_info: Dict,
    force_image_gen: bool,
    **kwargs,
):
    model_inputs = {}
    modality = set(modality_info["input"] + modality_info["output"])
    conversations = sample["conversations"]

    processor_input = {}
    if "image" in modality:
        images = fetch_images(sample.get("images", []), **kwargs)
    else:
        images = []
    if "video" in modality:
        videos, video_audios = fetch_videos(sample.get("videos", []), **kwargs)
        if "audio" not in modality:
            video_audios = [None] * len(videos)
    else:
        videos, video_audios = [], []
    if "audio" in modality:
        audio_audios = fetch_audios(sample.get("audios", []), **kwargs)
    else:
        audio_audios = []

    images, videos, audios, multimodal_output_mask = process_mm_data(
        conversations, images, videos, video_audios, audio_audios
    )

    if images:
        processor_input["input_images"] = images
    if videos:
        processor_input["input_videos"] = videos
    if audios and "audio" in modality:
        processor_input["input_audios"] = audios

    processor_kwargs = {"return_tensors": "pt", **processor_input}
    if processor_input.get("input_videos"):
        processor_kwargs["return_metadata"] = True

    modality_input = processor(**processor_kwargs)
    multimodal_config = get_multimodal_configs(modality_input, multimodal_output_mask)
    chat_template_kwargs = dict(multimodal_config)
    if "enable_thinking" in kwargs:
        chat_template_kwargs["enable_thinking"] = kwargs["enable_thinking"]
    if "thinking_prompt" in kwargs:
        chat_template_kwargs["thinking_prompt"] = kwargs["thinking_prompt"]
    text_inputs = chat_template.encode_messages(conversations, **chat_template_kwargs)

    if force_image_gen:
        text_inputs["input_ids"] = torch.cat(
            [text_inputs["input_ids"], torch.tensor([chat_template.image_start_id])],
            dim=-1,
        )
        text_inputs["attention_mask"] = torch.cat([text_inputs["attention_mask"], torch.tensor([1])], dim=-1)

    model_inputs.update(modality_input)
    model_inputs.update(text_inputs)

    # position_ids (dim, len)
    if position_id_func is None:  # default position_ids
        position_id_returns = {"position_ids": torch.arange(0, len(text_inputs["input_ids"])).unsqueeze(0)}
    else:  # customized position_ids
        input_ids = text_inputs["input_ids"].clone()
        attention_mask = text_inputs["attention_mask"].clone()
        position_id_extra_kwargs = build_position_id_extra_kwargs(chat_template, multimodal_config)
        position_id_returns = position_id_func(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            **multimodal_config,
            **position_id_extra_kwargs,
        )

    model_inputs.update(position_id_returns)

    input_ids, mask_dict = mask_input_ids(modality_info, model_inputs["input_ids"])
    model_inputs["input_ids"] = input_ids
    model_inputs.update(mask_dict)
    return [model_inputs]
