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


import random
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Sequence

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from ...utils import logging
from ...utils.constants import IGNORE_INDEX, TYPE2INDEX
from ..chat_template import ChatmlTemplate, ChatTemplate


logger = logging.get_logger(__name__)


class MultimodalChatTemplate(ChatTemplate):
    @abstractmethod
    def encode_messages(
        self, messages: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        """
        Encodes messages to a dictionary of input_ids, attention_mask, labels, and mm with mm_seqlens.
        """

    def get_jinja_template(self) -> str:
        return ""


class Qwen2VLTemplate(MultimodalChatTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer)
        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.image_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        self.bos = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)

        logger.info_rank0("Qwen2VLTemplate will not truncate sequence when longer than [max_seq_lens].")

        self.cfg_ratio = kwargs.get("cfg_ratio", None)

    @property
    def _unconditioned_generation(self):
        return self.cfg_ratio and random.random() < self.cfg_ratio

    def image_pattern(self, token_num):
        return "<|vision_start|>" + self.image_pad * token_num + "<|vision_end|>"

    def video_pattern(self, token_num):
        return "<|vision_start|>" + self.video_pad * token_num + "<|vision_end|>"

    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]], **kwargs) -> Dict[str, List[int]]:
        pass


class Qwen2VLPretrainTemplate(Qwen2VLTemplate):
    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        messages = []
        data_type = ""
        mm_num_tokens = {key: iter(item) for key, item in num_tokens.items()}
        for message in conversations:
            role = message[0]
            content = ""
            for item in message[1:]:
                mm_type = item[0]
                if mm_type == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    content += self.image_pattern(next(mm_num_tokens[mm_type]))
                elif mm_type == "video":
                    content += self.video_pattern(next(mm_num_tokens[mm_type]))
                else:
                    content += item[1]
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        input_ids, attention_mask, labels = [], [], []
        input_ids += self.bos
        attention_mask += [1] * len(self.bos)
        labels += self.bos
        for message in messages:
            content_str = message["content"].strip()
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            loss_mask = message["loss_mask"]
            if content_str == "":
                break
            if role == "user" and data_type == "t2i" and self._unconditioned_generation:
                input_ids += [self.tokenizer.pad_token_id] * len(content_ids)
            else:
                input_ids += content_ids

            attention_mask += [1] * len(content_ids)
            if loss_mask == 1:
                labels += content_ids
                input_ids += self.eos
                attention_mask += [1] * len(self.eos)
                labels += self.eos
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX

        if data_type == "t2i":
            labels = tokenized_example["labels"]
            labels[~output_image_mask] = IGNORE_INDEX
            tokenized_example["labels"] = labels
        return tokenized_example


class Qwen2VLChatTemplate(Qwen2VLTemplate):
    system_prompt = "You are a helpful assistant."

    def _get_system_mesage(self):
        system_message = {
            "role": "system",
            "content": self.system_prompt,
            "loss_mask": 0,
        }
        return system_message

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        sys_msg = self._get_system_mesage()
        messages = [] if sys_msg is None else [sys_msg]
        data_type = ""
        image_token_num_list = iter(num_tokens.pop("image", []))
        video_token_num_list = iter(num_tokens.pop("video", []))
        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    content += self.image_pattern(next(image_token_num_list))
                elif value[0] == "video":
                    content += self.video_pattern(next(video_token_num_list))
                else:
                    raise ValueError(f"Unknown value type: {value[0]}")
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["content"].strip()
            loss_mask = message["loss_mask"]
            role = message["role"]
            message_ids = self.tokenizer.encode("<|im_start|>" + message["role"] + "\n", add_special_tokens=False)

            if content_str:
                end_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                if (
                    role == "user" and data_type == "t2i" and self._unconditioned_generation
                ):
                    message_ids += [self.tokenizer.pad_token_id] * len(content_ids) + end_ids
                else:
                    message_ids += content_ids + end_ids

            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            if loss_mask == 1:
                labels += message_ids
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX
        if data_type == "t2i":
            labels = tokenized_example["labels"]
            labels[labels == self.image_start_id] = IGNORE_INDEX
            tokenized_example["labels"] = labels

        return tokenized_example


class Qwen3VLChatTemplate(Qwen2VLTemplate):
    system_prompt = "You are a helpful assistant."

    MERGE_SIZE = 2

    def _get_system_mesage(self):
        system_message = {
            "role": "system",
            "content": self.system_prompt,
            "loss_mask": 0,
        }
        return system_message

    def _calculate_timestamps(self, indices: List[int], video_fps: float, merge_size: int = 2):
        """
        Replicates Qwen3-VL official logic: Pad -> Convert to Seconds -> Average.
        """
        if len(indices) % merge_size != 0:
            indices.extend([indices[-1]] * (merge_size - len(indices) % merge_size))

        timestamps = [idx / video_fps for idx in indices]

        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps


    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        sys_msg = self._get_system_mesage()
        messages = [] if sys_msg is None else [sys_msg]
        data_type = ""
        image_token_num_list = iter(num_tokens.pop("image", []))
        video_token_num_list = iter(num_tokens.pop("video", []))

        video_metadata_list = iter(kwargs.get("video_metadata", []))

        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    content += self.image_pattern(next(image_token_num_list))

                elif value[0] == "video":
                    try:
                        total_video_tokens = next(video_token_num_list)
                    except StopIteration:
                        raise ValueError("Video token number is missing for a video input.")

                    try:
                        v_meta = next(video_metadata_list)
                    except StopIteration:
                        raise ValueError("Video metadata is missing for a video input.")

                    fps = v_meta.fps if v_meta.fps is not None else 2.0

                    if hasattr(v_meta, "frames_indices") and v_meta.frames_indices is not None:
                        indices = v_meta.frames_indices
                        if hasattr(indices, "tolist"):
                            indices = indices.tolist()
                        elif not isinstance(indices, list):
                            indices = list(indices)
                    else:
                        total_frames = v_meta.total_num_frames if v_meta.total_num_frames is not None else 16
                        indices = list(range(total_frames))

                    timestamps = self._calculate_timestamps(indices, fps, merge_size=self.MERGE_SIZE)

                    num_time_chunks = len(timestamps)

                    if num_time_chunks > 0:
                        tokens_per_chunk = total_video_tokens // num_time_chunks
                    else:
                        tokens_per_chunk = 0

                    video_str_buffer = ""
                    for t_val in timestamps:
                        video_str_buffer += f"<{float(t_val):.1f} seconds>"
                        video_str_buffer += "<|vision_start|>"
                        video_str_buffer += "<|video_pad|>" * tokens_per_chunk
                        video_str_buffer += "<|vision_end|>"

                    content += video_str_buffer

                else:
                    raise ValueError(f"Unknown value type: {value[0]}")

            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["content"].strip()
            loss_mask = message["loss_mask"]
            role = message["role"]
            message_ids = self.tokenizer.encode("<|im_start|>" + message["role"] + "\n", add_special_tokens=False)

            if content_str:
                end_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)

                if role == "user" and data_type == "t2i" and self._unconditioned_generation:
                    message_ids += [self.tokenizer.pad_token_id] * len(content_ids) + end_ids
                else:
                    message_ids += content_ids + end_ids

            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            if loss_mask == 1:
                labels += message_ids
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX

        if data_type == "t2i":
            labels = tokenized_example["labels"]
            labels[labels == self.image_start_id] = IGNORE_INDEX
            tokenized_example["labels"] = labels

        return tokenized_example


class Qwen35OmniChatTemplate(Qwen3VLChatTemplate):
    system_prompt = (
        "你是 OmniFysics-Plus，由飞捷科思智能科技研发的多模态 AI 助手。"
        "你能够理解和分析图像、音频、文本和视频输入，并根据用户需求生成清晰、准确、诚实的文本或图像回答。"
    )

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer, **kwargs)
        self.audio_bos_token = "<|audio_start|>"
        self.audio_eos_token = "<|audio_end|>"
        self.audio_pad = "<|audio_pad|>"

        add_token_num = self.tokenizer.add_tokens([self.audio_bos_token, self.audio_eos_token])
        add_token_num += self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_pad]})

        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)
        self.trained_embedding = []
        if add_token_num > 0:
            self.trained_embedding = [self.audio_bos_id, self.audio_eos_id]

        self.position_id_per_seconds = kwargs.get("position_id_per_seconds", 25)
        self.audio_tokens_per_second = kwargs.get("audio_tokens_per_second", 50)
        if self.audio_tokens_per_second <= 0:
            raise ValueError("audio_tokens_per_second must be positive.")
        self.audio_position_scale = kwargs.get(
            "audio_position_scale", self.position_id_per_seconds / self.audio_tokens_per_second
        )
        self.video_temporal_patch_size = kwargs.get("video_temporal_patch_size", 2)

        logger.info_rank0("Qwen35OmniTemplate will not truncate sequence when longer than [max_seq_lens].")

    def audio_pattern(self, token_num):
        return self.audio_bos_token + self.audio_pad * int(token_num) + self.audio_eos_token

    def _build_video_chunk(self, token_num: int, timestamp: float) -> str:
        return f"<{float(timestamp):.1f} seconds><|vision_start|>" + self.video_pad * int(token_num) + "<|vision_end|>"

    def _get_video_frame_indices(self, video_metadata) -> tuple[list[int], float]:
        fps = video_metadata.fps if video_metadata.fps is not None else 2.0

        if hasattr(video_metadata, "frames_indices") and video_metadata.frames_indices is not None:
            frame_indices = video_metadata.frames_indices
            if hasattr(frame_indices, "tolist"):
                frame_indices = frame_indices.tolist()
            elif not isinstance(frame_indices, list):
                frame_indices = list(frame_indices)
        else:
            total_frames = video_metadata.total_num_frames if video_metadata.total_num_frames is not None else 16
            frame_indices = list(range(total_frames))

        if len(frame_indices) % self.MERGE_SIZE != 0:
            frame_indices.extend([frame_indices[-1]] * (self.MERGE_SIZE - len(frame_indices) % self.MERGE_SIZE))
        return frame_indices, fps

    def _get_video_chunks(self, total_video_tokens: int, video_metadata) -> List[tuple[float, int, float]]:
        frame_indices, fps = self._get_video_frame_indices(video_metadata)
        timestamps = self._calculate_timestamps(frame_indices, fps, merge_size=self.MERGE_SIZE)
        if not timestamps:
            return []

        total_video_tokens = int(total_video_tokens)
        tokens_per_chunk, remainder = divmod(total_video_tokens, len(timestamps))
        chunk_lengths = [tokens_per_chunk] * len(timestamps)
        if remainder:
            chunk_lengths[-1] += remainder
        chunk_start_timestamps = [frame_indices[i] / fps for i in range(0, len(frame_indices), self.MERGE_SIZE)]
        return [
            (display_timestamp, token_num, align_timestamp)
            for display_timestamp, token_num, align_timestamp in zip(
                timestamps, chunk_lengths, chunk_start_timestamps, strict=True
            )
        ]

    def _build_video_pattern(self, total_video_tokens: int, video_metadata) -> str:
        return "".join(
            self._build_video_chunk(token_num, display_timestamp)
            for display_timestamp, token_num, _ in self._get_video_chunks(total_video_tokens, video_metadata)
        )

    def _build_token_interleaved_video_audio_pattern(
        self,
        total_video_tokens: int,
        audio_token_num: int,
        curr_video_grid_thw: torch.Tensor,
        video_metadata,
    ) -> str:
        audio_token_num = int(audio_token_num)
        if audio_token_num <= 0:
            return self._build_video_pattern(total_video_tokens, video_metadata)

        fps = video_metadata.fps if video_metadata.fps is not None else 2.0
        second_per_grid = self.video_temporal_patch_size / fps

        merge_size = int(torch.sqrt(curr_video_grid_thw.prod() // total_video_tokens).item())
        height = (curr_video_grid_thw[1] // merge_size).item()
        width = (curr_video_grid_thw[2] // merge_size).item()
        video_token_indices = torch.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
        video_token_indices = video_token_indices.expand(-1, height, width).reshape(-1).float()
        video_token_indices = video_token_indices * second_per_grid * self.position_id_per_seconds

        audio_token_indices = torch.arange(audio_token_num).float() * self.audio_position_scale

        placeholder_parts = ["<|vision_start|>", self.audio_bos_token]
        video_data_index, audio_data_index = 0, 0
        while video_data_index < len(video_token_indices) and audio_data_index < len(audio_token_indices):
            if video_token_indices[video_data_index] <= audio_token_indices[audio_data_index]:
                placeholder_parts.append(self.video_pad)
                video_data_index += 1
            else:
                placeholder_parts.append(self.audio_pad)
                audio_data_index += 1

        if video_data_index < len(video_token_indices):
            placeholder_parts.append(self.video_pad * (len(video_token_indices) - video_data_index))
        if audio_data_index < len(audio_token_indices):
            placeholder_parts.append(self.audio_pad * (len(audio_token_indices) - audio_data_index))

        placeholder_parts.extend([self.audio_eos_token, "<|vision_end|>"])
        return "".join(placeholder_parts)

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        sys_msg = self._get_system_mesage()
        messages = [] if sys_msg is None else [sys_msg]
        data_type = ""
        multimodal_num_tokens = {key: iter(item) for key, item in num_tokens.items()}
        audio_token_iter = multimodal_num_tokens.get("audio")
        video_metadata_list = iter(kwargs.get("video_metadata", []))
        video_grid_thw = kwargs.get("grid_thw", {}).get("video", None)
        video_grid_thw = iter(video_grid_thw) if video_grid_thw is not None else iter([])
        enable_thinking = kwargs.get("enable_thinking", True)
        thinking_prompt_mode = kwargs.get("thinking_prompt", None)
        return_answer_text_mask = kwargs.get("return_answer_text_mask", False)

        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    content += self.image_pattern(next(multimodal_num_tokens["image"]))
                elif value[0] == "video":
                    try:
                        video_token_num = next(multimodal_num_tokens["video"])
                    except StopIteration as exc:
                        raise ValueError("Video token number is missing for a video input.") from exc
                    try:
                        video_metadata = next(video_metadata_list)
                    except StopIteration as exc:
                        raise ValueError("Video metadata is missing for a video input.") from exc

                    audio_token_num = next(audio_token_iter, 0) if audio_token_iter is not None else 0
                    if audio_token_num > 0:
                        try:
                            curr_video_grid_thw = next(video_grid_thw)
                        except StopIteration as exc:
                            raise ValueError("Video grid_thw is missing for an audio-video input.") from exc

                        content += self._build_token_interleaved_video_audio_pattern(
                            video_token_num,
                            audio_token_num,
                            curr_video_grid_thw=curr_video_grid_thw,
                            video_metadata=video_metadata,
                        )
                    else:
                        content += self._build_video_pattern(video_token_num, video_metadata)
                elif value[0] == "audio":
                    if audio_token_iter is None:
                        raise ValueError("Audio token number is missing for an audio input.")
                    content += self.audio_pattern(next(audio_token_iter))
                else:
                    raise ValueError(f"Unknown value type: {value[0]}")

            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        input_ids, attention_mask, labels = [], [], []
        answer_text_mask = []
        for message in messages:
            content_str = message["content"].strip()
            loss_mask = message["loss_mask"]
            role = message["role"]
            message_ids = self.tokenizer.encode("<|im_start|>" + role + "\n", add_special_tokens=False)
            message_answer_mask = [False] * len(message_ids)

            if content_str:
                end_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)

                if role == "user" and data_type == "t2i" and self._unconditioned_generation:
                    message_ids += [self.tokenizer.pad_token_id] * len(content_ids) + end_ids
                else:
                    message_ids += content_ids + end_ids
                if return_answer_text_mask and loss_mask == 1:
                    message_answer_mask += [True] * len(content_ids) + [False] * len(end_ids)
                else:
                    message_answer_mask += [False] * (len(content_ids) + len(end_ids))
            elif role == "assistant":
                if thinking_prompt_mode == "none":
                    thinking_prompt = ""
                elif thinking_prompt_mode == "enabled" or (thinking_prompt_mode is None and enable_thinking):
                    thinking_prompt = "<think>\n"
                elif thinking_prompt_mode == "disabled" or thinking_prompt_mode is None:
                    thinking_prompt = "<think>\n\n</think>\n\n"
                else:
                    raise ValueError(f"Unknown thinking_prompt: {thinking_prompt_mode}")
                if thinking_prompt:
                    thinking_prompt_ids = self.tokenizer.encode(thinking_prompt, add_special_tokens=False)
                    message_ids += thinking_prompt_ids
                    message_answer_mask += [False] * len(thinking_prompt_ids)

            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            answer_text_mask += message_answer_mask
            if loss_mask == 1:
                labels += message_ids
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if return_answer_text_mask:
            tokenized_example["answer_text_mask"] = answer_text_mask
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        input_mask = tokenized_example["labels"] == IGNORE_INDEX

        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]

        audio_mask = tokenized_example["input_ids"] == self.audio_token_id
        tokenized_example["input_ids"][audio_mask] = TYPE2INDEX["input"]["audio"]

        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX
        if return_answer_text_mask:
            tokenized_example["answer_text_mask"] &= tokenized_example["labels"] != IGNORE_INDEX
            tokenized_example["answer_text_mask"] &= tokenized_example["input_ids"] >= 0
        if data_type == "t2i":
            labels = tokenized_example["labels"]
            labels[labels == self.image_start_id] = IGNORE_INDEX
            tokenized_example["labels"] = labels
            if return_answer_text_mask:
                tokenized_example["answer_text_mask"] &= tokenized_example["labels"] != IGNORE_INDEX

        return tokenized_example


class Qwen25OmniChatTemplate(Qwen2VLChatTemplate):
    system_prompt = (
        "You are Qwen, a virtual human developed by the Qwen Team, "
        "Alibaba Group, capable of perceiving auditory and visual inputs, "
        "as well as generating text and speech."
    )

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        MultimodalChatTemplate.__init__(self, tokenizer)
        self.image_pad = "<|IMAGE|>"
        self.video_pad = "<|VIDEO|>"
        self.audio_pad = "<|AUDIO|>"
        self.vision_bos_token = "<|vision_bos|>"
        self.vision_eos_token = "<|vision_eos|>"
        self.audio_bos_token = "<|audio_bos|>"
        self.audio_eos_token = "<|audio_eos|>"

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)

        self.vision_bos_id = self.tokenizer.convert_tokens_to_ids(self.vision_bos_token)
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids(self.vision_eos_token)
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

        self.bos = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        self.seconds_per_chunk = 2.0
        self.position_id_per_seconds = 25
        self.video_second_per_grid = 1.0

        logger.info_rank0("Qwen25OmniTemplate will not truncate sequence when longer than [max_seq_lens].")

    def image_pattern(self, token_num):
        return self.vision_bos_token + self.image_pad * token_num + self.vision_eos_token

    def get_chunked_index(self, token_indices, tokens_per_chunk):
        """Copied from processing_qwen2_5_omni.py"""

        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(token_indices):
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def video_pattern(
        self, video_token_num: torch.Tensor, audio_token_num: torch.Tensor, curr_video_grid_thw: torch.Tensor
    ):
        if audio_token_num == 0:
            return self.vision_bos_token + self.video_pad * video_token_num + self.vision_eos_token
        else:
            """Modified from processing_qwen2_5_omni.py
            """
            audio_token_indices = torch.arange(audio_token_num)
            merge_size = torch.sqrt(curr_video_grid_thw.prod() // video_token_num).int()
            height = (curr_video_grid_thw[1] // merge_size).item()
            width = (curr_video_grid_thw[2] // merge_size).item()
            video_token_indices = torch.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
            video_token_indices = video_token_indices.expand(-1, height, width).reshape(-1)
            video_token_indices = video_token_indices * self.video_second_per_grid * self.position_id_per_seconds

            tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
            video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
            audio_chunk_indexes = self.get_chunked_index(audio_token_indices, tokens_per_chunk)

            content = self.vision_bos_token + self.audio_bos_token
            for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                if j < len(video_chunk_indexes):
                    video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                    content += self.video_pad * video_seq_length
                if j < len(audio_chunk_indexes):
                    audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                    content += self.audio_pad * audio_seq_length
            content += self.audio_eos_token + self.vision_eos_token
            return content

    def audio_pattern(self, token_num):
        return self.audio_bos_token + self.audio_pad * token_num + self.audio_eos_token

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        sys_msg = self._get_system_mesage()
        messages = [] if sys_msg is None else [sys_msg]
        multimodal_num_tokens = {key: iter(item) for key, item in num_tokens.items()}

        video_grid_thw = kwargs.get("grid_thw", {}).get("video", None)
        video_grid_thw = iter(video_grid_thw) if video_grid_thw is not None else None

        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    content += self.image_pattern(next(multimodal_num_tokens["image"]))
                elif value[0] == "video":
                    if video_grid_thw is None:
                        raise ValueError(
                            f"video_grid_thw: {video_grid_thw} is None. "
                            "Make sure your video processor outputs `grid_thw`."
                        )
                    content += self.video_pattern(
                        next(multimodal_num_tokens["video"]),
                        next(multimodal_num_tokens["audio"]),
                        curr_video_grid_thw=next(video_grid_thw),
                    )
                elif value[0] == "audio":
                    content += self.audio_pattern(next(multimodal_num_tokens["audio"]))
                else:
                    raise ValueError(f"Unknown value type: {value[0]}")
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["content"].strip()
            loss_mask = message["loss_mask"]
            role = message["role"]
            if content_str:
                content_str = "<|im_start|>" + message["role"] + "\n" + content_str + "<|im_end|>\n"
            else:
                content_str = "<|im_start|>" + message["role"] + "\n"

            message_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            if loss_mask == 1:
                labels += message_ids
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        input_mask = tokenized_example["labels"] == IGNORE_INDEX

        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]

        audio_mask = tokenized_example["input_ids"] == self.audio_token_id
        tokenized_example["input_ids"][audio_mask] = TYPE2INDEX["input"]["audio"]

        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX
        return tokenized_example


class JanusChatTemplate(ChatmlTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, use_system_prompt=True) -> None:
        super().__init__(tokenizer)
        self.image_pad = "<image_placeholder>"
        self.image_start_tag = "<begin_of_image>"
        self.image_end_tag = "<end_of_image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids(self.image_start_tag)
        self.use_system_prompt = use_system_prompt
        self.system_prompt = (
            "You are a helpful language and vision assistant. "
            "You are able to understand the visual content that the user provides, "
            "and assist the user with a variety of tasks using natural language."
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.image_pad]})
        self.sep1 = "\n\n"
        self.sep2 = "<｜end▁of▁sentence｜>"
        self.eos = self.tokenizer.encode(self.sep2, add_special_tokens=False)

    def image_pattern(self, token_num):
        return self.image_start_tag + self.image_pad * token_num + self.image_end_tag

    def encode_messages(
        self,
        conversations: Sequence[Dict[str, str]],
        num_tokens: Dict[str, List[int]] = defaultdict(list),
        max_seq_len: int = 8192,
        **kwargs,
    ) -> Dict[str, List[int]]:
        image_index = 0
        token_num_list = num_tokens.pop("image", [])
        messages = []
        use_system_prompt = False
        for i, message in enumerate(conversations):
            role = message[0]
            message = message[1:]
            content = ""
            for value in message:
                if value[0] == "text":
                    content += value[1]
                else:
                    use_system_prompt = True if role == "user" else use_system_prompt
                    assert value[0] == "image"
                    content += self.image_pattern(token_num_list[image_index])
                    image_index += 1
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        if use_system_prompt:
            input_ids = self.tokenizer.encode(self.system_prompt + self.sep1)
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(input_ids)
        else:
            input_ids = self.tokenizer.encode("")
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(input_ids)

        for i, message in enumerate(messages):
            role: str = message["role"]
            if role == "user":
                content_str = role.capitalize() + ": " + message["content"] + self.sep1
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                input_ids += content_ids
                attention_mask += [1] * len(content_ids)
                labels += [IGNORE_INDEX] * len(content_ids)
            else:
                content_str = role.capitalize() + ":"
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                input_ids += content_ids
                attention_mask += [1] * len(content_ids)
                labels += [IGNORE_INDEX] * len(content_ids)
                if message["content"]:
                    content_str = message["content"] + self.sep2
                    content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                    input_ids += content_ids
                    attention_mask += [1] * len(content_ids)
                    labels += content_ids

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX
        if not use_system_prompt:
            tokenized_example["labels"][tokenized_example["labels"] == self.eos[0]] = (
                IGNORE_INDEX
            )
            tokenized_example["labels"][tokenized_example["labels"] == self.image_start_id] = (
                IGNORE_INDEX
            )
        return tokenized_example


class LlamaPretrainTemplate(MultimodalChatTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer)
        self.vision_bos_token = "<|vision_start|>"
        self.vision_eos_token = "<|vision_end|>"
        self.audio_bos_token = "<|audio_start|>"
        self.audio_eos_token = "<|audio_end|>"
        num_add = self.tokenizer.add_tokens(
            [self.vision_bos_token, self.vision_eos_token, self.audio_bos_token, self.audio_eos_token]
        )
        num_add += self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.add_token_num = num_add

        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.audio_pad = "<|audio_pad|>"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_pad, self.video_pad, self.audio_pad]}
        )

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)

        self.vision_bos_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|pad|>")

        if self.add_token_num > 0:
            self.trained_embedding = [
                self.vision_bos_id,
                self.vision_eos_id,
                self.audio_bos_id,
                self.audio_eos_id,
                self.pad_token_id,
            ]

        self.seconds_per_chunk = 2.0
        self.position_id_per_seconds = 25
        self.video_second_per_grid = 1.0

        self.eos = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)
        self.bos = self.tokenizer.encode(self.tokenizer.bos_token, add_special_tokens=False)
        self.cfg_ratio = kwargs.get("cfg_ratio", None)

    def image_pattern(self, token_num):
        return self.vision_bos_token + self.image_pad * token_num + self.vision_eos_token

    def get_chunked_index(self, token_indices, tokens_per_chunk):
        """Copied from processing_qwen2_5_omni.py"""

        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(token_indices):
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def video_pattern(
        self, video_token_num: torch.Tensor, audio_token_num: torch.Tensor, curr_video_grid_thw: torch.Tensor
    ):
        if audio_token_num == 0:
            return self.vision_bos_token + self.video_pad * video_token_num + self.vision_eos_token
        else:
            """Modified from processing_qwen2_5_omni.py
            """
            audio_token_indices = torch.arange(audio_token_num)
            merge_size = torch.sqrt(curr_video_grid_thw.prod() // video_token_num).int()
            height = (curr_video_grid_thw[1] // merge_size).item()
            width = (curr_video_grid_thw[2] // merge_size).item()
            video_token_indices = torch.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
            video_token_indices = video_token_indices.expand(-1, height, width).reshape(-1)
            video_token_indices = video_token_indices * self.video_second_per_grid * self.position_id_per_seconds

            tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
            video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
            audio_chunk_indexes = self.get_chunked_index(audio_token_indices, tokens_per_chunk)

            content = self.vision_bos_token + self.audio_bos_token
            for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                if j < len(video_chunk_indexes):
                    video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                    content += self.video_pad * video_seq_length
                if j < len(audio_chunk_indexes):
                    audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                    content += self.audio_pad * audio_seq_length
            content += self.audio_eos_token + self.vision_eos_token
            return content

    def audio_pattern(self, token_num):
        return self.audio_bos_token + self.audio_pad * token_num + self.audio_eos_token

    @property
    def _unconditioned_generation(self):
        return self.cfg_ratio and random.random() < self.cfg_ratio

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        messages = []
        multimodal_num_tokens = {key: iter(item) for key, item in num_tokens.items()}
        data_type = ""
        video_grid_thw = kwargs.get("grid_thw", {}).get("video", None)
        video_grid_thw = iter(video_grid_thw) if video_grid_thw is not None else None

        for message in conversations:
            role = message[0]
            content = ""
            for item in message[1:]:
                mm_type = item[0]
                if mm_type == "text":
                    content += item[1]
                elif mm_type == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    content += self.image_pattern(next(multimodal_num_tokens[mm_type]))
                elif mm_type == "video":
                    if video_grid_thw is None:
                        raise ValueError(
                            f"video_grid_thw: {video_grid_thw} is None. "
                            "Make sure your video processor outputs `grid_thw`."
                        )
                    content += self.video_pattern(
                        next(multimodal_num_tokens["video"]),
                        next(multimodal_num_tokens["audio"]),
                        curr_video_grid_thw=next(video_grid_thw),
                    )
                elif mm_type == "audio":
                    content += self.audio_pattern(next(multimodal_num_tokens["audio"]))
                else:
                    raise ValueError(f"Unknown value type: {item[0]}")
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )
        input_ids, attention_mask, labels = [], [], []

        input_ids += self.bos
        attention_mask += [1] * len(self.bos)
        labels += [IGNORE_INDEX] * len(self.bos)
        for message in messages:
            content_str = message["content"].strip()
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            loss_mask = message["loss_mask"]
            if content_str == "":
                break
            if role == "user" and data_type == "t2i" and self._unconditioned_generation:
                input_ids += [self.pad_token_id] * len(content_ids)
            else:
                input_ids += content_ids

            attention_mask += [1] * len(content_ids)
            if loss_mask == 1:
                labels += content_ids
                input_ids += self.eos
                attention_mask += [1] * len(self.eos)
                labels += self.eos
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        image_mask = tokenized_example["input_ids"] == self.image_token_id

        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]

        audio_mask = tokenized_example["input_ids"] == self.audio_token_id
        tokenized_example["input_ids"][audio_mask] = TYPE2INDEX["input"]["audio"]

        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX

        if data_type == "t2i":
            labels = tokenized_example["labels"]
            labels[~output_image_mask] = IGNORE_INDEX
            tokenized_example["labels"] = labels
        return tokenized_example


class Qwen3MoeChatTemplate(Qwen25OmniChatTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        MultimodalChatTemplate.__init__(self, tokenizer)
        self.image_pad = "<|IMAGE|>"
        self.video_pad = "<|VIDEO|>"
        self.audio_pad = "<|AUDIO|>"

        self.vision_bos_token = "<|vision_bos|>"
        self.vision_eos_token = "<|vision_eos|>"
        self.audio_bos_token = "<|audio_bos|>"
        self.audio_eos_token = "<|audio_eos|>"

        num_add = self.tokenizer.add_tokens(
            [self.vision_bos_token, self.vision_eos_token, self.audio_bos_token, self.audio_eos_token]
        )
        self.add_token_num = num_add

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_pad, self.video_pad, self.audio_pad]}
        )

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)

        self.vision_bos_id = self.tokenizer.convert_tokens_to_ids(self.vision_bos_token)
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids(self.vision_eos_token)
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

        self.trained_embedding = []
        if self.add_token_num > 0:
            self.trained_embedding = [self.vision_bos_id, self.vision_eos_id, self.audio_bos_id, self.audio_eos_id]

        self.bos = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        self.seconds_per_chunk = 2.0
        self.position_id_per_seconds = 25
        self.video_second_per_grid = 1.0

        logger.info_rank0("Qwen3MoeTemplate will not truncate sequence when longer than [max_seq_lens].")

    def _get_system_mesage(self):
        return None


class SeedOssPretrainTemplate(LlamaPretrainTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer)
        self.vision_bos_token = "<|vision_start|>"
        self.vision_eos_token = "<|vision_end|>"
        self.audio_bos_token = "<|audio_start|>"
        self.audio_eos_token = "<|audio_end|>"
        num_add = self.tokenizer.add_tokens(
            [self.vision_bos_token, self.vision_eos_token, self.audio_bos_token, self.audio_eos_token]
        )
        num_add += self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.add_token_num = num_add

        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.audio_pad = "<|audio_pad|>"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_pad, self.video_pad, self.audio_pad]}
        )

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)

        self.vision_bos_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|pad|>")

        if self.add_token_num > 0:
            self.trained_embedding = [
                self.vision_bos_id,
                self.vision_eos_id,
                self.audio_bos_id,
                self.audio_eos_id,
                self.pad_token_id,
            ]

        self.seconds_per_chunk = 2.0
        self.position_id_per_seconds = 25
        self.video_second_per_grid = 1.0

        self.eos = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)
        self.bos = self.tokenizer.encode(self.tokenizer.bos_token, add_special_tokens=False)
        self.cfg_ratio = kwargs.get("cfg_ratio", None)


TEMPLATES = {
    "qwen2vl": Qwen2VLChatTemplate,
    "qwen3vl": Qwen3VLChatTemplate,
    "qwen3_5omni": Qwen35OmniChatTemplate,
    "qwen35omni": Qwen35OmniChatTemplate,
    "qwen2vl_pretrain": Qwen2VLPretrainTemplate,
    "qwen2_5omni": Qwen25OmniChatTemplate,
    "qwen2_5vl": Qwen2VLChatTemplate,
    "janus": JanusChatTemplate,
    "llama": LlamaPretrainTemplate,
    "qwen3moe": Qwen3MoeChatTemplate,
    "seed_oss": SeedOssPretrainTemplate,
}


def build_multimodal_chat_template(template_name: str, tokenizer: AutoTokenizer, **kwargs) -> "ChatTemplate":
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown chat template: {template_name}")

    return TEMPLATES[template_name](tokenizer, **kwargs)
