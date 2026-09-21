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
import inspect
from functools import partial
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .....utils.constants import AUDIO_INPUT_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
)
from ..base import BaseFoundationModelMixin
from .configuration_qwen3_5_foundation import Qwen35FoundationConfig


def parse_position_id_kwargs(input_ids: torch.Tensor, attention_mask: torch.Tensor, grid_thw: Dict = {}, **kwargs):
    return_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "image" in grid_thw:
        return_dict["image_grid_thw"] = grid_thw["image"]
    if "video" in grid_thw:
        return_dict["video_grid_thw"] = grid_thw["video"]
    if "video_second_per_grid" in kwargs:
        return_dict["second_per_grids"] = kwargs["video_second_per_grid"].to(input_ids.device)
    elif "video" in grid_thw:
        return_dict["second_per_grids"] = torch.tensor([1.0] * len(grid_thw["video"])).to(input_ids.device)
    if "num_tokens" in kwargs and "audio" in kwargs["num_tokens"]:
        return_dict["audio_token_counts"] = kwargs["num_tokens"]["audio"]
    if "audio_start_token_id" in kwargs:
        return_dict["audio_start_token_id"] = kwargs["audio_start_token_id"]
    if "audio_end_token_id" in kwargs:
        return_dict["audio_end_token_id"] = kwargs["audio_end_token_id"]
    if "position_id_per_seconds" in kwargs:
        return_dict["position_id_per_seconds"] = kwargs["position_id_per_seconds"]
    if "audio_position_scale" in kwargs:
        return_dict["audio_position_scale"] = kwargs["audio_position_scale"]
    return return_dict


def _normalize_optional_tensor(data, device: torch.device) -> Optional[torch.Tensor]:
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(device=device)
    return torch.as_tensor(data, device=device)


def _get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: torch.Tensor,
    grid_hs: torch.Tensor,
    grid_ws: torch.Tensor,
) -> torch.Tensor:
    device = t_index.device
    llm_grid_h = int((grid_hs[vision_idx] // spatial_merge_size).item())
    llm_grid_w = int((grid_ws[vision_idx] // spatial_merge_size).item())
    h_index = torch.arange(llm_grid_h, device=device).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w, device=device).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
    t_index = t_index.view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
    return torch.stack([t_index.float(), h_index.float(), w_index.float()]) + start_idx


def _build_qwen3_5_omni_interleaved_multimodal_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    audio_token_counts: Optional[torch.Tensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
    image_token_id: int = IMAGE_INPUT_INDEX,
    video_token_id: int = VIDEO_INPUT_INDEX,
    audio_token_id: int = AUDIO_INPUT_INDEX,
    vision_start_token_id: Optional[int] = None,
    audio_start_token_id: Optional[int] = None,
    position_id_per_seconds: int = 25,
    audio_position_scale: float = 0.5,
    spatial_merge_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_mask = attention_mask == 1
    position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=torch.float, device=input_ids.device)

    audio_token_counts = _normalize_optional_tensor(audio_token_counts, input_ids.device)
    if audio_token_counts is None:
        audio_token_counts = torch.zeros(0, dtype=torch.long, device=input_ids.device)
    else:
        audio_token_counts = audio_token_counts.view(-1).to(dtype=torch.long)

    second_per_grids = _normalize_optional_tensor(second_per_grids, input_ids.device)
    if second_per_grids is None:
        second_per_grids = torch.ones(
            0 if video_grid_thw is None else video_grid_thw.shape[0],
            dtype=torch.float,
            device=input_ids.device,
        )
    else:
        second_per_grids = second_per_grids.view(-1).float()

    mrope_position_deltas = []
    image_idx, video_idx, audio_idx = 0, 0, 0
    image_grid_hs = image_grid_thw[:, 1] if image_grid_thw is not None else None
    image_grid_ws = image_grid_thw[:, 2] if image_grid_thw is not None else None
    video_grid_hs = video_grid_thw[:, 1] if video_grid_thw is not None else None
    video_grid_ws = video_grid_thw[:, 2] if video_grid_thw is not None else None

    for batch_idx, sample_input_ids in enumerate(input_ids):
        valid_input_ids = sample_input_ids[attention_mask[batch_idx]]
        if valid_input_ids.numel() == 0:
            mrope_position_deltas.append(torch.tensor(0.0, device=input_ids.device))
            continue

        vision_start_indices = torch.argwhere(valid_input_ids == vision_start_token_id).squeeze(1)
        if vision_start_indices.numel() > 0:
            vision_tokens = valid_input_ids[vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum().item())
            video_nums = int(
                ((vision_tokens == video_token_id) | (vision_tokens == audio_start_token_id)).sum().item()
            )
        else:
            image_nums, video_nums = 0, 0

        audio_start_indices = torch.argwhere(valid_input_ids == audio_start_token_id).squeeze(1)
        if audio_start_indices.numel() > 0:
            audio_nums = int((valid_input_ids[audio_start_indices - 1] != vision_start_token_id).sum().item())
        else:
            audio_nums = 0

        input_tokens = valid_input_ids.tolist()
        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
        multimodal_nums = image_nums + video_nums + audio_nums

        for _ in range(multimodal_nums):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            if (remain_videos > 0 or remain_images > 0) and vision_start_token_id in input_tokens:
                ed_vision_start = input_tokens.index(vision_start_token_id, st)
            else:
                ed_vision_start = len(input_tokens) + 1
            if remain_audios > 0 and audio_start_token_id in input_tokens:
                ed_audio_start = input_tokens.index(audio_start_token_id, st)
            else:
                ed_audio_start = len(input_tokens) + 1
            min_ed = min(ed_vision_start, ed_audio_start)

            text_len = min_ed - st
            if text_len != 0:
                llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)
                st_idx += text_len

            if min_ed == ed_vision_start and int(valid_input_ids[ed_vision_start + 1].item()) == audio_start_token_id:
                bos_len, eos_len = 2, 2
            else:
                bos_len, eos_len = 1, 1
            llm_pos_ids_list.append(torch.arange(bos_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)
            st_idx += bos_len

            if min_ed == ed_audio_start:
                audio_len = int(audio_token_counts[audio_idx].item())
                if audio_len > 0:
                    audio_time_index = torch.arange(audio_len, device=input_ids.device).float() * audio_position_scale
                    llm_pos_ids_list.append(
                        audio_time_index.view(1, -1).expand(3, -1) + st_idx
                    )
                st += int(text_len + bos_len + audio_len + eos_len)
                audio_idx += 1
                remain_audios -= 1

            elif min_ed == ed_vision_start and int(valid_input_ids[ed_vision_start + 1].item()) == image_token_id:
                grid_t = int(image_grid_thw[image_idx][0].item())
                t_index = (torch.arange(grid_t, device=input_ids.device) * 1 * position_id_per_seconds).float()
                image_llm_pos_ids = _get_llm_pos_ids_for_vision(
                    int(st_idx),
                    image_idx,
                    spatial_merge_size,
                    t_index,
                    image_grid_hs,
                    image_grid_ws,
                )
                image_len = int((image_grid_thw[image_idx].prod() // (spatial_merge_size**2)).item())
                llm_pos_ids_list.append(image_llm_pos_ids)
                st += int(text_len + bos_len + image_len + eos_len)
                image_idx += 1
                remain_images -= 1

            elif min_ed == ed_vision_start:
                next_token_id = int(valid_input_ids[ed_vision_start + 1].item())

                if next_token_id == video_token_id:
                    if audio_idx < audio_token_counts.numel() and int(audio_token_counts[audio_idx].item()) == 0:
                        audio_idx += 1
                    grid_t = int(video_grid_thw[video_idx][0].item())
                    t_index = (
                        torch.arange(grid_t, device=input_ids.device) * second_per_grids[video_idx] * position_id_per_seconds
                    ).float()
                    video_llm_pos_ids = _get_llm_pos_ids_for_vision(
                        int(st_idx),
                        video_idx,
                        spatial_merge_size,
                        t_index,
                        video_grid_hs,
                        video_grid_ws,
                    )
                    video_len = int((video_grid_thw[video_idx].prod() // (spatial_merge_size**2)).item())
                    llm_pos_ids_list.append(video_llm_pos_ids)
                    st += int(text_len + bos_len + video_len + eos_len)
                    video_idx += 1
                    remain_videos -= 1
                else:
                    audio_len = int(audio_token_counts[audio_idx].item())
                    audio_time_index = torch.arange(audio_len, device=input_ids.device).float() * audio_position_scale
                    audio_llm_pos_ids = (
                        audio_time_index.view(1, -1).expand(3, -1) + st_idx
                    )
                    grid_t = int(video_grid_thw[video_idx][0].item())
                    t_index = (
                        torch.arange(grid_t, device=input_ids.device) * second_per_grids[video_idx] * position_id_per_seconds
                    ).float()
                    video_llm_pos_ids = _get_llm_pos_ids_for_vision(
                        int(st_idx),
                        video_idx,
                        spatial_merge_size,
                        t_index,
                        video_grid_hs,
                        video_grid_ws,
                    )
                    video_data_index, audio_data_index = 0, 0
                    while video_data_index < video_llm_pos_ids.shape[-1] and audio_data_index < audio_llm_pos_ids.shape[-1]:
                        if video_llm_pos_ids[0, video_data_index] <= audio_llm_pos_ids[0, audio_data_index]:
                            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                            video_data_index += 1
                        else:
                            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                            audio_data_index += 1
                    if video_data_index < video_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index:])
                    if audio_data_index < audio_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index:])

                    video_len = int((video_grid_thw[video_idx].prod() // (spatial_merge_size**2)).item())
                    st += int(text_len + bos_len + audio_len + video_len + eos_len)
                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(eos_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)
        position_ids[:, batch_idx, attention_mask[batch_idx]] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - attention_mask[batch_idx].sum())

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def build_qwen3_5_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    audio_token_counts: Optional[torch.Tensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
    image_token_id: int = IMAGE_INPUT_INDEX,
    video_token_id: int = VIDEO_INPUT_INDEX,
    audio_token_id: int = AUDIO_INPUT_INDEX,
    vision_start_token_id: Optional[int] = None,
    audio_start_token_id: Optional[int] = None,
    audio_end_token_id: Optional[int] = None,
    position_id_per_seconds: int = 25,
    audio_position_scale: float = 0.5,
    spatial_merge_size: int = 2,
    **kwargs,
):
    text_position_ids = attention_mask.long().cumsum(-1) - 1
    text_position_ids = text_position_ids.masked_fill(attention_mask == 0, 0)

    use_qwen_omni_interleave = (
        audio_start_token_id is not None
        and audio_token_counts is not None
        and torch.any(input_ids == audio_start_token_id)
    )

    if use_qwen_omni_interleave:
        multimodal_position_ids, rope_deltas = _build_qwen3_5_omni_interleaved_multimodal_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            audio_token_counts=audio_token_counts,
            second_per_grids=second_per_grids,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
            vision_start_token_id=vision_start_token_id,
            audio_start_token_id=audio_start_token_id,
            position_id_per_seconds=position_id_per_seconds,
            audio_position_scale=audio_position_scale,
            spatial_merge_size=spatial_merge_size,
        )
    elif image_grid_thw is not None or video_grid_thw is not None:
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                vision_start_token_id=vision_start_token_id,
                vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size),
            )
        )
        multimodal_position_ids, rope_deltas = Qwen3_5Model.get_rope_index(
            fake_model,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
    else:
        multimodal_position_ids = text_position_ids.unsqueeze(0).expand(3, text_position_ids.shape[0], -1).clone()
        rope_deltas = torch.zeros(
            text_position_ids.shape[0], 1, dtype=text_position_ids.dtype, device=text_position_ids.device
        )

    position_ids = torch.cat([text_position_ids.unsqueeze(0), multimodal_position_ids], dim=0)

    if position_ids.shape[1] == 1:
        position_ids = position_ids[:, 0].contiguous()
    else:
        position_ids = position_ids.transpose(0, 1).contiguous()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
        **kwargs,
    }


class Qwen35FoundationModel(BaseFoundationModelMixin, Qwen3_5ForConditionalGeneration):
    config_class = Qwen35FoundationConfig
    _no_split_modules = ["Qwen3_5TextDecoderLayer", "Qwen3_5VisionBlock"]
    forward_extra_keys = (
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
        "deterministic",
        "softcap",
        "sliding_window",
        "s_aux",
    )

    def __init__(self, config: Qwen35FoundationConfig, **kwargs):
        BaseFoundationModelMixin.__init__(self, config, **kwargs)
        Qwen3_5PreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

        self.config.image_token_id = IMAGE_INPUT_INDEX
        self.config.video_token_id = VIDEO_INPUT_INDEX
        self.model.config.image_token_id = IMAGE_INPUT_INDEX
        self.model.config.video_token_id = VIDEO_INPUT_INDEX
        self.image_token_id = IMAGE_INPUT_INDEX
        self.video_token_id = VIDEO_INPUT_INDEX
        self.rope_deltas = None

        self.post_init()

    def get_position_id_func(self):
        return [
            parse_position_id_kwargs,
            partial(
                build_qwen3_5_position_ids,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.config.vision_start_token_id,
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
            ),
        ]

    def _filter_forward_kwargs(self, kwargs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hf_forward_keys = {
            key
            for key in inspect.signature(Qwen3_5ForConditionalGeneration.forward).parameters.keys()
            if key not in {"self", "kwargs"}
        }
        allowed_extra_keys = set(self.forward_extra_keys)
        return {
            key: value for key, value in kwargs.items() if key in hf_forward_keys or key in allowed_extra_keys
        }

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        is_first_iteration=False,
        **kwargs,
    ):
        if rope_deltas is not None:
            self.rope_deltas = rope_deltas
            self.model.rope_deltas = rope_deltas

        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5CausalLMOutputWithPast]:
        batch_size = None
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        elif attention_mask is not None:
            batch_size = attention_mask.shape[0]

        if (
            position_ids is not None
            and position_ids.ndim == 3
            and batch_size is not None
            and position_ids.shape[0] == batch_size
            and position_ids.shape[1] in (1, 3, 4)
        ):
            position_ids = position_ids.transpose(0, 1).contiguous()

        foundation_kwargs = self._filter_forward_kwargs(kwargs)
        raw_multimodal_inputs_present = any(
            foundation_kwargs.get(key) is not None
            for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
        )
        prefer_inputs_embeds = inputs_embeds is not None and (
            position_ids is not None or not raw_multimodal_inputs_present
        )

        outputs = super().forward(
            input_ids=None if prefer_inputs_embeds else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **foundation_kwargs,
        )
        if hasattr(outputs, "rope_deltas"):
            self.rope_deltas = outputs.rope_deltas
            self.model.rope_deltas = outputs.rope_deltas
        return outputs
