import inspect
import os
from dataclasses import replace
from typing import List, Optional, Union

import torch
from transformers import BatchFeature
from transformers.image_utils import ImageInput, PILImageResampling
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.video_utils import VideoInput, VideoMetadata, is_valid_image, is_valid_video

from ..base import BaseEncoderProcessorMixin


IMAGE_PROCESS_KWARGS = set(Qwen2VLImageProcessor.valid_kwargs.__annotations__.keys()) | {"return_tensors"}
VIDEO_PROCESS_KWARGS = set(Qwen3VLVideoProcessor.valid_kwargs.__annotations__.keys()) | {"return_tensors"}


class Qwen35VisionModelProcessor(BaseEncoderProcessorMixin):
    attributes = ["image_processor", "video_processor"]
    optional_attributes = BaseEncoderProcessorMixin.optional_attributes
    # Transformers >= 5 expects a single class name here. The exported
    # processor config uses the Fast implementation.
    image_processor_class = "Qwen2VLImageProcessorFast"
    video_processor_class = "Qwen3VLVideoProcessor"
    valid_kwargs = BaseEncoderProcessorMixin.valid_kwargs + sorted(
        {
            key
            for key in inspect.signature(Qwen2VLImageProcessor.__init__).parameters.keys()
            if key not in {"self", "kwargs"}
        }
        | set(Qwen3VLVideoProcessor.valid_kwargs.__annotations__.keys())
        | {"max_input_frames"}
    )

    def __init__(
        self,
        image_processor: Optional[Qwen2VLImageProcessor] = None,
        video_processor: Optional[Qwen3VLVideoProcessor] = None,
        token_num: int = None,
        token_size: List = None,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 16,
        temporal_patch_size: int = 2, ###
        merge_size: int = 2,
        min_frames: int = 4,
        max_frames: int = 768,
        max_input_frames: Optional[int] = None,
        # fps 是视频采样策略，它只在“processor 自己负责从原始视频再采样”时才有意义。
        # 现在已经把默认改成 do_sample_frames=False，所以如果传入的是“已经抽好的帧序列”，processor 只负责把这串帧 patchify，不会再假定 2fps 去重采样。
        do_sample_frames: bool = False, 
        fps: Union[int, float, None] = None,
        size: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if (image_mean is None) ^ (image_std is None):
            raise ValueError("`image_mean` and `image_std` must either both be provided or both be None.")
        if max_input_frames is not None and max_input_frames <= 0:
            raise ValueError("`max_input_frames` must be positive when provided.")

        if image_processor is None:
            image_processor = Qwen2VLImageProcessor(
                do_resize=do_resize,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_convert_rgb=do_convert_rgb,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
            )

        if video_processor is None:
            video_kwargs = {
                "do_resize": do_resize,
                "resample": resample,
                "do_rescale": do_rescale,
                "rescale_factor": rescale_factor,
                "do_normalize": do_normalize,
                "do_convert_rgb": do_convert_rgb,
                "patch_size": patch_size,
                "temporal_patch_size": temporal_patch_size,
                "merge_size": merge_size,
                "min_frames": min_frames,
                "max_frames": max_frames,
                "do_sample_frames": do_sample_frames,
            }
            if image_mean is not None:
                video_kwargs["image_mean"] = image_mean
            if image_std is not None:
                video_kwargs["image_std"] = image_std
            if fps is not None:
                video_kwargs["fps"] = fps
            if size is not None:
                video_kwargs["size"] = size
            video_processor = Qwen3VLVideoProcessor(**video_kwargs)

        super().__init__(
            token_num=token_num,
            token_size=token_size,
            image_processor=image_processor,
            video_processor=video_processor,
            **kwargs,
        )
        self.max_input_frames = max_input_frames

    def process(
        self,
        images: Optional[ImageInput] = None,
        videos: Optional[VideoInput] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        assert images is None or videos is None, "Only one of images and videos can be provided."
        assert images is not None or videos is not None, "One of images and videos must be provided."

        if images is not None:
            image_kwargs = self._select_kwargs(kwargs, IMAGE_PROCESS_KWARGS)
            output = self.image_processor.preprocess(images=images, return_tensors=return_tensors, **image_kwargs)
            pixel_values = output["pixel_values"]
            image_grid_thw = output["image_grid_thw"].to(torch.int32)
            num_image_tokens = image_grid_thw.prod(dim=-1).to(torch.int32) // (self.image_processor.merge_size**2)
            return BatchFeature(
                data={"features": pixel_values, "num_tokens": num_image_tokens, "grid_thw": image_grid_thw},
                tensor_type=return_tensors,
            )

        max_input_frames = kwargs.pop("max_input_frames", self.max_input_frames)
        video_kwargs = self._select_kwargs(kwargs, VIDEO_PROCESS_KWARGS)
        if max_input_frames is not None and not video_kwargs.get("do_sample_frames", self.video_processor.do_sample_frames):
            videos = self._truncate_video_inputs(videos, max_input_frames)
            if "video_metadata" in video_kwargs:
                video_kwargs["video_metadata"] = self._truncate_video_metadata(
                    video_kwargs["video_metadata"], max_input_frames
                )

        output = self.video_processor(videos=videos, return_tensors=return_tensors, **video_kwargs)
        pixel_values = output["pixel_values_videos"]
        video_grid_thw = output["video_grid_thw"].to(torch.int32)
        num_video_tokens = video_grid_thw.prod(dim=-1).to(torch.int32) // (self.video_processor.merge_size**2)
        data = {"features": pixel_values, "num_tokens": num_video_tokens, "grid_thw": video_grid_thw}
        if "video_metadata" in output:
            data["video_metadata"] = output["video_metadata"]
        return BatchFeature(data=data, tensor_type=return_tensors)

    @staticmethod
    def _select_kwargs(kwargs, valid_keys):
        return {key: value for key, value in kwargs.items() if key in valid_keys}

    @classmethod
    def _truncate_video_inputs(cls, videos, max_input_frames: int):
        if videos is None:
            return None

        if isinstance(videos, torch.Tensor):
            if videos.ndim == 5:
                return videos[:, :max_input_frames, ...]
            if videos.ndim == 4:
                return videos[:max_input_frames, ...]
            return videos

        if hasattr(videos, "ndim") and hasattr(videos, "shape"):
            if videos.ndim == 5:
                return videos[:, :max_input_frames, ...]
            if videos.ndim == 4:
                return videos[:max_input_frames, ...]
            return videos

        if isinstance(videos, tuple):
            return tuple(cls._truncate_video_inputs(list(videos), max_input_frames))

        if not isinstance(videos, list):
            return videos

        if not videos:
            return videos

        first_item = videos[0]
        if cls._is_frame_item(first_item):
            return videos[:max_input_frames]

        return [cls._truncate_video_inputs(video, max_input_frames) for video in videos]

    @classmethod
    def _truncate_video_metadata(cls, metadata, max_input_frames: int):
        if metadata is None:
            return None

        if isinstance(metadata, VideoMetadata):
            frames_indices = metadata.frames_indices[:max_input_frames] if metadata.frames_indices is not None else None
            total_num_frames = min(metadata.total_num_frames, max_input_frames)
            return replace(metadata, total_num_frames=total_num_frames, frames_indices=frames_indices)

        if isinstance(metadata, dict):
            new_metadata = dict(metadata)
            if new_metadata.get("frames_indices") is not None:
                new_metadata["frames_indices"] = new_metadata["frames_indices"][:max_input_frames]
            if new_metadata.get("total_num_frames") is not None:
                new_metadata["total_num_frames"] = min(new_metadata["total_num_frames"], max_input_frames)
            return new_metadata

        if isinstance(metadata, tuple):
            return tuple(cls._truncate_video_metadata(list(metadata), max_input_frames))

        if isinstance(metadata, list):
            return [cls._truncate_video_metadata(item, max_input_frames) for item in metadata]

        return metadata

    @staticmethod
    def _is_frame_item(item) -> bool:
        if is_valid_image(item):
            return True
        if isinstance(item, (str, os.PathLike)):
            return True
        if is_valid_video(item):
            return False
        return False
