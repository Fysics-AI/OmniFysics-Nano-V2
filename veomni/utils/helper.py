"""Small inference-only helpers."""

import gc
import logging as builtin_logging
import os
import sys
from functools import lru_cache
from typing import Optional

import torch
import transformers

from .device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from . import logging


def create_logger(name: Optional[str] = None):
    logger = builtin_logging.getLogger(name)
    if not logger.handlers:
        handler = builtin_logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            builtin_logging.Formatter(
                fmt="[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(builtin_logging.INFO)
    logger.propagate = False
    return logger


def empty_cache() -> None:
    gc.collect()
    if IS_CUDA_AVAILABLE:
        torch.cuda.empty_cache()
    elif IS_NPU_AVAILABLE:
        torch.npu.empty_cache()


def get_cache_dir(path: Optional[str] = None) -> str:
    cache_dir = os.environ.get("VEOMNI_CACHE_DIR", os.path.join("/tmp", "veomni"))
    if path is None:
        return cache_dir
    return os.path.join(cache_dir, os.path.basename(os.path.normpath(path)), "")


@lru_cache
def get_dtype_size(dtype: torch.dtype) -> int:
    sizes = {
        torch.int64: 8,
        torch.float64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
    }
    return sizes[dtype]


def enable_third_party_logging() -> None:
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()
