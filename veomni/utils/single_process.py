"""Single-device helpers for the inference-only release."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeState:
    global_rank: int = 0
    world_size: int = 1
    dp_size: int = 1
    tp_size: int = 1
    tp_rank: int = 0
    sp_size: int = 1
    sp_rank: int = 0
    dp_group: object = None
    sp_group: object = None
    ulysses_group: object = None
    sp_enabled: bool = False
    ulysses_enabled: bool = False
    tp_enabled: bool = False


RUNTIME_STATE = RuntimeState()


def get_runtime_state() -> RuntimeState:
    return RUNTIME_STATE


def identity_layout(tensor, *args, **kwargs):
    return tensor
