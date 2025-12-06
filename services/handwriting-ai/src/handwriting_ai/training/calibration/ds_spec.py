from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

BaseKind = Literal["mnist", "inline"]


class MNISTSpec(TypedDict):
    root: Path
    train: bool


class InlineSpec(TypedDict):
    # Lightweight, test-only dataset description
    n: int
    sleep_s: float
    fail: bool


class AugmentSpec(TypedDict):
    augment: bool
    aug_rotate: float
    aug_translate: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph: str


class PreprocessSpec(TypedDict):
    base_kind: BaseKind
    mnist: MNISTSpec | None
    inline: InlineSpec | None
    augment: AugmentSpec


__all__ = [
    "AugmentSpec",
    "BaseKind",
    "InlineSpec",
    "MNISTSpec",
    "PreprocessSpec",
]
