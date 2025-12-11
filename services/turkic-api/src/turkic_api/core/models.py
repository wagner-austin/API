from __future__ import annotations

from typing import Literal, TypeGuard

from typing_extensions import TypedDict

Source = Literal["oscar", "wikipedia", "culturax"]
Language = Literal["kk", "ky", "uz", "tr", "ug", "fi", "az", "en"]


class ProcessSpec(TypedDict):
    source: Source
    language: Language
    max_sentences: int
    transliterate: bool
    confidence_threshold: float


def is_source(value: str) -> TypeGuard[Source]:
    return value in ("oscar", "wikipedia", "culturax")


def is_language(value: str) -> TypeGuard[Language]:
    return value in ("kk", "ky", "uz", "tr", "ug", "fi", "az", "en")
