from __future__ import annotations

from enum import StrEnum

from typing_extensions import TypedDict


class Source(StrEnum):
    """A corpus a job reads from, as requests and queue payloads spell it."""

    OSCAR = "oscar"
    WIKIPEDIA = "wikipedia"
    CULTURAX = "culturax"


class Language(StrEnum):
    """An ISO 639-1 language code a job extracts sentences in."""

    KK = "kk"
    KY = "ky"
    UZ = "uz"
    TR = "tr"
    UG = "ug"
    FI = "fi"
    AZ = "az"
    EN = "en"
    RU = "ru"


class Script(StrEnum):
    """An ISO 15924 script a job may restrict its sentences to."""

    LATN = "Latn"
    CYRL = "Cyrl"
    ARAB = "Arab"


class ProcessSpec(TypedDict):
    source: Source
    language: Language
    max_sentences: int
    transliterate: bool
    confidence_threshold: float
