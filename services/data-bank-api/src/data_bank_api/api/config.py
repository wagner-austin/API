from __future__ import annotations

from typing import TypedDict


class Settings(TypedDict):
    """Settings for the job processing client.

    This mirrors only the fields required by the tests and by the job implementation.
    """

    data_dir: str
    environment: str
    data_bank_api_url: str
    data_bank_api_key: str


class JobParams(TypedDict):
    """Parameters for corpus processing jobs."""

    source: str
    language: str
    max_sentences: int
    transliterate: bool
    confidence_threshold: float
