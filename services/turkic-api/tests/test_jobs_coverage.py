"""Coverage tests for jobs.py missing branches."""

from __future__ import annotations

import pytest

from turkic_api.api.jobs import _decode_job_params


def test_decode_job_params_user_id_not_int() -> None:
    """Cover line 66: user_id must be an int TypeError."""
    with pytest.raises(TypeError, match="user_id must be an int"):
        _decode_job_params(
            {
                "user_id": "42",  # string, not int
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            }
        )


def test_decode_job_params_user_id_none() -> None:
    """Cover line 66: user_id None is not an int."""
    with pytest.raises(TypeError, match="user_id must be an int"):
        _decode_job_params(
            {
                "user_id": None,
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            }
        )


def test_decode_job_params_user_id_float() -> None:
    """Cover line 66: user_id as float is not an int."""
    with pytest.raises(TypeError, match="user_id must be an int"):
        _decode_job_params(
            {
                "user_id": 42.5,
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            }
        )
