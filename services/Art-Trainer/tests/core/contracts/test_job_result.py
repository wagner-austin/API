"""Tests for job result contract."""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from art_trainer.core.contracts.job_result import (
    JobResult,
    decode_job_result,
    encode_job_result,
)


def test_encode_job_result_with_all_fields() -> None:
    """Test encode_job_result with all fields populated."""
    result: JobResult = {
        "job_id": "job-123",
        "lora_file_id": "file-456",
        "lora_name": "my_lora",
    }

    encoded = encode_job_result(result)

    assert encoded["job_id"] == "job-123"
    assert encoded["lora_file_id"] == "file-456"
    assert encoded["lora_name"] == "my_lora"


def test_encode_job_result_with_none_fields() -> None:
    """Test encode_job_result with None optional fields."""
    result: JobResult = {
        "job_id": "job-789",
        "lora_file_id": None,
        "lora_name": None,
    }

    encoded = encode_job_result(result)

    assert encoded["job_id"] == "job-789"
    assert encoded["lora_file_id"] is None
    assert encoded["lora_name"] is None


def test_decode_job_result_with_all_fields() -> None:
    """Test decode_job_result with all fields populated."""
    obj: JSONObject = {
        "job_id": "job-123",
        "lora_file_id": "file-456",
        "lora_name": "my_lora",
    }

    decoded = decode_job_result(obj)

    assert decoded["job_id"] == "job-123"
    assert decoded["lora_file_id"] == "file-456"
    assert decoded["lora_name"] == "my_lora"


def test_decode_job_result_with_none_fields() -> None:
    """Test decode_job_result with None optional fields."""
    obj: JSONObject = {
        "job_id": "job-789",
        "lora_file_id": None,
        "lora_name": None,
    }

    decoded = decode_job_result(obj)

    assert decoded["job_id"] == "job-789"
    assert decoded["lora_file_id"] is None
    assert decoded["lora_name"] is None


def test_roundtrip_encode_decode() -> None:
    """Test roundtrip encoding and decoding preserves data."""
    original: JobResult = {
        "job_id": "roundtrip-job",
        "lora_file_id": "roundtrip-file",
        "lora_name": "roundtrip_lora",
    }

    encoded = encode_job_result(original)
    decoded = decode_job_result(encoded)

    assert decoded["job_id"] == original["job_id"]
    assert decoded["lora_file_id"] == original["lora_file_id"]
    assert decoded["lora_name"] == original["lora_name"]
