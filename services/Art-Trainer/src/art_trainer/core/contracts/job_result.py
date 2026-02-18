"""Job result contract with encode/decode functions.

This module defines the JobResult TypedDict for storing training
job outcomes and provides encode/decode functions for serialization.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    optional_str,
    require_str,
)
from typing_extensions import TypedDict


class JobResult(TypedDict, total=True):
    """Training job result stored in Redis.

    Attributes:
        job_id: Unique job identifier.
        lora_file_id: File ID of the uploaded LoRA in data-bank.
            None if training failed or LoRA was not uploaded.
        lora_name: Name of the deployed LoRA in ComfyUI.
            None if training failed or LoRA was not deployed.
    """

    job_id: str
    lora_file_id: str | None
    lora_name: str | None


def encode_job_result(result: JobResult) -> JSONObject:
    """Encode JobResult to JSONObject for serialization.

    Args:
        result: Job result to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "job_id": result["job_id"],
        "lora_file_id": result["lora_file_id"],
        "lora_name": result["lora_name"],
    }


def decode_job_result(obj: JSONObject) -> JobResult:
    """Decode JSONObject to JobResult with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated JobResult TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    job_id = require_str(obj, "job_id")
    lora_file_id = optional_str(obj, "lora_file_id")
    lora_name = optional_str(obj, "lora_name")

    return {
        "job_id": job_id,
        "lora_file_id": lora_file_id,
        "lora_name": lora_name,
    }


__all__ = [
    "JobResult",
    "decode_job_result",
    "encode_job_result",
]
