"""LoRA upload service.

This module provides functions to upload trained LoRA files to data-bank.
"""

from __future__ import annotations

from pathlib import Path

from art_trainer.core.config.settings import Settings

from . import _test_hooks
from ._test_hooks import UploadResult


def upload_lora(
    settings: Settings,
    lora_path: Path,
) -> UploadResult:
    """Upload a trained LoRA file to data-bank.

    Args:
        settings: Application settings.
        lora_path: Path to the .safetensors file.

    Returns:
        UploadResult with file_id and filename.

    Raises:
        DataBankUploadError: If upload fails.
        FileNotFoundError: If lora_path does not exist.
    """
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    # Build upload URL
    base_url = settings["app"]["data_bank_api_url"].rstrip("/")
    upload_url = f"{base_url}/files"

    # Build headers
    headers: dict[str, str] = {}
    api_key = settings["app"]["data_bank_api_key"]
    if api_key:
        headers["X-API-Key"] = api_key

    # Read file content
    file_content = lora_path.read_bytes()
    filename = lora_path.name

    # Upload the file
    return _test_hooks.http_upload(
        upload_url,
        headers,
        filename,
        file_content,
    )


__all__ = [
    "UploadResult",
    "upload_lora",
]
