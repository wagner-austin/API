"""Dataset download and extraction service.

This module provides functions to download datasets from data-bank
and extract them to the local filesystem.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from art_trainer.core.config.settings import Settings
from art_trainer.core.infra.paths import dataset_dir

from . import _test_hooks


def download_dataset(
    settings: Settings,
    dataset_file_id: str,
    dataset_id: str,
) -> Path:
    """Download dataset from data-bank and extract to local directory.

    Downloads the dataset ZIP file from data-bank using the file_id,
    extracts it to the dataset directory, and returns the path.

    Args:
        settings: Application settings.
        dataset_file_id: File ID in data-bank.
        dataset_id: Unique identifier for the local dataset directory.

    Returns:
        Path to the extracted dataset directory.

    Raises:
        DataBankDownloadError: If download fails.
        zipfile.BadZipFile: If the downloaded file is not a valid ZIP.
    """
    # Build download URL
    base_url = settings["app"]["data_bank_api_url"].rstrip("/")
    download_url = f"{base_url}/files/{dataset_file_id}"

    # Build headers
    headers: dict[str, str] = {}
    api_key = settings["app"]["data_bank_api_key"]
    if api_key:
        headers["X-API-Key"] = api_key

    # Download the file
    zip_bytes = _test_hooks.http_get(download_url, headers)

    # Prepare output directory
    output_dir = dataset_dir(settings, dataset_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract ZIP to output directory
    import io

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        zf.extractall(output_dir)

    return output_dir


def dataset_exists(settings: Settings, dataset_id: str) -> bool:
    """Check if a dataset already exists locally.

    Args:
        settings: Application settings.
        dataset_id: Dataset identifier.

    Returns:
        True if dataset directory exists and contains files.
    """
    path = dataset_dir(settings, dataset_id)
    if not path.exists():
        return False
    # Check if directory has any files (not just subdirectories)
    return any(path.iterdir())


__all__ = [
    "dataset_exists",
    "download_dataset",
]
