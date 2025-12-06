from __future__ import annotations

from typing import TypedDict

from ._utils import _optional_env_str, _require_env_csv, _require_env_str


class DataBankSettings(TypedDict):
    redis_url: str
    data_root: str
    min_free_gb: int
    delete_strict_404: bool
    max_file_bytes: int
    api_upload_keys: frozenset[str]
    api_read_keys: frozenset[str]
    api_delete_keys: frozenset[str]


def load_data_bank_settings() -> DataBankSettings:
    redis_url = _require_env_str("REDIS_URL")
    upload_keys = _require_env_csv("API_UPLOAD_KEYS")
    read_keys = _optional_env_str("API_READ_KEYS")
    delete_keys = _optional_env_str("API_DELETE_KEYS")

    read_set = _require_env_csv("API_READ_KEYS") if read_keys is not None else upload_keys
    delete_set = _require_env_csv("API_DELETE_KEYS") if delete_keys is not None else upload_keys

    return {
        "redis_url": redis_url,
        "data_root": "/data/files",
        "min_free_gb": 1,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": upload_keys,
        "api_read_keys": read_set,
        "api_delete_keys": delete_set,
    }


__all__ = ["DataBankSettings", "load_data_bank_settings"]
