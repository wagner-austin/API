from __future__ import annotations

from typing import TypedDict

from ._utils import _optional_env_str, _require_env_str


class TurkicApiSettings(TypedDict):
    redis_url: str
    data_dir: str
    environment: str
    data_bank_api_url: str
    data_bank_api_key: str


def load_turkic_api_settings() -> TurkicApiSettings:
    redis_env = _optional_env_str("TURKIC_REDIS_URL")
    data_dir_env = _optional_env_str("TURKIC_DATA_DIR")
    api_url_env = _optional_env_str("TURKIC_DATA_BANK_API_URL")
    api_key = _require_env_str("TURKIC_DATA_BANK_API_KEY")
    return {
        "redis_url": redis_env if redis_env is not None else "redis://redis:6379/0",
        "data_dir": data_dir_env if data_dir_env is not None else "/data",
        "environment": "local",
        "data_bank_api_url": api_url_env if api_url_env is not None else "",
        "data_bank_api_key": api_key,
    }


__all__ = ["TurkicApiSettings", "load_turkic_api_settings"]
