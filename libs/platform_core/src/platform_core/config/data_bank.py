from __future__ import annotations

from typing import TypedDict

from ._utils import (
    _optional_env_str,
    _parse_bool,
    _parse_int,
    _parse_str,
    _require_env_csv,
    _require_env_str,
)


class Settings(TypedDict):
    redis_url: str
    data_root: str
    min_free_gb: int
    delete_strict_404: bool
    max_file_bytes: int
    api_upload_keys: frozenset[str]
    api_read_keys: frozenset[str]
    api_delete_keys: frozenset[str]


def load_settings() -> Settings:
    redis_url = _require_env_str("REDIS_URL")
    upload_keys = _require_env_csv("API_UPLOAD_KEYS")
    read_keys = _optional_env_str("API_READ_KEYS")
    delete_keys = _optional_env_str("API_DELETE_KEYS")

    read_set = _require_env_csv("API_READ_KEYS") if read_keys is not None else upload_keys
    delete_set = _require_env_csv("API_DELETE_KEYS") if delete_keys is not None else upload_keys

    # These four were LITERALS, and three of them are set by the service's
    # own `railway.toml` -- DATA_ROOT, MIN_FREE_GB, DELETE_STRICT_404 -- so
    # the deployment has been configuring a loader that never read them. It
    # runs on MIN_FREE_GB=1 while its own manifest says 2.
    #
    # The literals become the defaults, so an environment that sets nothing
    # gets exactly what it got before. What changes is that setting them now
    # does something, and that `create_app` can be pointed somewhere
    # writable: with `/data/files` unconditional, building the app on any
    # host without a writable filesystem root raises PermissionError before
    # a single route is reachable, which is why this service's app-factory
    # test could only ever pass on Windows.
    return {
        "redis_url": redis_url,
        "data_root": _parse_str("DATA_ROOT", "/data/files"),
        "min_free_gb": _parse_int("MIN_FREE_GB", 1),
        "delete_strict_404": _parse_bool("DELETE_STRICT_404", False),
        "max_file_bytes": _parse_int("MAX_FILE_BYTES", 0),
        "api_upload_keys": upload_keys,
        "api_read_keys": read_set,
        "api_delete_keys": delete_set,
    }


__all__ = ["Settings", "load_settings"]
