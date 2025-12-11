from __future__ import annotations

from platform_core.testing import make_fake_env

from data_bank_api.config import settings_from_env


def test_settings_from_env_reads_values() -> None:
    env = make_fake_env()
    env.set("API_UPLOAD_KEYS", "u1,u2")
    env.set("REDIS_URL", "redis://ignored")
    s = settings_from_env()
    assert s["data_root"] == "/data/files"
    assert s["min_free_gb"] == 1
    assert s["delete_strict_404"] is False
    assert s["max_file_bytes"] == 0


def test_settings_api_keys_from_env() -> None:
    env = make_fake_env()
    env.set("API_UPLOAD_KEYS", "u1,u2")
    env.set("REDIS_URL", "redis://ignored")
    # leave read/delete unset to inherit from upload
    s = settings_from_env()
    assert s["api_upload_keys"] == frozenset({"u1", "u2"})
    assert s["api_read_keys"] == s["api_upload_keys"]
    assert s["api_delete_keys"] == s["api_upload_keys"]
