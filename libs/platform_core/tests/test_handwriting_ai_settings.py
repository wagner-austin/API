from __future__ import annotations

from pathlib import Path

import pytest

from platform_core.config.handwriting_ai import (
    HandwritingAiSettings,
    _apply_app_env,
    _apply_digits_env,
    _apply_security_env,
    _base_settings,
    _bool_env,
    _finalize,
    limits_from_handwriting_ai_settings,
    load_handwriting_ai_settings,
)
from platform_core.testing import make_fake_env


def test_bool_env_returns_default_when_none() -> None:
    assert _bool_env(None, True) is True
    assert _bool_env(None, False) is False


def test_bool_env_parses_true_variants() -> None:
    for val in ["1", "true", "TRUE", "yes", "YES", "y", "Y", "on", "ON", " true ", " 1 "]:
        assert _bool_env(val, False) is True


def test_bool_env_parses_false_variants() -> None:
    for val in ["0", "false", "FALSE", "no", "NO", "n", "N", "off", "OFF", " false ", " 0 "]:
        assert _bool_env(val, True) is False


def test_bool_env_raises_on_invalid() -> None:
    with pytest.raises(ValueError):
        _bool_env("invalid", False)


def test_base_settings_creates_dirs(tmp_path: Path) -> None:
    env = make_fake_env()
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    logs_root = tmp_path / "logs"
    model_dir = tmp_path / "models"

    env.set("HANDWRITING_DATA_ROOT", str(data_root))
    env.set("HANDWRITING_ARTIFACTS_ROOT", str(artifacts_root))
    env.set("HANDWRITING_LOGS_ROOT", str(logs_root))
    env.set("HANDWRITING_MODEL_DIR", str(model_dir))
    env.set("HANDWRITING_MODEL_ID", "test-model")
    env.set("HANDWRITING_API_KEY", "test-key")

    assert not data_root.exists()
    assert not artifacts_root.exists()
    assert not logs_root.exists()
    assert not model_dir.exists()

    settings = _base_settings(create_dirs=True)

    assert data_root.exists()
    assert artifacts_root.exists()
    assert logs_root.exists()
    assert model_dir.exists()
    assert settings["app"]["data_root"] == data_root.resolve()
    assert settings["app"]["artifacts_root"] == artifacts_root.resolve()
    assert settings["app"]["logs_root"] == logs_root.resolve()
    assert settings["digits"]["model_dir"] == model_dir.resolve()


def test_base_settings_skips_create_dirs(tmp_path: Path) -> None:
    env = make_fake_env()
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    logs_root = tmp_path / "logs"

    env.set("HANDWRITING_DATA_ROOT", str(data_root))
    env.set("HANDWRITING_ARTIFACTS_ROOT", str(artifacts_root))
    env.set("HANDWRITING_LOGS_ROOT", str(logs_root))
    env.set("HANDWRITING_MODEL_ID", "test-model")
    env.set("HANDWRITING_API_KEY", "test-key")

    settings = _base_settings(create_dirs=False)

    assert not data_root.exists()
    assert not artifacts_root.exists()
    assert not logs_root.exists()
    assert settings["app"]["data_root"] == data_root.resolve()


def test_base_settings_with_allowed_hosts(tmp_path: Path) -> None:
    env = make_fake_env()
    env.set("HANDWRITING_DATA_ROOT", str(tmp_path / "data"))
    env.set("HANDWRITING_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    env.set("HANDWRITING_LOGS_ROOT", str(tmp_path / "logs"))
    env.set("HANDWRITING_MODEL_ID", "test-model")
    env.set("HANDWRITING_API_KEY", "test-key")
    env.set("HANDWRITING_ALLOWED_HOSTS", "host1.com, host2.com, host3.com")

    settings = _base_settings(create_dirs=False)

    assert settings["digits"]["allowed_hosts"] == frozenset({"host1.com", "host2.com", "host3.com"})


def test_base_settings_without_allowed_hosts(tmp_path: Path) -> None:
    env = make_fake_env()
    env.set("HANDWRITING_DATA_ROOT", str(tmp_path / "data"))
    env.set("HANDWRITING_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    env.set("HANDWRITING_LOGS_ROOT", str(tmp_path / "logs"))
    env.set("HANDWRITING_MODEL_ID", "test-model")
    env.set("HANDWRITING_API_KEY", "test-key")
    # HANDWRITING_ALLOWED_HOSTS not set

    settings = _base_settings(create_dirs=False)

    assert settings["digits"]["allowed_hosts"] == frozenset()


def test_apply_app_env_overrides_all_fields(tmp_path: Path) -> None:
    env = make_fake_env()
    from platform_core.config.handwriting_ai import HandwritingAiAppConfig

    base_app: HandwritingAiAppConfig = {
        "data_root": Path("/base/data"),
        "artifacts_root": Path("/base/artifacts"),
        "logs_root": Path("/base/logs"),
        "threads": 0,
        "port": 8081,
    }

    env.set("APP__DATA_ROOT", str(tmp_path / "override_data"))
    env.set("APP__ARTIFACTS_ROOT", str(tmp_path / "override_artifacts"))
    env.set("APP__LOGS_ROOT", str(tmp_path / "override_logs"))
    env.set("APP__THREADS", "4")
    env.set("APP__PORT", "9000")

    result = _apply_app_env(base_app)

    assert result["data_root"] == (tmp_path / "override_data").resolve()
    assert result["artifacts_root"] == (tmp_path / "override_artifacts").resolve()
    assert result["logs_root"] == (tmp_path / "override_logs").resolve()
    assert result["threads"] == 4
    assert result["port"] == 9000


def test_apply_app_env_port_out_of_range_raises() -> None:
    env = make_fake_env()
    from platform_core.config.handwriting_ai import HandwritingAiAppConfig

    base_app: HandwritingAiAppConfig = {"port": 8081}

    env.set("APP__PORT", "0")
    with pytest.raises(RuntimeError):
        _apply_app_env(base_app)

    env.set("APP__PORT", "65536")
    with pytest.raises(RuntimeError):
        _apply_app_env(base_app)


def test_apply_digits_env_overrides_all_fields(tmp_path: Path) -> None:
    env = make_fake_env()
    from platform_core.config.handwriting_ai import HandwritingAiDigitsConfig

    base_digits: HandwritingAiDigitsConfig = {
        "model_dir": Path("/base/models"),
        "active_model": "base-model",
        "tta": False,
        "uncertain_threshold": 0.7,
        "max_image_mb": 10,
        "max_image_side_px": 2048,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 64,
        "retention_keep_runs": 5,
    }

    env.set("DIGITS__MODEL_DIR", str(tmp_path / "override_models"))
    env.set("DIGITS__ACTIVE_MODEL", "override-model")
    env.set("DIGITS__TTA", "true")
    env.set("DIGITS__UNCERTAIN_THRESHOLD", "0.85")
    env.set("DIGITS__MAX_IMAGE_MB", "20")
    env.set("DIGITS__MAX_IMAGE_SIDE_PX", "4096")
    env.set("DIGITS__PREDICT_TIMEOUT_SECONDS", "10")
    env.set("DIGITS__VISUALIZE_MAX_KB", "128")
    env.set("DIGITS__RETENTION_KEEP_RUNS", "10")

    result = _apply_digits_env(base_digits)

    assert result["model_dir"] == (tmp_path / "override_models").resolve()
    assert result["active_model"] == "override-model"
    assert result["tta"] is True
    assert result["uncertain_threshold"] == 0.85
    assert result["max_image_mb"] == 20
    assert result["max_image_side_px"] == 4096
    assert result["predict_timeout_seconds"] == 10
    assert result["visualize_max_kb"] == 128
    assert result["retention_keep_runs"] == 10


def test_apply_security_env_overrides_all_fields() -> None:
    env = make_fake_env()
    from platform_core.config.handwriting_ai import HandwritingAiSecurityConfig

    base_sec: HandwritingAiSecurityConfig = {
        "api_key": "base-key",
        "api_key_enabled": True,
    }

    env.set("SECURITY__API_KEY", "override-key")
    env.set("SECURITY__API_KEY_ENABLED", "false")

    result = _apply_security_env(base_sec)

    assert result["api_key"] == "override-key"
    assert result["api_key_enabled"] is False


def test_finalize_disabled_api_key_clears_key() -> None:
    settings: HandwritingAiSettings = {
        "app": {},
        "digits": {},
        "security": {"api_key": "secret", "api_key_enabled": False},
    }

    result = _finalize(settings)

    assert result["security"]["api_key"] == ""
    assert result["security"]["api_key_enabled"] is False


def test_finalize_enabled_api_key_preserves_key() -> None:
    settings: HandwritingAiSettings = {
        "app": {},
        "digits": {},
        "security": {"api_key": "secret", "api_key_enabled": True},
    }

    result = _finalize(settings)

    assert result["security"]["api_key"] == "secret"
    assert result["security"]["api_key_enabled"] is True


def test_finalize_missing_api_key_enabled_defaults_to_enabled() -> None:
    settings: HandwritingAiSettings = {
        "app": {},
        "digits": {},
        "security": {"api_key": "secret"},
    }

    result = _finalize(settings)

    assert result["security"]["api_key"] == "secret"
    assert result["security"]["api_key_enabled"] is True


def test_load_handwriting_ai_settings_integration(tmp_path: Path) -> None:
    env = make_fake_env()
    env.set("HANDWRITING_DATA_ROOT", str(tmp_path / "data"))
    env.set("HANDWRITING_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    env.set("HANDWRITING_LOGS_ROOT", str(tmp_path / "logs"))
    env.set("HANDWRITING_MODEL_ID", "test-model")
    env.set("HANDWRITING_API_KEY", "test-key")

    settings = load_handwriting_ai_settings(create_dirs=True)

    assert settings["app"]["data_root"] == (tmp_path / "data").resolve()
    assert settings["app"]["artifacts_root"] == (tmp_path / "artifacts").resolve()
    assert settings["app"]["logs_root"] == (tmp_path / "logs").resolve()
    assert settings["digits"]["active_model"] == "test-model"
    assert settings["security"]["api_key"] == "test-key"
    assert settings["security"]["api_key_enabled"] is True


def test_load_handwriting_ai_settings_with_overrides(tmp_path: Path) -> None:
    env = make_fake_env()
    env.set("HANDWRITING_DATA_ROOT", str(tmp_path / "data"))
    env.set("HANDWRITING_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    env.set("HANDWRITING_LOGS_ROOT", str(tmp_path / "logs"))
    env.set("HANDWRITING_MODEL_ID", "test-model")
    env.set("HANDWRITING_API_KEY", "test-key")
    env.set("APP__PORT", "9090")
    env.set("DIGITS__TTA", "true")
    env.set("SECURITY__API_KEY_ENABLED", "false")

    settings = load_handwriting_ai_settings(create_dirs=False)

    assert settings["app"]["port"] == 9090
    assert settings["digits"]["tta"] is True
    assert settings["security"]["api_key"] == ""
    assert settings["security"]["api_key_enabled"] is False


def test_limits_from_handwriting_ai_settings() -> None:
    settings: HandwritingAiSettings = {
        "app": {},
        "digits": {"max_image_mb": 15, "max_image_side_px": 3000},
        "security": {},
    }

    limits = limits_from_handwriting_ai_settings(settings)

    assert limits["max_bytes"] == 15 * 1024 * 1024
    assert limits["max_side_px"] == 3000


def test_limits_from_handwriting_ai_settings_uses_defaults() -> None:
    settings: HandwritingAiSettings = {
        "app": {},
        "digits": {},
        "security": {},
    }

    limits = limits_from_handwriting_ai_settings(settings)

    assert limits["max_bytes"] == 10 * 1024 * 1024  # DEFAULT_MAX_IMAGE_MB
    assert limits["max_side_px"] == 2048  # DEFAULT_MAX_IMAGE_SIDE_PX


def test_apply_app_env_data_bank_overrides() -> None:
    env = make_fake_env()
    from platform_core.config.handwriting_ai import HandwritingAiAppConfig, _apply_app_env

    base_app: HandwritingAiAppConfig = {"threads": 0, "port": 8081}
    env.set("APP__DATA_BANK_API_URL", "http://db")
    env.set("APP__DATA_BANK_API_KEY", "secret")

    result = _apply_app_env(base_app)
    assert result["data_bank_api_url"] == "http://db"
    assert result["data_bank_api_key"] == "secret"
