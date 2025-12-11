"""Tests for covenant_radar configuration loading."""

from __future__ import annotations

from platform_core.config import CovenantRadarSettings, load_covenant_radar_settings
from platform_core.testing import make_fake_env


def test_load_covenant_radar_settings_success() -> None:
    """Test loading covenant radar settings from environment."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test-redis:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host:5432/db")

    settings = load_covenant_radar_settings()

    # Verify nested structure
    assert settings["redis"]["url"] == "redis://test-redis:6379/0"
    assert settings["database_url"] == "postgresql://user:pass@host:5432/db"
    # Verify defaults
    assert settings["app"]["models_root"] == "/data/models"
    assert settings["app"]["logs_root"] == "/data/logs"
    assert settings["app"]["data_root"] == "/data"
    assert settings["app"]["active_model_path"] == "/data/models/active.ubj"
    assert settings["rq"]["queue_name"] == "covenant"
    assert settings["logging"]["level"] == "INFO"
    assert settings["app_env"] == "dev"


def test_load_covenant_radar_settings_uses_defaults() -> None:
    """Test load_covenant_radar_settings uses defaults when env vars not set."""
    _env = make_fake_env()  # Install fake env but don't set any vars
    # Neither REDIS_URL nor DATABASE_URL set - uses defaults

    settings = load_covenant_radar_settings()

    # Redis defaults to redis://redis:6379/0
    assert settings["redis"]["url"] == "redis://redis:6379/0"
    # DATABASE_URL defaults to empty string
    assert settings["database_url"] == ""


def test_load_covenant_radar_settings_custom_app_config() -> None:
    """Test load_covenant_radar_settings uses custom app config."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP__DATA_ROOT", "/custom/data")
    env.set("APP__MODELS_ROOT", "/custom/models")
    env.set("APP__LOGS_ROOT", "/custom/logs")
    env.set("APP__ACTIVE_MODEL_PATH", "/custom/models/my_model.ubj")

    settings = load_covenant_radar_settings()

    assert settings["app"]["data_root"] == "/custom/data"
    assert settings["app"]["models_root"] == "/custom/models"
    assert settings["app"]["logs_root"] == "/custom/logs"
    assert settings["app"]["active_model_path"] == "/custom/models/my_model.ubj"


def test_load_covenant_radar_settings_custom_rq_config() -> None:
    """Test load_covenant_radar_settings uses custom RQ config."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("RQ__QUEUE_NAME", "custom-queue")
    env.set("RQ__JOB_TIMEOUT_SEC", "7200")
    env.set("RQ__RESULT_TTL_SEC", "172800")
    env.set("RQ__FAILURE_TTL_SEC", "1209600")

    settings = load_covenant_radar_settings()

    assert settings["rq"]["queue_name"] == "custom-queue"
    assert settings["rq"]["job_timeout_sec"] == 7200
    assert settings["rq"]["result_ttl_sec"] == 172800
    assert settings["rq"]["failure_ttl_sec"] == 1209600


def test_load_covenant_radar_settings_prefers_redis__url() -> None:
    """Test REDIS__URL takes precedence over REDIS_URL."""
    env = make_fake_env()
    env.set("REDIS__URL", "redis://from-double-underscore:6379/0")
    env.set("REDIS_URL", "redis://from-single-underscore:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    settings = load_covenant_radar_settings()

    assert settings["redis"]["url"] == "redis://from-double-underscore:6379/0"


def test_load_covenant_radar_settings_prod_env() -> None:
    """Test load_covenant_radar_settings sets prod app_env."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP_ENV", "prod")

    settings = load_covenant_radar_settings()

    assert settings["app_env"] == "prod"


def test_load_covenant_radar_settings_logging_levels() -> None:
    """Test load_covenant_radar_settings parses logging levels correctly."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    # Test DEBUG level
    env.set("LOGGING__LEVEL", "DEBUG")
    settings = load_covenant_radar_settings()
    assert settings["logging"]["level"] == "DEBUG"

    # Test WARNING level
    env.set("LOGGING__LEVEL", "WARNING")
    settings = load_covenant_radar_settings()
    assert settings["logging"]["level"] == "WARNING"

    # Test ERROR level
    env.set("LOGGING__LEVEL", "ERROR")
    settings = load_covenant_radar_settings()
    assert settings["logging"]["level"] == "ERROR"

    # Test CRITICAL level
    env.set("LOGGING__LEVEL", "CRITICAL")
    settings = load_covenant_radar_settings()
    assert settings["logging"]["level"] == "CRITICAL"


def test_covenant_radar_settings_is_typed_dict() -> None:
    """Test CovenantRadarSettings is a proper TypedDict."""
    annotations = CovenantRadarSettings.__annotations__
    assert "redis" in annotations
    assert "database_url" in annotations
    assert "app" in annotations
    assert "logging" in annotations
    assert "rq" in annotations
    assert "app_env" in annotations
