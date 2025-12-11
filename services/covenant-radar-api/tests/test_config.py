"""Tests for configuration loading."""

from __future__ import annotations

from platform_core.testing import make_fake_env

from covenant_radar_api.core import Settings, settings_from_env
from covenant_radar_api.core.config import Settings as ConfigSettings


def test_settings_from_env_loads_required_vars() -> None:
    """Test settings_from_env loads from environment variables."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test-redis:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    s = settings_from_env()

    # Verify nested structure
    assert s["redis"]["url"] == "redis://test-redis:6379/0"
    assert s["database_url"] == "postgresql://user:pass@host/db"
    # Verify defaults
    assert s["app"]["models_root"] == "/data/models"
    assert s["app"]["logs_root"] == "/data/logs"
    assert s["app"]["data_root"] == "/data"
    assert s["rq"]["queue_name"] == "covenant"
    assert s["logging"]["level"] == "INFO"
    assert s["app_env"] == "dev"


def test_settings_from_env_uses_redis_url_fallback() -> None:
    """Test settings uses REDIS_URL when REDIS__URL not set."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://from-redis-url:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    s = settings_from_env()
    assert s["redis"]["url"] == "redis://from-redis-url:6379/0"


def test_settings_from_env_prefers_redis__url() -> None:
    """Test settings prefers REDIS__URL over REDIS_URL."""
    env = make_fake_env()
    env.set("REDIS__URL", "redis://from-redis-double-underscore:6379/0")
    env.set("REDIS_URL", "redis://from-redis-url:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    s = settings_from_env()
    assert s["redis"]["url"] == "redis://from-redis-double-underscore:6379/0"


def test_settings_from_env_custom_app_config() -> None:
    """Test settings_from_env uses custom app config when set."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP__DATA_ROOT", "/custom/data")
    env.set("APP__MODELS_ROOT", "/custom/models")
    env.set("APP__LOGS_ROOT", "/custom/logs")

    s = settings_from_env()

    assert s["app"]["data_root"] == "/custom/data"
    assert s["app"]["models_root"] == "/custom/models"
    assert s["app"]["logs_root"] == "/custom/logs"


def test_settings_from_env_custom_rq_config() -> None:
    """Test settings_from_env uses custom RQ config when set."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("RQ__QUEUE_NAME", "custom-queue")
    env.set("RQ__JOB_TIMEOUT_SEC", "7200")

    s = settings_from_env()

    assert s["rq"]["queue_name"] == "custom-queue"
    assert s["rq"]["job_timeout_sec"] == 7200


def test_settings_from_env_custom_logging_level() -> None:
    """Test settings_from_env uses custom logging level when set."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("LOGGING__LEVEL", "DEBUG")

    s = settings_from_env()

    assert s["logging"]["level"] == "DEBUG"


def test_settings_from_env_prod_env() -> None:
    """Test settings_from_env sets prod app_env."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP_ENV", "prod")

    s = settings_from_env()

    assert s["app_env"] == "prod"


def test_settings_type_is_typed_dict() -> None:
    """Test Settings type is a TypedDict."""
    assert Settings is ConfigSettings
