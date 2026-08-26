"""Tests for covenant_radar configuration loading."""

from __future__ import annotations

import pytest

from platform_core.config.covenant_radar import Settings, load_settings
from platform_core.testing import make_fake_env


def test_load_covenant_radar_settings_success() -> None:
    """Test loading covenant radar settings from environment."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test-redis:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host:5432/db")

    settings = load_settings()

    # Verify nested structure
    assert settings["redis"]["url"] == "redis://test-redis:6379/0"
    assert settings["database_url"] == "postgresql://user:pass@host:5432/db"
    # Verify defaults
    assert settings["app"]["models_root"] == "/data/models"
    assert settings["app"]["logs_root"] == "/data/logs"
    assert settings["app"]["data_root"] == "/data"
    assert settings["app"]["ml_backend"] == "xgboost"
    assert settings["app"]["active_model_path_xgb"] == "/data/models/active_xgb.ubj"
    assert settings["app"]["active_model_path_mlp"] == "/data/models/active_mlp.pt"
    assert settings["rq"]["queue_name"] == "covenant"
    assert settings["logging"]["level"] == "INFO"
    assert settings["app_env"] == "dev"


def test_load_covenant_radar_settings_uses_defaults() -> None:
    """Test load_covenant_radar_settings uses defaults when optional vars unset."""
    env = make_fake_env()
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    settings = load_settings()

    # Redis defaults to redis://redis:6379/0
    assert settings["redis"]["url"] == "redis://redis:6379/0"


def test_load_covenant_radar_settings_requires_database_url() -> None:
    """A missing DATABASE_URL fails fast instead of defaulting to an empty DSN.

    An empty DSN is not an error to libpq: it connects to the local socket
    using the PG* defaults. A deployment that omitted the variable would
    therefore attach to whatever Postgres happened to be reachable rather than
    failing, so the loader must reject it.
    """
    _env = make_fake_env()  # Install fake env but set no vars

    with pytest.raises(RuntimeError, match="Missing required env var: DATABASE_URL"):
        load_settings()


def test_load_covenant_radar_settings_rejects_blank_database_url() -> None:
    """A DATABASE_URL of only whitespace is rejected as well as an absent one."""
    env = make_fake_env()
    env.set("DATABASE_URL", "   ")

    with pytest.raises(RuntimeError, match="Empty env var: DATABASE_URL"):
        load_settings()


def test_load_covenant_radar_settings_custom_app_config() -> None:
    """Test load_covenant_radar_settings uses custom app config."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP__DATA_ROOT", "/custom/data")
    env.set("APP__MODELS_ROOT", "/custom/models")
    env.set("APP__LOGS_ROOT", "/custom/logs")
    env.set("APP__ACTIVE_MODEL_PATH_XGB", "/custom/models/my_xgb.ubj")
    env.set("APP__ACTIVE_MODEL_PATH_MLP", "/custom/models/my_mlp.pt")

    settings = load_settings()

    assert settings["app"]["data_root"] == "/custom/data"
    assert settings["app"]["models_root"] == "/custom/models"
    assert settings["app"]["logs_root"] == "/custom/logs"
    assert settings["app"]["active_model_path_xgb"] == "/custom/models/my_xgb.ubj"
    assert settings["app"]["active_model_path_mlp"] == "/custom/models/my_mlp.pt"


def test_load_covenant_radar_settings_custom_rq_config() -> None:
    """Test load_covenant_radar_settings uses custom RQ config."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("RQ__QUEUE_NAME", "custom-queue")
    env.set("RQ__JOB_TIMEOUT_SEC", "7200")
    env.set("RQ__RESULT_TTL_SEC", "172800")
    env.set("RQ__FAILURE_TTL_SEC", "1209600")

    settings = load_settings()

    assert settings["rq"]["queue_name"] == "custom-queue"
    assert settings["rq"]["job_timeout_sec"] == 7200
    assert settings["rq"]["result_ttl_sec"] == 172800
    assert settings["rq"]["failure_ttl_sec"] == 1209600


def test_load_covenant_radar_settings_reads_only_the_name_deployment_sets() -> None:
    """One spelling, and it is the one in this service's compose and .env.

    The dual read preferred `REDIS__URL` and fell back to `REDIS_URL`.
    Nothing in the repository sets `REDIS__URL` for this service --
    `services/covenant-radar-api/docker-compose.yml` and its `.env` both set
    `REDIS_URL` -- so the preferred branch was the dead one and the fallback
    was the only path ever taken.
    """
    env = make_fake_env()
    env.set("REDIS__URL", "redis://a-name-nothing-sets:6379/0")
    env.set("REDIS_URL", "redis://what-compose-actually-sets:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    settings = load_settings()

    assert settings["redis"]["url"] == "redis://what-compose-actually-sets:6379/0"


def test_load_covenant_radar_settings_prod_env() -> None:
    """Test load_covenant_radar_settings sets prod app_env."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP_ENV", "prod")

    settings = load_settings()

    assert settings["app_env"] == "prod"


def test_load_covenant_radar_settings_logging_levels() -> None:
    """Test load_covenant_radar_settings parses logging levels correctly."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    # Test DEBUG level
    env.set("LOGGING__LEVEL", "DEBUG")
    settings = load_settings()
    assert settings["logging"]["level"] == "DEBUG"

    # Test WARNING level
    env.set("LOGGING__LEVEL", "WARNING")
    settings = load_settings()
    assert settings["logging"]["level"] == "WARNING"

    # Test ERROR level
    env.set("LOGGING__LEVEL", "ERROR")
    settings = load_settings()
    assert settings["logging"]["level"] == "ERROR"

    # Test CRITICAL level
    env.set("LOGGING__LEVEL", "CRITICAL")
    settings = load_settings()
    assert settings["logging"]["level"] == "CRITICAL"


def test_covenant_radar_settings_is_typed_dict() -> None:
    """Test Settings is a proper TypedDict."""
    annotations = Settings.__annotations__
    assert "redis" in annotations
    assert "database_url" in annotations
    assert "app" in annotations
    assert "logging" in annotations
    assert "rq" in annotations
    assert "app_env" in annotations
    assert "datadog" in annotations


def test_load_covenant_radar_settings_mlp_backend() -> None:
    """Test load_covenant_radar_settings with MLP backend."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP__ML_BACKEND", "mlp")

    settings = load_settings()

    assert settings["app"]["ml_backend"] == "mlp"
    assert settings["app"]["active_model_path_xgb"] == "/data/models/active_xgb.ubj"
    assert settings["app"]["active_model_path_mlp"] == "/data/models/active_mlp.pt"


def test_load_covenant_radar_settings_lstm_backend() -> None:
    """Test load_covenant_radar_settings with LSTM backend."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP__ML_BACKEND", "lstm")

    settings = load_settings()

    assert settings["app"]["ml_backend"] == "lstm"


def test_load_covenant_radar_settings_lightgbm_backend() -> None:
    """Test load_covenant_radar_settings with LightGBM backend."""
    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP__ML_BACKEND", "lightgbm")

    settings = load_settings()

    assert settings["app"]["ml_backend"] == "lightgbm"


def test_load_covenant_radar_settings_invalid_backend_raises() -> None:
    """Test load_covenant_radar_settings raises for invalid backend."""
    import pytest

    env = make_fake_env()
    env.set("REDIS_URL", "redis://test:6379/0")
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("APP__ML_BACKEND", "invalid_backend")

    with pytest.raises(ValueError, match="must be 'xgboost', 'mlp', 'lstm', or 'lightgbm'"):
        load_settings()


def test_load_covenant_radar_settings_datadog_defaults() -> None:
    """Test load_covenant_radar_settings uses Datadog defaults."""
    env = make_fake_env()
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")

    settings = load_settings()

    assert settings["datadog"]["enabled"] is False
    assert settings["datadog"]["service"] == "covenant-radar-api"
    assert settings["datadog"]["env"] == "dev"
    assert settings["datadog"]["version"] == "0.0.0"
    assert settings["datadog"]["agent_host"] == "localhost"
    assert settings["datadog"]["dogstatsd_port"] == 8125
    assert settings["datadog"]["trace_enabled"] is True


def test_load_covenant_radar_settings_datadog_custom() -> None:
    """Test load_covenant_radar_settings uses custom Datadog config."""
    env = make_fake_env()
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("DATADOG__ENABLED", "true")
    env.set("DATADOG__SERVICE", "my-service")
    env.set("DATADOG__ENV", "production")
    env.set("DATADOG__VERSION", "2.0.0")
    env.set("DATADOG__AGENT_HOST", "datadog-agent")
    env.set("DATADOG__DOGSTATSD_PORT", "9125")
    env.set("DATADOG__TRACE_ENABLED", "false")

    settings = load_settings()

    assert settings["datadog"]["enabled"] is True
    assert settings["datadog"]["service"] == "my-service"
    assert settings["datadog"]["env"] == "production"
    assert settings["datadog"]["version"] == "2.0.0"
    assert settings["datadog"]["agent_host"] == "datadog-agent"
    assert settings["datadog"]["dogstatsd_port"] == 9125
    assert settings["datadog"]["trace_enabled"] is False


def test_load_covenant_radar_settings_datadog_staging_env() -> None:
    """Test load_covenant_radar_settings parses staging env."""
    env = make_fake_env()
    env.set("DATABASE_URL", "postgresql://user:pass@host/db")
    env.set("DATADOG__ENV", "staging")

    settings = load_settings()

    assert settings["datadog"]["env"] == "staging"


def test_load_covenant_radar_settings_datadog_invalid_env_raises() -> None:
    """Test load_covenant_radar_settings raises for invalid Datadog env."""
    import pytest

    env = make_fake_env()
    env.set("DATADOG__ENV", "invalid_env")

    with pytest.raises(ValueError, match="must be 'dev', 'staging', or 'production'"):
        load_settings()
