"""Tests for Art-Trainer settings configuration."""

from __future__ import annotations

from art_trainer.core.config.settings import Settings, load_settings
from tests.conftest import SettingsFactory


def test_load_settings_returns_settings() -> None:
    """Test that load_settings returns valid Settings."""
    settings = load_settings()
    assert settings["app_env"] in ("dev", "prod")
    assert settings["logging"]["level"] in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    assert settings["redis"]["url"] != ""
    assert settings["rq"]["queue_name"] != ""
    assert settings["app"]["data_root"] != ""
    assert settings["security"]["api_key"] == "" or len(settings["security"]["api_key"]) > 0


def test_settings_has_app_config(settings_factory: SettingsFactory) -> None:
    """Test that Settings contains app configuration."""
    settings: Settings = settings_factory()
    assert settings["app"]["data_root"] != ""
    assert settings["app"]["output_root"] != ""
    assert settings["app"]["logs_root"] != ""
    assert settings["app"]["kohya_ss_path"] != ""


def test_settings_factory_overrides(settings_factory: SettingsFactory) -> None:
    """Test that settings factory applies overrides."""
    settings: Settings = settings_factory(
        data_root="/custom/data",
        output_root="/custom/output",
        security_api_key="custom-key",
    )
    assert settings["app"]["data_root"] == "/custom/data"
    assert settings["app"]["output_root"] == "/custom/output"
    assert settings["security"]["api_key"] == "custom-key"
