from __future__ import annotations

from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_container_has_char_lstm_backend(settings_with_paths: Settings) -> None:
    c = ServiceContainer.from_settings(settings_with_paths)
    b = c.model_registry.get("char_lstm")
    assert b.name() == "char_lstm"
