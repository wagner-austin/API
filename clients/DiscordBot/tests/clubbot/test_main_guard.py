from __future__ import annotations

from typing import Protocol

import pytest
from tests.support.settings import build_settings

from clubbot.config import DiscordbotSettings


class _ContainerProto(Protocol):
    cfg: DiscordbotSettings


class _FakeContainer:
    def __init__(self, cfg: DiscordbotSettings) -> None:
        self.cfg = cfg

    @classmethod
    def from_env(cls) -> _FakeContainer:
        cfg = build_settings()
        return cls(cfg=cfg)


class _FakeOrchestrator:
    def __init__(self, container: _ContainerProto) -> None:
        self.container = container
        self._ran = False

    def run(self) -> None:
        self._ran = True


def test_main_executes_and_calls_setup_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}

    def _fake_setup_logging(
        *,
        level: str,
        service_name: str,
        format_mode: str | None = None,
        instance_id: str | None = None,
        extra_fields: list[str] | None = None,
    ) -> None:
        _ = (format_mode, instance_id, extra_fields)
        called["level"] = level
        called["service_name"] = service_name

    ran: dict[str, bool] = {"ok": False}

    class _TrackedOrchestrator(_FakeOrchestrator):
        def run(self) -> None:
            ran["ok"] = True

    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setattr("clubbot.main.ServiceContainer", _FakeContainer)
    monkeypatch.setattr("clubbot.main.BotOrchestrator", _TrackedOrchestrator)
    monkeypatch.setattr("clubbot.main.setup_logging", _fake_setup_logging)

    from clubbot import main as main_mod

    main_mod.main()

    assert ran["ok"] is True
    assert called["level"] == "INFO"
    assert called["service_name"] == "discordbot"
