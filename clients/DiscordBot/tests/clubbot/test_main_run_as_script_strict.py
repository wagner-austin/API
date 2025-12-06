from __future__ import annotations

import runpy
import sys
from typing import Protocol

import pytest
from tests.support.settings import build_settings

from clubbot.config import DiscordbotSettings


class _ContainerProto(Protocol):
    cfg: DiscordbotSettings

    @classmethod
    def from_env(cls) -> _ContainerProto: ...


def test_run_as_main_with_injected_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a strictly typed config
    cfg = build_settings()
    marker: dict[str, bool] = {"ran": False}

    class _FakeContainer:
        def __init__(self, cfg: DiscordbotSettings) -> None:
            self.cfg = cfg

        @classmethod
        def from_env(cls) -> _FakeContainer:
            return cls(cfg)

    class _Orch:
        def __init__(self, container: _ContainerProto) -> None:
            self.container = container

        def run(self) -> None:
            marker["ran"] = True

    def _setup_logging(
        *,
        level: str,
        service_name: str,
        format_mode: str | None = None,
        instance_id: str | None = None,
        extra_fields: list[str] | None = None,
    ) -> None:
        _ = (level, service_name, format_mode, instance_id, extra_fields)
        return

    # Import real modules and monkeypatch attributes so names exist for mypy
    import platform_core.logging as plog

    import clubbot.container as cmod
    import clubbot.orchestrator as omod

    monkeypatch.setattr(plog, "setup_logging", _setup_logging, raising=True)
    monkeypatch.setattr(cmod, "ServiceContainer", _FakeContainer, raising=True)
    monkeypatch.setattr(omod, "BotOrchestrator", _Orch, raising=True)

    # Ensure a clean import to avoid runtime warnings about preloaded modules
    sys.modules.pop("clubbot.main", None)
    # Execute module as __main__ to cover guard
    runpy.run_module("clubbot.main", run_name="__main__")
    assert marker["ran"] is True
