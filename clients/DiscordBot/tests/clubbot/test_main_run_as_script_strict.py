from __future__ import annotations

import runpy
import sys

from platform_core.logging import LogFormat, LogLevel
from tests.support.settings import build_settings

from clubbot import _test_hooks


def test_run_as_main_with_injected_dependencies() -> None:
    # Build a strictly typed config
    cfg = build_settings()
    marker: dict[str, bool] = {"ran": False}

    class _FakeContainer:
        def __init__(self) -> None:
            self.cfg = cfg

        @classmethod
        def from_env(cls) -> _test_hooks.ServiceContainerProtocol:
            return cls()

    class _FakeOrchestrator:
        def __init__(self, container: _test_hooks.ServiceContainerProtocol) -> None:
            self.container = container

        def run(self) -> None:
            marker["ran"] = True

    def _fake_setup_logging(
        *,
        level: LogLevel,
        service_name: str,
        format_mode: LogFormat,
        instance_id: str | None = None,
        extra_fields: list[str] | None = None,
    ) -> None:
        _ = (level, service_name, format_mode, instance_id, extra_fields)
        return

    def _fake_create_container() -> _test_hooks.ServiceContainerProtocol:
        result: _test_hooks.ServiceContainerProtocol = _FakeContainer()
        return result

    def _fake_create_orchestrator(
        container: _test_hooks.ServiceContainerProtocol,
    ) -> _test_hooks.BotOrchestratorProtocol:
        result: _test_hooks.BotOrchestratorProtocol = _FakeOrchestrator(container)
        return result

    _test_hooks.setup_logging = _fake_setup_logging
    _test_hooks.create_service_container = _fake_create_container
    _test_hooks.create_bot_orchestrator = _fake_create_orchestrator

    # Ensure a clean import to avoid runtime warnings about preloaded modules
    sys.modules.pop("clubbot.main", None)
    # Execute module as __main__ to cover guard
    runpy.run_module("clubbot.main", run_name="__main__")
    assert marker["ran"] is True
