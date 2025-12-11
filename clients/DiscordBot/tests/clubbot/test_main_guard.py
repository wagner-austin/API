from __future__ import annotations

from platform_core.logging import LogFormat, LogLevel
from tests.support.settings import build_settings

from clubbot import _test_hooks


def test_main_executes_and_calls_setup_logging() -> None:
    called: dict[str, str] = {}
    ran: dict[str, bool] = {"ok": False}

    def _fake_setup_logging(
        *,
        level: LogLevel,
        service_name: str,
        format_mode: LogFormat,
        instance_id: str | None = None,
        extra_fields: list[str] | None = None,
    ) -> None:
        _ = (format_mode, instance_id, extra_fields)
        called["level"] = level
        called["service_name"] = service_name

    cfg = build_settings()

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
            ran["ok"] = True

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

    from clubbot import main as main_mod

    main_mod.main()

    assert ran["ok"] is True
    assert called["level"] == "INFO"
    assert called["service_name"] == "discordbot"
