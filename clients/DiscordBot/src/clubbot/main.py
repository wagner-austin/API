from __future__ import annotations

from . import _test_hooks


def main() -> None:
    container = _test_hooks.create_service_container()
    _test_hooks.setup_logging(
        level=container.cfg["discord"]["log_level"],
        format_mode="text",
        service_name="discordbot",
        instance_id=None,
        extra_fields=["request_id"],
    )
    orchestrator = _test_hooks.create_bot_orchestrator(container)
    orchestrator.run()


if __name__ == "__main__":
    main()
