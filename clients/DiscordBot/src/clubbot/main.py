from __future__ import annotations

from platform_core.logging import setup_logging

from .container import ServiceContainer
from .orchestrator import BotOrchestrator


def main() -> None:
    container = ServiceContainer.from_env()
    setup_logging(
        level=container.cfg["discord"]["log_level"],
        format_mode="text",
        service_name="discordbot",
        instance_id=None,
        extra_fields=["request_id"],
    )
    BotOrchestrator(container).run()


if __name__ == "__main__":
    main()
