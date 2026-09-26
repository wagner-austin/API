from __future__ import annotations

from fastapi import FastAPI
from platform_core.logging import LogFormat, LogLevel, setup_logging

from .api.main import AppDeps, create_app
from .settings import build_clients_from_env, build_config_from_env


def make_app_from_env() -> FastAPI:
    # Initialize centralized logging
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.JSON,
        service_name="transcript-api",
        instance_id=None,
        extra_fields=None,
    )

    cfg = build_config_from_env()
    clients = build_clients_from_env()
    deps = AppDeps(config=cfg, clients=clients)
    return create_app(deps)


__all__ = ["make_app_from_env"]
