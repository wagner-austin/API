from __future__ import annotations

from platform_core.logging import setup_logging
from platform_core.request_context import install_request_id_middleware
from procart.ffmpeg_runner import RealFfmpegRunner

from procart_api import _test_hooks as _hooks
from procart_api.app import create_app


def _set_hooks() -> None:
    # Assign real implementations at startup, via hooks module.
    _hooks.FFMPEG_RUNNER = RealFfmpegRunner()


def _setup_logging() -> None:
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="procart-api",
        instance_id=None,
        extra_fields=None,
    )


_setup_logging()
_set_hooks()
app = create_app()
# Installed here rather than inside create_app: a second install would add a
# second middleware layer, and with no inbound x-request-id header each layer
# mints its own id — the response header would then disagree with the logs.
install_request_id_middleware(app)


if __name__ == "__main__":
    # Hypercorn entry: hypercorn services.procart-api.src.procart_api.main:app
    # Keep minimal; local runs should prefer Hypercorn CLI.
    pass
