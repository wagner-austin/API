from __future__ import annotations

from platform_core.logging import setup_logging
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


if __name__ == "__main__":
    # Hypercorn entry: hypercorn services.procart-api.src.procart_api.main:app
    # Keep minimal; local runs should prefer Hypercorn CLI.
    pass
