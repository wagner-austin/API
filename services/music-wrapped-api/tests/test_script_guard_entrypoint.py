from __future__ import annotations

import runpy
import sys


def test_guard_entrypoint_runs_as_main() -> None:
    # Ensure a clean import to avoid RuntimeWarning about existing module
    sys.modules.pop("music_wrapped_api.asgi", None)
    runpy.run_module("music_wrapped_api.asgi", run_name="__main__")
