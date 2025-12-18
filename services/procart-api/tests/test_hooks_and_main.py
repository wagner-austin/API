from __future__ import annotations

import importlib
from pathlib import Path

from procart_api import _test_hooks as _hooks


def test_hooks_default_and_main_startup_sets_runner() -> None:
    # Reset hook and reload main to exercise assignment path.
    _hooks.FFMPEG_RUNNER = None
    import procart_api.main as main_mod

    importlib.reload(main_mod)
    assert _hooks.FFMPEG_RUNNER is not None
    runner = _hooks.FFMPEG_RUNNER
    # Verify protocol-style attribute exists without using Any.
    has_attr = hasattr(runner, "encode_frames_to_video")
    assert has_attr is True


def test_main_module_main_block_executes() -> None:
    # Execute main.py with __name__ == "__main__" to cover the main guard.
    main_path = Path(__file__).resolve().parents[1] / "src" / "procart_api" / "main.py"
    code = main_path.read_text(encoding="utf-8")
    globals_dict = {"__name__": "__main__", "__file__": str(main_path)}
    exec(compile(code, str(main_path), "exec"), globals_dict, globals_dict)
