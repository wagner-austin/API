from __future__ import annotations

import importlib
from pathlib import Path

from procart.ffmpeg_runner import RealFfmpegRunner

from procart_api import _test_hooks as _hooks


def test_ffmpeg_runner_hook_is_bound_to_the_real_runner() -> None:
    """The hook holds the real runner at import time, with no startup assignment.

    Binding it in _test_hooks rather than in main() means every caller reaches a
    usable runner without a None check, and reloading main cannot change it.
    """
    assert type(_hooks.FFMPEG_RUNNER).__name__ == RealFfmpegRunner.__name__


def test_reloading_main_leaves_the_hook_bound() -> None:
    """main no longer assigns the hook, so a reload leaves the binding intact."""
    import procart_api.main as main_mod

    importlib.reload(main_mod)

    assert type(_hooks.FFMPEG_RUNNER).__name__ == RealFfmpegRunner.__name__


def test_main_module_main_block_executes() -> None:
    # Execute main.py with __name__ == "__main__" to cover the main guard.
    main_path = Path(__file__).resolve().parents[1] / "src" / "procart_api" / "main.py"
    code = main_path.read_text(encoding="utf-8")
    globals_dict = {"__name__": "__main__", "__file__": str(main_path)}
    exec(compile(code, str(main_path), "exec"), globals_dict, globals_dict)
