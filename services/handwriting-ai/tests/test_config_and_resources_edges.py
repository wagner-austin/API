from __future__ import annotations

from pathlib import Path

import handwriting_ai.training.resources as res
from handwriting_ai import _test_hooks


def test_detect_cpu_cores_fallback_to_os() -> None:
    def _read(path: Path) -> str:
        _ = path
        # Force all cgroup reads to fail or be invalid
        return "bad"

    _test_hooks.read_text_file = _read

    # Patch cpu_count deterministically via hook
    _test_hooks.os_cpu_count = lambda: 7
    assert res._detect_cpu_cores() == 7
