from __future__ import annotations

from pathlib import Path

import pytest

import handwriting_ai.training.resources as res


def test_detect_cpu_cores_fallback_to_os(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(_: Path) -> str | None:
        # Force all cgroup reads to fail or be invalid
        return "bad"

    monkeypatch.setattr(res, "_read_text_file", _read, raising=True)

    import os as _os

    # Patch cpu_count deterministically
    monkeypatch.setattr(_os, "cpu_count", lambda: 7, raising=True)
    assert res._detect_cpu_cores() == 7
