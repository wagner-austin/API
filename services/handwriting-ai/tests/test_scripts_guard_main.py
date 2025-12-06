from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    # tests/ -> handwriting-ai/
    return Path(__file__).resolve().parents[1]


def test_guard_main_with_empty_root_succeeds(tmp_path: Path) -> None:
    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr

    assert result.returncode == 0
    assert "Guard rule summary" in out
