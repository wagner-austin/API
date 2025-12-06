from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _project_root() -> Path:
    # tests/ -> transcript-api/
    return Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_BAD_SNIPPET = (
    "from typing import Any\n"
    "x: Any = 1  # type: ignore\n"
    "from typing import cast\n"
    "y = cast(int, 1)\n"
    "import contextlib\n"
    "with contextlib.suppress(Exception):\n"
    "    pass\n"
    "try:\n"
    "    1/0\n"
    "except Exception as exc:\n"
    "    raise RuntimeError('fail') from exc\n"
)


def test_guard_detects_violations(tmp_path: Path) -> None:
    root = tmp_path
    src = root / "src"
    bad = src / "bad.py"

    _write(bad, _BAD_SNIPPET)

    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(root)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Guard rule summary" in out
    assert "Guard checks failed" in out


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_guard_main_direct_violations(tmp_path: Path) -> None:
    from scripts import guard as guard_mod

    root = tmp_path
    src = root / "src"
    bad = src / "bad.py"

    _write(bad, _BAD_SNIPPET)

    rc = guard_mod.main(["--root", str(root)])
    assert rc != 0


def test_guard_main_direct_clean(tmp_path: Path) -> None:
    from scripts import guard as guard_mod

    rc = guard_mod.main(["--root", str(tmp_path), "--verbose"])
    assert rc == 0


def test_guard_main_unknown_flag_is_ignored(tmp_path: Path) -> None:
    from scripts import guard as guard_mod

    # Extra positional token should be skipped by the parser
    rc = guard_mod.main(["--root", str(tmp_path), "ignored-flag"])
    assert rc == 0


def test_transcript_api_package_importable() -> None:
    import transcript_api as pkg

    assert pkg.__name__ == "transcript_api"


def test_find_monorepo_root_raises_when_missing(tmp_path: Path) -> None:
    from scripts import guard as guard_mod

    with pytest.raises(RuntimeError):
        _ = guard_mod._find_monorepo_root(tmp_path)
