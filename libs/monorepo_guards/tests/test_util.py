from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from monorepo_guards.config import GuardConfig
from monorepo_guards.util import iter_py_files, read_lines


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_iter_py_files_excludes_cache_and_handles_missing_dirs(tmp_path: Path) -> None:
    root = tmp_path
    incl = root / "src"
    excl = incl / ".pytest_cache" / "skip.py"
    keep = incl / "keep.py"
    _write(excl, "x=1\n")
    _write(keep, "y=2\n")

    cfg = GuardConfig(
        root=root,
        directories=("src", "missing"),  # 'missing' directory does not exist
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )
    files = iter_py_files(cfg)
    # Only 'keep.py' should be included
    assert [p.name for p in files] == ["keep.py"]


def test_read_lines_raises_on_os_error(tmp_path: Path) -> None:
    path = tmp_path / "unreadable.py"
    path.write_text("x = 1\n", encoding="utf-8")

    with (
        patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
        pytest.raises(RuntimeError, match=r"failed to read.*unreadable\.py"),
    ):
        read_lines(path)
