"""The calibration's file hooks, run for real through their failures.

Every calibration test replaces ``emit_result_file`` and ``remove_temp_tree``
with fakes or reaches them only on the happy path, so none shows the
production writer refusing a directory that is gone, or the production
remover refusing a tree that is already removed. These call the
implementations themselves (``effect-seam-twin``, board task cc7222ca).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_ai._hook_defaults_training import (
    _default_emit_result_file,
    _default_remove_temp_tree,
    _default_tempfile_mkdtemp,
)
from handwriting_ai._hook_protocols_training import CalibrationRunnerResultDict

_RESULT: CalibrationRunnerResultDict = {
    "intra_threads": 4,
    "interop_threads": None,
    "num_workers": 2,
    "batch_size": 64,
    "samples_per_sec": 812.5,
    "p95_ms": 31.25,
}


def test_the_result_file_is_written_whole_and_its_temp_is_gone(tmp_path: Path) -> None:
    """The parent polls for this file, so it must appear complete or not at all."""
    out_path = tmp_path / "result.kv"

    _default_emit_result_file(str(out_path), _RESULT)

    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "ok=1",
        "intra_threads=4",
        "interop_threads=",
        "num_workers=2",
        "batch_size=64",
        "samples_per_sec=812.5",
        "p95_ms=31.25",
    ]
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["result.kv"]


def test_a_result_file_whose_directory_is_gone_is_raised(tmp_path: Path) -> None:
    """The parent removing the work directory early must not read as success."""
    out_path = tmp_path / "removed" / "result.kv"

    with pytest.raises(FileNotFoundError):
        _default_emit_result_file(str(out_path), _RESULT)

    assert not out_path.parent.exists()


def test_the_temp_tree_is_removed_with_what_it_holds() -> None:
    work = Path(_default_tempfile_mkdtemp("handwriting-calibration-"))
    (work / "result.kv").write_text("ok=1", encoding="utf-8")

    _default_remove_temp_tree(str(work))

    assert not work.exists()


def test_removing_a_temp_tree_that_is_already_gone_is_raised(tmp_path: Path) -> None:
    """A second removal means two owners thought they held the directory."""
    gone = tmp_path / "handwriting-calibration-gone"

    with pytest.raises(FileNotFoundError):
        _default_remove_temp_tree(str(gone))
