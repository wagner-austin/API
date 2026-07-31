"""The cross-batch ledger, driven against a scratch sweep root.

One row per match across every batch is the longitudinal record; what is
pinned here is that the row carries the batch, the arm/seed split, and the
scorecard columns in order -- and that the filter and the non-directory
clutter a real sweep root accumulates are both handled.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.ledger import COLUMNS, EXIT_BAD_USAGE, EXIT_OK, main


def _card(path: Path, values: dict[str, str]) -> None:
    """Write one scorecard the way the sweep does: 15-column labels."""
    lines = [f"### {path.stem}"]
    lines.extend(f"{label:<15}{value}" for label, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scratch_root(tmp_path: Path) -> Path:
    root = tmp_path / "sweeps"
    (root / "batch-a").mkdir(parents=True)
    (root / "batch-b").mkdir()
    # The clutter a real root carries: a stray file is not a batch.
    (root / "stray.txt").write_text("not a batch\n", encoding="utf-8")
    _card(
        root / "batch-a" / "alpha-s777.txt",
        {
            "verdict": "won (won)",
            "extractors": "0 -> 5",
            "income": "62/s",
            "army value": "500 -> 15050",
            "total worth": "3500 -> 25050",
            "best rival": "3500 -> 1950 (peak 15300, worst dip 13350)",
            "intercepted": "171",
            "raids": "0",
            "marches": "0",
            "samples seen": "2861",
        },
    )
    # A card missing most columns still gets a row: blanks, not a crash.
    _card(root / "batch-b" / "beta-s123.txt", {"verdict": "survived (sample_limit)"})
    return root


def test_extra_arguments_print_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["a", "b"]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: ledger [batch-filter]\n"


def test_every_batch_lands_in_one_table(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main([], root=_scratch_root(tmp_path)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split("\t") == ["batch", "arm", "seed", *COLUMNS]
    assert lines[1].split("\t") == [
        "batch-a",
        "alpha",
        "777",
        "won (won)",
        "0 -> 5",
        "62/s",
        "500 -> 15050",
        "3500 -> 25050",
        "3500 -> 1950 (peak 15300, worst dip 13350)",
        "171",
        "0",
        "0",
        "2861",
    ]
    # Absent labels are blank cells, and the row count says the stray file
    # contributed nothing.
    assert lines[2].split("\t") == [
        "batch-b",
        "beta",
        "123",
        "survived (sample_limit)",
        *([""] * 9),
    ]
    assert len(lines) == 3


def test_the_filter_selects_batches_by_substring(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["batch-b"], root=_scratch_root(tmp_path)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("batch-b\tbeta\t123\t")


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.ledger")
    sys.argv = ["ledger", "a", "b"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.ledger", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.ledger"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: ledger [batch-filter]\n"
