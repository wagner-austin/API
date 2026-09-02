"""The paired-seed reader: the comparison chaos actually permits.

Panel totals of 13W and 14W read as noise; the same panels seed by seed
can carry a decisive verdict. These pin the pairing, the flip counts, and
the refusal to silently drop a seed only one side played.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.pairs import EXIT_BAD_USAGE, EXIT_NO_BATCH, EXIT_OK, main


def _card(root: Path, batch: str, seed: int, verdict: str, label: str = "arm") -> None:
    folder = root / batch
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{label}-s{seed}.txt").write_text(
        f"### {label}-s{seed}\nverdict        {verdict} ({verdict})\n", encoding="utf-8"
    )


@pytest.mark.parametrize("argv", [[], ["a"], ["a", "b", "c"]])
def test_a_bad_argument_count_prints_usage(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: pairs")


def test_the_module_entry_point_exits_with_the_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.pairs")
    sys.argv = ["pairs"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.pairs", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.pairs"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: pairs")


def test_flips_are_counted_by_direction_and_named_seed_by_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point: 2W vs 2W in totals, decisively different in pairs."""
    for seed, verdict in ((1, "won"), (2, "wiped"), (3, "won"), (4, "defeated")):
        _card(tmp_path, "old", seed, verdict)
    for seed, verdict in ((1, "won"), (2, "won"), (3, "wiped"), (4, "survived")):
        _card(tmp_path, "new", seed, verdict)
    assert main(["old", "new"], root=tmp_path) == EXIT_OK
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "paired 4 seed(s): old (left) vs new (right)",
        "wins   2 -> 2",
        "flips  1 to-W against 1 from-W (net +0 for new)",
        "p      1.000 two-sided binomial on 2 discordant",
        "  s2: L -> W",
        "  s3: W -> L",
        "  s4: L -> S",
    ]


def test_a_seed_only_one_side_played_is_reported_not_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _card(tmp_path, "old", 1, "won")
    _card(tmp_path, "old", 2, "won")
    _card(tmp_path, "new", 1, "won")
    _card(tmp_path, "new", 9, "wiped")
    assert main(["old", "new"], root=tmp_path) == EXIT_OK
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "paired 1 seed(s): old (left) vs new (right)",
        "wins   1 -> 1",
        "flips  0 to-W against 0 from-W (net +0 for new)",
        "p      1.000 (0 discordant pairs)",
        "unpaired  1 only in old, 1 only in new",
    ]


def test_two_labels_in_one_interleaved_batch_pair_against_each_other(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The close48 shape: control and arm on the same seeds in ONE
    directory, filenames differing only in the label prefix. A decisive
    to-W flip must surface, and the other label's cards must not leak
    into either side."""
    for seed, verdict in ((1, "won"), (2, "survived"), (3, "won")):
        _card(tmp_path, "mix", seed, verdict, label="control")
    for seed, verdict in ((1, "won"), (2, "won"), (3, "won")):
        _card(tmp_path, "mix", seed, verdict, label="close")
    assert main(["mix:control", "mix:close"], root=tmp_path) == EXIT_OK
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "paired 3 seed(s): mix:control (left) vs mix:close (right)",
        "wins   2 -> 3",
        "flips  1 to-W against 0 from-W (net +1 for mix:close)",
        "p      1.000 two-sided binomial on 1 discordant",
        "  s2: S -> W",
    ]


def test_missing_or_empty_batches_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _card(tmp_path, "old", 1, "won")
    assert main(["old", "ghost"], root=tmp_path) == EXIT_NO_BATCH
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "no scorecards: old has 1, ghost has 0"


def test_a_card_without_a_verdict_is_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crashed match writes a partial card; it must not pair as a loss.

    Nor may a verdict outside the grade table -- an unknown outcome word is
    skipped rather than guessed at, and the seed reads as unplayed."""
    _card(tmp_path, "old", 0, "abandoned")
    _card(tmp_path, "old", 1, "won")
    (tmp_path / "old" / "arm-s2.txt").write_text("### arm-s2\n", encoding="utf-8")
    _card(tmp_path, "new", 1, "wiped")
    _card(tmp_path, "new", 2, "won")
    assert main(["old", "new"], root=tmp_path) == EXIT_OK
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "paired 1 seed(s): old (left) vs new (right)",
        "wins   1 -> 0",
        "flips  0 to-W against 1 from-W (net -1 for new)",
        "p      1.000 two-sided binomial on 1 discordant",
        "unpaired  0 only in old, 1 only in new",
        "  s1: W -> L",
    ]
