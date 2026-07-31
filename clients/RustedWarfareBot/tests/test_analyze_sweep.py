"""The per-batch analyzer, driven against a scratch results tree.

The table it prints is what every verdict since the 24-seed standard is read
from, so its parsing -- scorecard fields by shape, extractor drops from the
trace -- is pinned here against files written the way the sweep writes them.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.analyze_sweep import EXIT_BAD_USAGE, EXIT_EMPTY, EXIT_OK, main


def _card(path: Path, name: str, values: dict[str, str]) -> None:
    """Write one scorecard the way the sweep does: 15-column labels."""
    lines = [f"### {name}"]
    lines.extend(f"{label:<15}{value}" for label, value in values.items())
    # A continuation line: lowercase and long enough, but blank at the label
    # boundary, which is the shape the parser must ignore.
    lines.append(f"{'note':<16}not a field")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trace(path: Path, extractor_counts: list[int]) -> None:
    """Write one trace: a header row, then twelve-column samples."""
    lines = ["   frame  army  credits  enemies  extractors  and  the  rest"]
    for i, count in enumerate(extractor_counts):
        lines.append(f"{i * 75} 0 4000 2 {count} 0 0 0 0 0 3500 3500")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("args", [[], ["a", "b"]])
def test_a_bad_argument_count_prints_usage(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: analyze_sweep <sweep-name>\n"


def test_an_empty_batch_reports_no_results(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "sweeps" / "trial").mkdir(parents=True)
    code = main(["trial"], sweeps=tmp_path / "sweeps", traces=tmp_path / "traces")
    assert code == EXIT_EMPTY
    assert capsys.readouterr().out == "no results yet\n"


def test_the_table_joins_scorecards_with_their_traces(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Every figure a verdict turns on, from files shaped like the real ones.

    Three matches cover the field variants one batch actually mixes: a win
    with a full card and a real trace, a loss whose card is missing the
    optional fields and whose trace never existed, and a survivor whose trace
    is present but malformed. The last two must read as zeroes, not crashes --
    a partial match is exactly when the table is wanted most.
    """
    results = tmp_path / "sweeps" / "trial"
    traces = tmp_path / "traces" / "trial"
    results.mkdir(parents=True)
    traces.mkdir(parents=True)
    _card(
        results / "alpha-s777.txt",
        "alpha-s777",
        {
            "verdict": "won (won)",
            "extractors": "0 -> 5",
            "total worth": "3500 -> 25050",
            "best rival": "3500 -> 1950 (peak 15300, worst dip 13350)",
            "intercepted": "171",
            "enemies seen": "2 -> 6 (6 engageable)",
            "income": "62/s",
        },
    )
    # Peak 3 extractors, 1 left at the end: 2 dropped.
    _trace(traces / "alpha-s777.ndjson", [0, 3, 1])
    _card(
        results / "alpha-s12345.txt",
        "alpha-s12345",
        {"verdict": "defeated (base razed)", "extractors": "0 -> 2", "total worth": "3500 -> 9000"},
    )
    _card(
        results / "beta-s777.txt",
        "beta-s777",
        {
            "verdict": "survived (sample_limit)",
            "extractors": "0 -> 4",
            "total worth": "3500 -> 20000",
            "best rival": "3500 -> 12000 (peak 12500, worst dip 800)",
            "intercepted": "12",
            "enemies seen": "1 -> 3 (2 engageable)",
            "income": "70/s",
        },
    )
    # A header-only trace with junk rows: too few columns, then a non-numeric
    # lead. Neither is a sample, so the peak stays zero.
    (traces / "beta-s777.ndjson").write_text(
        "   frame  header\n5 1 2\nx 0 0 0 9 0 0 0 0 0 0 0\n", encoding="utf-8"
    )
    code = main(["trial"], sweeps=tmp_path / "sweeps", traces=tmp_path / "traces")
    assert code == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "arm",
        "seed",
        "verdict",
        "extr",
        "peak",
        "drop",
        "worth",
        "rival",
        "dip",
        "tgts",
        "eng",
        "icpt",
        "income",
    ]
    # Rows order by arm then numeric seed: 777 before 12345.
    assert lines[1].split() == [
        "alpha",
        "777",
        "won",
        "5",
        "3",
        "2",
        "25050",
        "1950",
        "13350",
        "6",
        "6",
        "171",
        "62/s",
    ]
    assert lines[2].split() == [
        "alpha",
        "12345",
        "defeated",
        "2",
        "0",
        "0",
        "9000",
        "0",
        "0",
        "0",
        "0",
        "0",
        "?",
    ]
    assert lines[3].split() == [
        "beta",
        "777",
        "survived",
        "4",
        "0",
        "0",
        "20000",
        "12000",
        "800",
        "3",
        "2",
        "12",
        "70/s",
    ]
    assert lines[4] == ""
    assert lines[5].split() == [
        "alpha",
        "won",
        "1/2",
        "lost",
        "1",
        "drops",
        "2",
        "median",
        "worth",
        "25050",
        "unengageable",
        "0",
        "intercepts",
        "171",
    ]
    assert lines[6].split() == [
        "beta",
        "won",
        "0/1",
        "lost",
        "0",
        "drops",
        "0",
        "median",
        "worth",
        "20000",
        "unengageable",
        "1",
        "intercepts",
        "12",
    ]


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.analyze_sweep")
    sys.argv = ["analyze_sweep"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.analyze_sweep", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.analyze_sweep"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: analyze_sweep <sweep-name>\n"
