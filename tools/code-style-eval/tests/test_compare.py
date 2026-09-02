"""Comparing two scored arms, and the report the comparison writes."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Generator, Sequence

import pytest
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from code_style_eval.cli import _test_hooks as cli_hooks
from code_style_eval.cli.compare import (
    arm_name,
    build_report,
    entrypoint,
    main,
    parse_arguments,
    read_outcomes,
    render,
)
from code_style_eval.contracts.outcomes import (
    CHECKERS,
    CheckOutcome,
    ItemOutcome,
    decode_comparison_report,
    encode_comparison_report,
    encode_item_outcome,
)


def _outcome(item_id: str, arm: str, *, passed: bool) -> ItemOutcome:
    """Build an outcome whose checkers all agree.

    Args:
        item_id: The item.
        arm: The arm.
        passed: Whether every checker passed.

    Returns:
        The outcome.
    """
    checks = tuple(
        CheckOutcome(checker=name, passed=passed, exit_code=0 if passed else 1, detail="")
        for name in CHECKERS
    )
    return ItemOutcome(item_id=item_id, arm=arm, checks=checks, all_passed=passed)


def _write(path: pathlib.Path, outcomes: Sequence[ItemOutcome]) -> pathlib.Path:
    """Write outcomes as JSONL.

    Args:
        path: File to write.
        outcomes: Outcomes to serialise.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(dump_json_str(encode_item_outcome(o)) + "\n" for o in outcomes),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Restore the CLI hook around every test."""
    cli_hooks.reset_hooks()


class TestReadingOutcomes:
    """One arm's file, keyed by item."""

    def test_rows_are_keyed_by_item(self, tmp_path: pathlib.Path) -> None:
        """The key is what makes the later comparison paired."""
        path = _write(
            tmp_path / "a.jsonl",
            [_outcome("a.py", "base", passed=True), _outcome("b.py", "base", passed=False)],
        )

        outcomes = read_outcomes(path)

        assert sorted(outcomes) == ["a.py", "b.py"]

    def test_blank_lines_are_skipped(self, tmp_path: pathlib.Path) -> None:
        """Trailing newlines are framing, not rows."""
        path = tmp_path / "a.jsonl"
        path.write_text(
            dump_json_str(encode_item_outcome(_outcome("a.py", "base", passed=True))) + "\n\n",
            encoding="utf-8",
        )

        assert len(read_outcomes(path)) == 1

    def test_a_duplicated_item_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Two rows for one item would let the last one decide, silently."""
        path = _write(
            tmp_path / "a.jsonl",
            [_outcome("a.py", "base", passed=True), _outcome("a.py", "base", passed=False)],
        )

        with pytest.raises(ValueError, match="more than once"):
            _ = read_outcomes(path)


class TestArmNaming:
    """A file records one arm."""

    def test_the_single_arm_is_read(self, tmp_path: pathlib.Path) -> None:
        """The name lands in the report."""
        path = _write(tmp_path / "a.jsonl", [_outcome("a.py", "candidate", passed=True)])

        assert arm_name(read_outcomes(path), path) == "candidate"

    def test_an_empty_file_names_no_arm(self, tmp_path: pathlib.Path) -> None:
        """An empty sweep has no arm to name, and that is not an error."""
        path = _write(tmp_path / "a.jsonl", [])

        assert arm_name(read_outcomes(path), path) == ""

    def test_a_file_mixing_arms_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Comparing a mixed file would build a table from three models."""
        path = _write(
            tmp_path / "a.jsonl",
            [_outcome("a.py", "base", passed=True), _outcome("b.py", "cand", passed=True)],
        )

        with pytest.raises(ValueError, match="mixes arms"):
            _ = arm_name(read_outcomes(path), path)


class TestTheReport:
    """Every figure comes from the shared items."""

    def test_rates_use_the_shared_denominator(self) -> None:
        """An arm scored on more items must not report a rate the table cannot explain.

        Baseline has an extra item it passed. If rates were taken over each
        file's own length the baseline would read 1.000 while the table shows
        one of its two shared items failing.
        """
        baseline = {
            "shared_pass.py": _outcome("shared_pass.py", "base", passed=True),
            "shared_fail.py": _outcome("shared_fail.py", "base", passed=False),
            "base_only.py": _outcome("base_only.py", "base", passed=True),
        }
        candidate = {
            "shared_pass.py": _outcome("shared_pass.py", "cand", passed=True),
            "shared_fail.py": _outcome("shared_fail.py", "cand", passed=True),
        }

        report = build_report(baseline, candidate, baseline_arm="base", candidate_arm="cand")

        assert report["shared_items"] == 2
        assert report["baseline_pass_rate"] == 0.5
        assert report["candidate_pass_rate"] == 1.0
        assert report["net_improvement"] == 1

    def test_both_p_values_are_carried(self) -> None:
        """Mid-p is the one to read; exact is kept beside it."""
        baseline = {f"i{n}.py": _outcome(f"i{n}.py", "base", passed=False) for n in range(6)}
        candidate = {f"i{n}.py": _outcome(f"i{n}.py", "cand", passed=True) for n in range(6)}

        report = build_report(baseline, candidate, baseline_arm="base", candidate_arm="cand")

        assert report["counts"]["candidate_only"] == 6
        assert report["mid_p"] < report["exact_p"]

    def test_the_report_round_trips(self) -> None:
        """The record survives serialization intact."""
        baseline = {"a.py": _outcome("a.py", "base", passed=False)}
        candidate = {"a.py": _outcome("a.py", "cand", passed=True)}
        report = build_report(baseline, candidate, baseline_arm="base", candidate_arm="cand")

        assert decode_comparison_report(encode_comparison_report(report)) == report

    def test_a_report_whose_table_contradicts_its_count_is_refused(self) -> None:
        """A denominator that disagrees with the table cannot be read."""
        baseline = {"a.py": _outcome("a.py", "base", passed=True)}
        candidate = {"a.py": _outcome("a.py", "cand", passed=True)}
        encoded = encode_comparison_report(
            build_report(baseline, candidate, baseline_arm="base", candidate_arm="cand")
        )
        encoded["shared_items"] = 99

        with pytest.raises(JSONTypeError, match="2x2 table sums to"):
            _ = decode_comparison_report(encoded)


class TestRendering:
    """The human-readable lines."""

    def test_every_figure_appears(self) -> None:
        """A reader must see the effect size beside the p-value."""
        baseline = {"a.py": _outcome("a.py", "base", passed=False)}
        candidate = {"a.py": _outcome("a.py", "cand", passed=True)}
        report = build_report(baseline, candidate, baseline_arm="base", candidate_arm="cand")

        lines = render(report)
        joined = "\n".join(lines)

        assert "items scored by both" in joined
        assert "net items fixed           +1" in joined
        assert "mid-p" in joined
        assert "exact conditional p" in joined


class TestParsingArguments:
    """All three flags are required."""

    def test_a_full_command_line_parses(self, tmp_path: pathlib.Path) -> None:
        """Paths come back in order."""
        parsed = parse_arguments(
            ["--baseline", "b.jsonl", "--candidate", "c.jsonl", "--out", "r.json"]
        )

        assert parsed == (
            pathlib.Path("b.jsonl"),
            pathlib.Path("c.jsonl"),
            pathlib.Path("r.json"),
        )

    @pytest.mark.parametrize("missing", ["--baseline", "--candidate", "--out"])
    def test_each_flag_is_required(self, missing: str) -> None:
        """Parametrised so a fourth flag cannot be added untested.

        Args:
            missing: The flag to drop.
        """
        tokens = ["--baseline", "b", "--candidate", "c", "--out", "r"]
        index = tokens.index(missing)
        del tokens[index : index + 2]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            _ = parse_arguments(tokens)

    def test_an_unknown_flag_is_refused(self) -> None:
        """A typo must not be ignored."""
        with pytest.raises(ValueError, match="unknown argument"):
            _ = parse_arguments(["--nope", "x"])

    def test_a_flag_without_a_value_is_refused(self) -> None:
        """A trailing flag would read past the end."""
        with pytest.raises(ValueError, match="requires a value"):
            _ = parse_arguments(["--baseline"])


class TestTheComparison:
    """End to end over real files."""

    def test_a_comparison_writes_a_decodable_report(self, tmp_path: pathlib.Path) -> None:
        """The written JSON is the same record the codec produced."""
        base = _write(
            tmp_path / "base.jsonl",
            [_outcome("a.py", "base", passed=False), _outcome("b.py", "base", passed=True)],
        )
        cand = _write(
            tmp_path / "cand.jsonl",
            [_outcome("a.py", "cand", passed=True), _outcome("b.py", "cand", passed=True)],
        )
        out = tmp_path / "nested" / "report.json"
        emitted: list[str] = []
        cli_hooks.emit = emitted.append

        code = main(["--baseline", str(base), "--candidate", str(cand), "--out", str(out)])

        assert code == 0
        report = decode_comparison_report(
            narrow_json_to_dict(load_json_str(out.read_text(encoding="utf-8")))
        )
        assert report["baseline_arm"] == "base"
        assert report["candidate_arm"] == "cand"
        assert report["net_improvement"] == 1
        assert any("net items fixed" in line for line in emitted)


def _make_argv() -> Generator[list[str], None, None]:
    """Give a test control of ``sys.argv`` and restore it afterwards.

    Yields:
        The live argument list, for the test to replace in place.
    """
    original = list(sys.argv)
    yield sys.argv
    sys.argv[:] = original


# The call form resolves pytest's overloaded decorator to a concrete type.
argv = pytest.fixture(_make_argv)


class TestTheEntryPoint:
    """The console script reads the process arguments."""

    def test_the_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, argv: list[str]
    ) -> None:
        """Run for real rather than excluded from coverage.

        Args:
            tmp_path: Directory for the files.
            argv: The live process arguments, replaced in place.
        """
        base = _write(tmp_path / "base.jsonl", [_outcome("a.py", "base", passed=True)])
        cand = _write(tmp_path / "cand.jsonl", [_outcome("a.py", "cand", passed=True)])
        out = tmp_path / "report.json"
        argv[:] = [
            "prog",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--out",
            str(out),
        ]

        with pytest.raises(SystemExit) as raised:
            entrypoint()

        assert raised.value.code == 0
        assert out.is_file()
