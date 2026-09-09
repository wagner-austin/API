"""The grouping this corpus claims, and the series it is measured on.

The arithmetic is :mod:`platform_core.clustering`'s and is tested there. What
is tested here is the half that is a claim about THIS corpus: which files
count as one unit, and that the series handed to the instrument is the paired
difference rather than either arm's raw outcome.

The distinction is not cosmetic. On the committed corpus the two series
disagree by up to five times and, at the aggregate, in sign, so a version of
:func:`paired_differences` that returned the candidate's pass indicator would
produce a plausible table with the wrong numbers in it and no test would
notice unless one asserts the shape of the series directly.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError

from code_style_eval.cli import _test_hooks as cli_hooks
from code_style_eval.cli.clustering import (
    build_records,
    main,
    parse_arguments,
    read_finished,
    read_outcomes,
    render,
)
from code_style_eval.contracts.generation import (
    decode_generation_outcome,
    encode_generation_outcome,
)
from code_style_eval.contracts.outcomes import CHECKERS, CheckOutcome, ItemOutcome
from code_style_eval.core.clustering import (
    ClusteringUnit,
    both_finished,
    cluster_key,
    grouped_differences,
    paired_differences,
)


def _outcome(item_id: str, *, ruff: bool, mypy: bool, guards: bool) -> ItemOutcome:
    """Build one item's row.

    Args:
        item_id: Repository-relative path.
        ruff: Whether ruff passed.
        mypy: Whether mypy passed.
        guards: Whether the guards passed.

    Returns:
        The row, with ``all_passed`` consistent with the three checks.
    """
    checks = tuple(
        CheckOutcome(checker=name, passed=passed, exit_code=0 if passed else 1, detail="")
        for name, passed in zip(CHECKERS, (ruff, mypy, guards), strict=True)
    )
    return ItemOutcome(
        item_id=item_id, arm="test", checks=checks, all_passed=ruff and mypy and guards
    )


def _pair(
    baseline: dict[str, bool], candidate: dict[str, bool]
) -> tuple[dict[str, ItemOutcome], dict[str, ItemOutcome]]:
    """Build two arms from per-item guard results.

    Args:
        baseline: Guard result per item for the baseline arm.
        candidate: Guard result per item for the candidate arm.

    Returns:
        The two arms, keyed by item id. ``ruff`` and ``mypy`` are forced TRUE
        so that ``all_passed`` tracks ``guards`` exactly -- a fixture where
        the combined outcome and the single checker can differ would let a
        test pass while reading the wrong one.
    """
    return (
        {item: _outcome(item, ruff=True, mypy=True, guards=ok) for item, ok in baseline.items()},
        {item: _outcome(item, ruff=True, mypy=True, guards=ok) for item, ok in candidate.items()},
    )


def _row(item_id: str, arm: str, *, passed: bool) -> str:
    """Serialise one outcome row as the CLI will read it.

    Args:
        item_id: Repository-relative path.
        arm: Arm name.
        passed: Whether every checker passed.

    Returns:
        One JSON line, newline-terminated. The three check rows are written
        out rather than left empty because the contract refuses an
        ``all_passed`` that disagrees with its checks, and ``all([])`` is
        True -- an empty list can only ever encode a pass.
    """
    flag = "true" if passed else "false"
    checks = ",".join(
        f'{{"checker":"{name}","passed":{flag},"exit_code":{0 if passed else 1},"detail":""}}'
        for name in CHECKERS
    )
    return f'{{"item_id":"{item_id}","arm":"{arm}","checks":[{checks}],"all_passed":{flag}}}\n'


class TestTheClusterProjection:
    """Which files are one unit, and the paths that do not fit the pattern."""

    @pytest.mark.parametrize(
        ("unit", "expected"),
        [
            (ClusteringUnit.TOP_LEVEL_CATEGORY, "libs"),
            (ClusteringUnit.PACKAGE, "libs/platform_core"),
            (ClusteringUnit.CONTAINING_DIRECTORY, "libs/platform_core/src/platform_core"),
        ],
    )
    def test_each_unit_takes_its_own_prefix(self, unit: ClusteringUnit, expected: str) -> None:
        """One path, three groupings, three answers.

        Args:
            unit: The grouping under test.
            expected: The cluster key it must produce.
        """
        assert cluster_key("libs/platform_core/src/platform_core/errors.py", unit) == expected

    def test_a_short_path_becomes_its_own_package(self) -> None:
        """A top-level file belongs to no package.

        Returning the whole path makes it a singleton cluster. Truncating to
        the first segment instead would put unrelated root files in one
        cluster and assert a correlation nothing supports.
        """
        assert cluster_key("setup.py", ClusteringUnit.PACKAGE) == "setup.py"

    def test_a_root_file_clusters_at_the_repository_root(self) -> None:
        """The empty string is a real cluster under the directory unit.

        Root files genuinely share a directory, so they genuinely share a
        cluster, and the key for it is the empty string rather than a
        missing value.
        """
        assert cluster_key("setup.py", ClusteringUnit.CONTAINING_DIRECTORY) == ""


class TestTheDifferenceSeries:
    """The series is candidate MINUS baseline, and only over shared items."""

    def test_it_is_the_paired_difference_and_not_either_arm(self) -> None:
        """THE DEFECT THIS FILE EXISTS FOR, asserted on values not shape.

        A fixed item is +1, a broken item is -1, agreement is 0 whichever
        way both arms agreed. A version returning the candidate's raw pass
        indicator would give ``{fixed: 1, broken: 0, both_pass: 1,
        both_fail: 0}`` and every downstream table would still render.
        """
        baseline, candidate = _pair(
            {"a/fixed.py": False, "a/broken.py": True, "a/both.py": True, "a/neither.py": False},
            {"a/fixed.py": True, "a/broken.py": False, "a/both.py": True, "a/neither.py": False},
        )

        assert paired_differences(baseline, candidate, "guards") == {
            "a/both.py": 0,
            "a/broken.py": -1,
            "a/fixed.py": 1,
            "a/neither.py": 0,
        }

    def test_only_shared_items_appear(self) -> None:
        """An item one arm never produced is not evidence about the other."""
        baseline, candidate = _pair({"a/x.py": True, "a/only_base.py": True}, {"a/x.py": False})

        assert set(paired_differences(baseline, candidate, "guards")) == {"a/x.py"}

    def test_the_combined_outcome_is_selectable(self) -> None:
        """``all`` reads ``all_passed`` rather than a single checker."""
        baseline = {"a/x.py": _outcome("a/x.py", ruff=True, mypy=False, guards=True)}
        candidate = {"a/x.py": _outcome("a/x.py", ruff=True, mypy=True, guards=True)}

        assert paired_differences(baseline, candidate, "all") == {"a/x.py": 1}
        assert paired_differences(baseline, candidate, "guards") == {"a/x.py": 0}

    def test_an_unknown_checker_is_refused(self) -> None:
        """Refused rather than treated as a miss on every item."""
        baseline, candidate = _pair({"a/x.py": True}, {"a/x.py": True})

        with pytest.raises(ValueError, match="must be 'all' or one of"):
            _ = paired_differences(baseline, candidate, "pyright")

    def test_a_row_missing_its_checker_is_refused(self) -> None:
        """Scoring an absent row as a failure would move discordant cells."""
        baseline = {"a/x.py": ItemOutcome(item_id="a/x.py", arm="b", checks=(), all_passed=True)}
        candidate = {"a/x.py": _outcome("a/x.py", ruff=True, mypy=True, guards=True)}

        with pytest.raises(ValueError, match="has no 'guards' row"):
            _ = paired_differences(baseline, candidate, "guards")


class TestTheStratum:
    """Both-finished, which is both arms and not either."""

    def test_it_keeps_only_items_both_arms_finished(self) -> None:
        """An item one arm truncated is not in the stratum."""
        scored = {"a.py": 1, "b.py": 0, "c.py": -1}

        assert both_finished(("a.py", "b.py"), ("a.py", "c.py"), scored) == {"a.py": 1}

    def test_an_id_neither_arm_scored_cannot_join_its_way_in(self) -> None:
        """Finishing is necessary and not sufficient: the item must be scored."""
        assert both_finished(("ghost.py",), ("ghost.py",), {"a.py": 1}) == {}


class TestTheGrouping:
    """What reaches the instrument."""

    def test_items_in_one_directory_land_in_one_cluster(self) -> None:
        """The mapping the instrument takes, keyed and valued as it expects."""
        differences = {"libs/p/a.py": 1, "libs/p/b.py": -1, "libs/q/c.py": 0}

        assert grouped_differences(differences, ClusteringUnit.PACKAGE) == {
            "libs/p": [1.0, -1.0],
            "libs/q": [0.0],
        }

    def test_every_unit_is_reported_coarsest_first(self) -> None:
        """The reader meets the largest design effect before the kind one."""
        differences = {f"top{index % 3}/pkg{index % 5}/dir/f{index}.py": 0 for index in range(30)}
        differences["top0/pkg0/dir/f0.py"] = 1
        differences["top1/pkg1/dir/f1.py"] = -1

        records = build_records(differences)

        assert tuple(record["unit"] for record in records) == tuple(
            unit.value for unit in ClusteringUnit
        )

    def test_a_corpus_too_thin_to_group_is_refused_not_skipped(self) -> None:
        """A table silently missing a row reads as though that row were fine."""
        with pytest.raises(AppError) as excinfo:
            _ = build_records({"onlytop/pkg/f.py": 1, "onlytop/pkg/g.py": 0})

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_CLUSTERS_INSUFFICIENT


class TestTheGenerationContract:
    """The manifest that defines the stratum, decoded rather than assumed."""

    def test_a_record_survives_a_round_trip(self) -> None:
        """Both fields, both values."""
        record = decode_generation_outcome(
            encode_generation_outcome({"item_id": "a/x.py", "finished": False})
        )

        assert record == {"item_id": "a/x.py", "finished": False}

    def test_a_missing_finished_field_is_refused(self) -> None:
        """It would have read as falsy and moved the item out of the stratum."""
        with pytest.raises(JSONTypeError, match="finished"):
            _ = decode_generation_outcome({"item_id": "a/x.py"})

    def test_an_empty_item_id_is_refused(self) -> None:
        """An id that joins to nothing changes a denominator without failing."""
        with pytest.raises(JSONTypeError, match="joins to nothing"):
            _ = decode_generation_outcome({"item_id": "", "finished": True})


class TestTheCommandLine:
    """Flags, and the one combination that would misreport a stratum."""

    def test_the_three_required_flags_are_parsed(self) -> None:
        """No generation manifests means no stratum restriction."""
        parsed = parse_arguments(
            ["--baseline", "b.jsonl", "--candidate", "c.jsonl", "--checker", "guards"]
        )

        assert parsed == (pathlib.Path("b.jsonl"), pathlib.Path("c.jsonl"), "guards", None)

    def test_both_generation_manifests_are_parsed_together(self) -> None:
        """The stratum form."""
        parsed = parse_arguments(
            [
                "--baseline",
                "b.jsonl",
                "--candidate",
                "c.jsonl",
                "--checker",
                "all",
                "--baseline-generation",
                "bg.jsonl",
                "--candidate-generation",
                "cg.jsonl",
            ]
        )

        assert parsed[3] == (pathlib.Path("bg.jsonl"), pathlib.Path("cg.jsonl"))

    def test_one_generation_manifest_alone_is_refused(self) -> None:
        """THE SUBTLE ONE. Restricting by one arm is a different denominator.

        Honouring it would report a stratum defined by the baseline's
        truncations under the name of the both-finished stratum, and nothing
        downstream could tell the two apart.
        """
        with pytest.raises(ValueError, match="without its pair"):
            _ = parse_arguments(
                [
                    "--baseline",
                    "b.jsonl",
                    "--candidate",
                    "c.jsonl",
                    "--checker",
                    "guards",
                    "--baseline-generation",
                    "bg.jsonl",
                ]
            )

    def test_an_unknown_checker_is_refused_before_any_file_is_read(self) -> None:
        """Parsing catches it, so a typo does not read two large files first."""
        with pytest.raises(ValueError, match="must be 'all' or one of"):
            _ = parse_arguments(
                ["--baseline", "b.jsonl", "--candidate", "c.jsonl", "--checker", "ruff2"]
            )

    def test_an_unknown_flag_is_refused(self) -> None:
        """Silently ignoring it would run a different command than was typed."""
        with pytest.raises(ValueError, match="unknown argument"):
            _ = parse_arguments(
                [
                    "--baseline",
                    "b.jsonl",
                    "--candidate",
                    "c.jsonl",
                    "--checker",
                    "all",
                    "--unit",
                    "x",
                ]
            )

    def test_a_flag_without_a_value_is_refused(self) -> None:
        """The next flag must not be swallowed as this one's value."""
        with pytest.raises(ValueError, match="requires a value"):
            _ = parse_arguments(["--baseline", "b.jsonl", "--candidate", "c.jsonl", "--checker"])

    def test_a_missing_required_flag_is_refused(self) -> None:
        """Every one of the three, not merely the first."""
        with pytest.raises(ValueError, match="--checker is required"):
            _ = parse_arguments(["--baseline", "b.jsonl", "--candidate", "c.jsonl"])


class TestReading:
    """The two file readers."""

    def test_blank_lines_are_skipped(self, tmp_path: pathlib.Path) -> None:
        """A trailing newline is not a row."""
        path = tmp_path / "o.jsonl"
        path.write_text(
            '{"item_id":"a/x.py","arm":"b","checks":[],"all_passed":true}\n\n', encoding="utf-8"
        )

        assert set(read_outcomes(path)) == {"a/x.py"}

    def test_a_duplicated_item_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Whichever row came last would decide the arm's verdict, silently."""
        row = '{"item_id":"a/x.py","arm":"b","checks":[],"all_passed":true}\n'
        path = tmp_path / "o.jsonl"
        path.write_text(row * 2, encoding="utf-8")

        with pytest.raises(ValueError, match="more than once"):
            _ = read_outcomes(path)

    def test_unfinished_rows_are_dropped_from_the_finished_set(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The reader returns the SET, not the manifest."""
        path = tmp_path / "g.jsonl"
        path.write_text(
            '{"item_id":"a/x.py","finished":true}\n{"item_id":"a/y.py","finished":false}\n\n',
            encoding="utf-8",
        )

        assert read_finished(path) == ("a/x.py",)


class TestTheReport:
    """What a reader is shown, including the line about the floor."""

    def test_the_counts_above_the_table_come_from_the_series(self) -> None:
        """Discordant split and net, so the table is read beside its data."""
        differences = {"a/p/x.py": 1, "a/p/y.py": 1, "b/q/z.py": -1, "b/q/w.py": 0}

        lines = render(build_records(differences), "guards", differences)

        assert any("items                    4" in line for line in lines)
        assert any("baseline-only 1, candidate-only 2" in line for line in lines)
        assert any("net                      +1" in line for line in lines)

    def test_the_floor_is_explained_where_it_is_visible(self) -> None:
        """A DE of exactly 1.000 must not read as a coincidence."""
        differences = {"a/p/x.py": 1, "a/p/y.py": -1, "b/q/z.py": 1, "b/q/w.py": -1}

        lines = render(build_records(differences), "all", differences)

        assert any("cannot raise a sample size" in line for line in lines)

    def test_a_full_run_reports_every_unit(self, tmp_path: pathlib.Path) -> None:
        """End to end through main, files on disk, nothing rebound but emit."""
        baseline = tmp_path / "b.jsonl"
        candidate = tmp_path / "c.jsonl"
        items = [f"top{index % 3}/pkg{index % 4}/dir{index % 6}/f{index}.py" for index in range(24)]
        baseline.write_text(
            "".join(_row(item, "base", passed=bool(index % 2)) for index, item in enumerate(items)),
            encoding="utf-8",
        )
        candidate.write_text(
            "".join(_row(item, "cand", passed=bool(index % 3)) for index, item in enumerate(items)),
            encoding="utf-8",
        )
        emitted: list[str] = []
        cli_hooks.emit = emitted.append

        code = main(
            ["--baseline", str(baseline), "--candidate", str(candidate), "--checker", "all"]
        )

        assert code == 0
        for unit in ClusteringUnit:
            assert any(line.startswith(unit.value) for line in emitted)

    def test_a_stratum_run_reports_the_restricted_denominator(self, tmp_path: pathlib.Path) -> None:
        """The both-finished path, and the label that says so."""
        items = [f"top{index % 3}/pkg{index % 4}/dir{index % 6}/f{index}.py" for index in range(24)]
        baseline = tmp_path / "b.jsonl"
        candidate = tmp_path / "c.jsonl"
        baseline.write_text(
            "".join(_row(item, "base", passed=False) for item in items), encoding="utf-8"
        )
        candidate.write_text(
            "".join(_row(item, "cand", passed=bool(index % 2)) for index, item in enumerate(items)),
            encoding="utf-8",
        )
        base_gen = tmp_path / "bg.jsonl"
        cand_gen = tmp_path / "cg.jsonl"
        base_gen.write_text(
            "".join(f'{{"item_id":"{item}","finished":true}}\n' for item in items), encoding="utf-8"
        )
        cand_gen.write_text(
            "".join(
                f'{{"item_id":"{item}","finished":{"true" if index < 18 else "false"}}}\n'
                for index, item in enumerate(items)
            ),
            encoding="utf-8",
        )
        emitted: list[str] = []
        cli_hooks.emit = emitted.append

        code = main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--checker",
                "all",
                "--baseline-generation",
                str(base_gen),
                "--candidate-generation",
                str(cand_gen),
            ]
        )

        assert code == 0
        assert any("all (both-finished)" in line for line in emitted)
        assert any("items                    18" in line for line in emitted)
