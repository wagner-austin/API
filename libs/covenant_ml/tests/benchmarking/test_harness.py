"""Tests for the benchmarking harness.

The harness exists so that a benchmark cannot write a manifest without also
writing the record that makes its numbers re-derivable -- the defect board
task ``6d5536cc`` found in five of six entry points. So the load-bearing test
here is not that the writer works; it is that BOTH FILES APPEAR FROM ONE CALL,
and that the record carries the declaration's own experiment name rather than
a shared one.

Manifests are built as plain encoded JSON rather than through a family's
encoder. That is the boundary the harness actually reads at, and using a real
encoder would tie these tests to one family's shape while claiming to test the
path all six share.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.comparability import NO_VALUE
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import determinism_record
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)
from platform_core.run_record import Observation
from platform_core.testing import sample_run_fingerprint

from covenant_ml.benchmarking.harness import (
    VALUE_TYPES,
    ArgumentKind,
    BenchmarkArgument,
    BenchmarkDeclaration,
    arm_name,
    declared_observations,
    shared_parser,
    summary_lines,
    write_manifest_and_record,
)

_FINGERPRINT = sample_run_fingerprint(
    image_digest="sha256:" + "ab" * 32,
    gpu_model=NO_VALUE,
    driver_version=NO_VALUE,
    determinism=determinism_record("cpu", {"OMP_NUM_THREADS": SINGLE_THREAD}),
)

#: A two-axis family, because the one-axis case cannot show an arm collision.
_TWO_AXIS = BenchmarkDeclaration(
    experiment="test-two-axis",
    description="A family measured on two axes.",
    arguments=(
        BenchmarkArgument(flag="--rows", kind=ArgumentKind.INT, help="Rows."),
        BenchmarkArgument(flag="--rate", kind=ArgumentKind.FLOAT, help="Rate."),
        BenchmarkArgument(flag="--name", kind=ArgumentKind.STRING, help="Name."),
        BenchmarkArgument(flag="--dir", kind=ArgumentKind.PATH, help="Directory."),
        BenchmarkArgument(flag="--extras", kind=ArgumentKind.FLAG, help="Extras."),
    ),
    arm_axes=("model", "sampling"),
    quality_metrics=("auc",),
    result_metrics=("fit_seconds",),
)


def _manifest(results: list[JSONObject]) -> JSONValue:
    """Build an encoded manifest carrying the given results.

    Args:
        results: The per-arm-per-seed records.

    Returns:
        An encoded manifest.
    """
    return {"seeds": [42, 43], "results": list(results)}


def _result(model: str, sampling: str, auc: float, fit_seconds: float) -> JSONObject:
    """Build one encoded per-arm-per-seed record.

    Args:
        model: Arm's model name.
        sampling: Arm's sampling mode.
        auc: Held-out AUC.
        fit_seconds: Wall time for the fit.

    Returns:
        An encoded result.
    """
    return {
        "model": model,
        "sampling": sampling,
        "quality": {"auc": auc},
        "fit_seconds": fit_seconds,
    }


class TestSharedParser:
    """Every family's flags come from its declaration."""

    def test_seeds_and_out_are_required_of_every_family(self) -> None:
        parser = shared_parser(_TWO_AXIS)
        with pytest.raises(SystemExit):
            parser.parse_args(["--rows", "1"])

    def test_each_declared_kind_reaches_argparse_as_its_type(self) -> None:
        parsed = shared_parser(_TWO_AXIS).parse_args(
            [
                "--seeds",
                "42",
                "43",
                "--out",
                "m.json",
                "--rows",
                "7",
                "--rate",
                "0.25",
                "--name",
                "alpha",
                "--dir",
                "corpora",
            ]
        )
        # Bound to annotated names first: argparse hands back Any, and the
        # annotation is what makes the assertion a typed one.
        seeds: list[int] = parsed.seeds
        out: Path = parsed.out
        rows: int = parsed.rows
        rate: float = parsed.rate
        name: str = parsed.name
        directory: Path = parsed.dir
        assert seeds == [42, 43]
        assert out == Path("m.json")
        assert rows == 7
        assert rate == 0.25
        assert name == "alpha"
        assert directory == Path("corpora")

    def test_a_flag_is_absent_by_default_and_true_when_given(self) -> None:
        # A flag carries its meaning by being present, so it cannot be
        # "required" the way a valued argument is -- absent IS the false case.
        base = [
            "--seeds",
            "42",
            "--out",
            "m.json",
            "--rows",
            "1",
            "--rate",
            "0.5",
            "--name",
            "a",
            "--dir",
            "d",
        ]
        absent: bool = shared_parser(_TWO_AXIS).parse_args(base).extras
        present: bool = shared_parser(_TWO_AXIS).parse_args([*base, "--extras"]).extras
        assert absent is False
        assert present is True


class TestValueTypes:
    """Every valued kind has a parser, and the flag kind deliberately has none."""

    def test_every_kind_but_the_flag_has_a_parser(self) -> None:
        # A kind added without a parser raises KeyError at parse time rather
        # than silently becoming one of the others.
        valued = {kind for kind in ArgumentKind if kind is not ArgumentKind.FLAG}
        assert set(VALUE_TYPES) == valued

    def test_the_flag_kind_has_no_parser_because_it_takes_no_value(self) -> None:
        assert ArgumentKind.FLAG not in VALUE_TYPES


class TestArmName:
    """An arm is every axis that varies, not just the model."""

    def test_two_axes_join_into_one_name(self) -> None:
        assert arm_name({"model": "cleargbm", "sampling": "goss"}, ("model", "sampling")) == (
            "cleargbm@goss"
        )

    def test_no_axes_is_the_empty_name(self) -> None:
        # vs_lightgbm declares none: its observations are cross-arm ratios,
        # so there is no arm to scope them to.
        assert arm_name({"model": "cleargbm"}, ()) == ""

    def test_a_missing_axis_is_refused_rather_than_skipped(self) -> None:
        with pytest.raises(JSONTypeError):
            arm_name({"model": "cleargbm"}, ("model", "sampling"))


class TestDeclaredObservations:
    """One builder for every family; what differs is the declaration."""

    def test_the_sampling_axis_keeps_a_models_two_arms_apart(self) -> None:
        # THE COLLISION THE ARM AXES EXIST FOR. Named by model alone, a
        # model's GOSS number and its own full-sampling number would land on
        # one observation and average together.
        observations = declared_observations(
            _TWO_AXIS,
            [
                _result("cleargbm", "full", 0.80, 1.0),
                _result("cleargbm", "goss", 0.90, 2.0),
            ],
        )
        by_name = {o["name"]: o["value"] for o in observations}
        assert by_name["mean_cleargbm@full.auc"] == 0.80
        assert by_name["mean_cleargbm@goss.auc"] == 0.90

    def test_a_metric_is_averaged_across_the_seeds_that_reported_it(self) -> None:
        observations = declared_observations(
            _TWO_AXIS,
            [
                _result("cleargbm", "full", 0.80, 1.0),
                _result("cleargbm", "full", 0.90, 3.0),
            ],
        )
        by_name = {o["name"]: o["value"] for o in observations}
        assert by_name["mean_cleargbm@full.auc"] == pytest.approx(0.85)
        assert by_name["mean_cleargbm@full.fit_seconds"] == pytest.approx(2.0)

    def test_a_declaration_with_no_metrics_observes_nothing(self) -> None:
        empty = BenchmarkDeclaration(
            experiment="test-empty",
            description="Declares no per-arm metric.",
            arguments=(),
            arm_axes=(),
            quality_metrics=(),
            result_metrics=(),
        )
        assert declared_observations(empty, [_result("cleargbm", "full", 0.8, 1.0)]) == ()

    def test_a_declared_metric_the_manifest_lacks_is_refused(self) -> None:
        # The declaration claiming a measurement the run did not carry.
        lying = BenchmarkDeclaration(
            experiment="test-lying",
            description="Declares a metric nothing reports.",
            arguments=(),
            arm_axes=("model",),
            quality_metrics=("ndcg",),
            result_metrics=(),
        )
        with pytest.raises(JSONTypeError):
            declared_observations(lying, [_result("cleargbm", "full", 0.8, 1.0)])


class TestSummaryLines:
    """The report a human reads, rendered from the record's own observations."""

    def test_every_observation_gets_a_line(self) -> None:
        lines = summary_lines(
            _TWO_AXIS,
            _manifest(
                [
                    _result("cleargbm", "full", 0.80, 1.0),
                    _result("cleargbm", "goss", 0.90, 2.0),
                ]
            ),
        )
        # Two arms x (one quality metric + one result metric).
        assert len(lines) == 4
        assert all(line.startswith("  mean_cleargbm@") for line in lines)

    def test_the_report_and_the_record_cannot_disagree(self, tmp_path: Path) -> None:
        # Both are built from `declared_observations`, so a number in the
        # report that is not in the record is not expressible.
        encoded = _manifest([_result("cleargbm", "full", 0.75, 4.0)])
        record_path = write_manifest_and_record(
            _TWO_AXIS, encoded, _FINGERPRINT, tmp_path / "m.json", [42]
        )
        record = narrow_json_to_dict(load_json_str(record_path.read_text(encoding="utf-8")))
        recorded = {
            require_str(narrow_json_to_dict(entry), "name")
            for entry in require_list(record, "observations")
        }
        rendered = {line.split()[0] for line in summary_lines(_TWO_AXIS, encoded)}
        assert rendered == recorded

    def test_a_family_that_observes_nothing_renders_nothing(self) -> None:
        # vs_lightgbm's shape: no per-arm metric declared. Rendering a header
        # over an empty table would be output that says a measurement happened.
        empty = BenchmarkDeclaration(
            experiment="test-empty",
            description="Declares no per-arm metric.",
            arguments=(),
            arm_axes=(),
            quality_metrics=(),
            result_metrics=(),
        )
        assert summary_lines(empty, _manifest([_result("cleargbm", "full", 0.8, 1.0)])) == ()

    def test_derived_observations_are_rendered_beside_the_declared_ones(self) -> None:
        lines = summary_lines(
            _TWO_AXIS,
            _manifest([_result("cleargbm", "full", 0.8, 1.0)]),
            derived=(Observation(name="raw_ratio", value=1.25),),
        )
        assert any(line.split()[0] == "raw_ratio" for line in lines)

    def test_names_are_column_aligned_on_the_longest(self) -> None:
        lines = summary_lines(_TWO_AXIS, _manifest([_result("cleargbm", "full", 0.8, 1.0)]))
        # Every line's value starts at the same column, which is what makes a
        # column of numbers readable at a glance.
        assert len({len(line) - len(line.split()[-1]) for line in lines}) == 1


class TestWriteManifestAndRecord:
    """The point of the module: neither file can be written without the other."""

    def test_one_call_writes_both_files(self, tmp_path: Path) -> None:
        out_path = tmp_path / "m.json"
        record_path = write_manifest_and_record(
            _TWO_AXIS,
            _manifest([_result("cleargbm", "full", 0.8, 1.0)]),
            _FINGERPRINT,
            out_path,
            [42, 43],
        )
        assert out_path.exists()
        assert record_path.exists()

    def test_the_record_carries_its_own_experiment_not_a_shared_one(self, tmp_path: Path) -> None:
        # comparability decides subtractability by experiment name. Before
        # this module there was ONE constant, and routing every family through
        # it would have let a ranking NDCG be subtracted from a regression R^2.
        record_path = write_manifest_and_record(
            _TWO_AXIS,
            _manifest([_result("cleargbm", "full", 0.8, 1.0)]),
            _FINGERPRINT,
            tmp_path / "m.json",
            [42, 43],
        )
        record = narrow_json_to_dict(load_json_str(record_path.read_text(encoding="utf-8")))
        assert require_str(record, "experiment") == "test-two-axis"
        assert require_str(record, "label") == "2seeds"

    def test_a_label_prefix_distinguishes_runs_the_seed_count_cannot(self, tmp_path: Path) -> None:
        record_path = write_manifest_and_record(
            _TWO_AXIS,
            _manifest([_result("cleargbm", "full", 0.8, 1.0)]),
            _FINGERPRINT,
            tmp_path / "m.json",
            [42],
            label_prefix="deadbeef-median-",
        )
        record = narrow_json_to_dict(load_json_str(record_path.read_text(encoding="utf-8")))
        assert require_str(record, "label") == "deadbeef-median-1seeds"

    def test_derived_observations_are_appended_to_the_declared_ones(self, tmp_path: Path) -> None:
        record_path = write_manifest_and_record(
            _TWO_AXIS,
            _manifest([_result("cleargbm", "full", 0.8, 1.0)]),
            _FINGERPRINT,
            tmp_path / "m.json",
            [42],
            derived=(Observation(name="raw_ratio", value=1.25),),
        )
        record = narrow_json_to_dict(load_json_str(record_path.read_text(encoding="utf-8")))
        names = [
            require_str(narrow_json_to_dict(entry), "name")
            for entry in require_list(record, "observations")
        ]
        assert "mean_cleargbm@full.auc" in names
        assert "raw_ratio" in names

    def test_the_payload_digest_changes_when_a_seeds_result_changes(self, tmp_path: Path) -> None:
        # The digest covers the per-seed detail the observations summarise, so
        # two runs with equal means and different seeds are still tellable
        # apart.
        first = write_manifest_and_record(
            _TWO_AXIS,
            _manifest([_result("cleargbm", "full", 0.8, 1.0)]),
            _FINGERPRINT,
            tmp_path / "a.json",
            [42],
        )
        second = write_manifest_and_record(
            _TWO_AXIS,
            _manifest([_result("cleargbm", "full", 0.8, 2.0)]),
            _FINGERPRINT,
            tmp_path / "b.json",
            [42],
        )
        left = narrow_json_to_dict(load_json_str(first.read_text(encoding="utf-8")))
        right = narrow_json_to_dict(load_json_str(second.read_text(encoding="utf-8")))
        assert require_str(left, "payload_digest") != require_str(right, "payload_digest")

    def test_a_manifest_without_results_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(JSONTypeError):
            write_manifest_and_record(
                _TWO_AXIS, {"seeds": [42]}, _FINGERPRINT, tmp_path / "m.json", [42]
            )

    def test_two_arms_sharing_an_observation_name_are_refused(self, tmp_path: Path) -> None:
        # A collision means a reader cannot tell two arms apart, so the record
        # is refused rather than written with one silently winning.
        with pytest.raises(ValueError):
            write_manifest_and_record(
                _TWO_AXIS,
                _manifest([_result("cleargbm", "full", 0.8, 1.0)]),
                _FINGERPRINT,
                tmp_path / "m.json",
                [42],
                derived=(Observation(name="mean_cleargbm@full.auc", value=0.1),),
            )
