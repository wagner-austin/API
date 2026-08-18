"""Tests for the two growth-policy script entry points.

Both scripts are driven end to end against real data written to ``tmp_path``
and the real learners, at sizes small enough to run quickly. That is the point:
a script tested with its measurement layer replaced would prove only that
``argparse`` works, and the wiring between the parser and the experiment is
exactly where a path or a flag goes astray.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str
from scripts.experiment_growth_policy_multi_dataset import (
    GERMAN_RELATIVE,
    TAIWAN_RELATIVE,
)
from scripts.experiment_growth_policy_multi_dataset import build_parser as multi_parser
from scripts.experiment_growth_policy_multi_dataset import main as multi_main
from scripts.experiment_growth_policy_xgb_instrument import _write
from scripts.experiment_growth_policy_xgb_instrument import build_parser as xgb_parser
from scripts.experiment_growth_policy_xgb_instrument import main as xgb_main

from covenant_ml.growth_policy.types import (
    REPORT_SCHEMA_VERSION,
    decode_growth_policy_report,
    require_mapping,
)

from .factories import write_bankruptcy_csv, write_german_data, write_taiwan_csv

_FAST_ARGS = ["--estimators", "2", "--repeats", "1", "--warmups", "0"]


def _arm_names(output: str, dataset_prefix: str) -> list[str]:
    """Extract the arm column from a rendered table.

    Args:
        output: Captured stdout.
        dataset_prefix: The dataset line the table follows.

    Returns:
        The arm names, in table order.
    """
    lines = output.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(dataset_prefix))
    names: list[str] = []
    for line in lines[start + 3 :]:
        if len(line.strip()) == 0 or line.startswith("wrote "):
            break
        names.append(line[:24].strip())
    return names


def _bankruptcy_args(csv_path: Path) -> list[str]:
    """Build a fast argument list for the instrument script.

    Args:
        csv_path: Dataset to read.

    Returns:
        The argument list.
    """
    return [
        "--csv",
        str(csv_path),
        "--seeds",
        "42",
        "--leaf-budgets",
        "4",
        "--max-depth",
        "2",
        *_FAST_ARGS,
    ]


class TestWriteHelper:
    """The stdout helper both scripts share in shape."""

    def test_writes_without_raising(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The helper should reach stdout rather than a logger."""
        _write("hello")

        assert capsys.readouterr().out == "hello"


class TestInstrumentParser:
    """The instrument script's argument surface."""

    def test_defaults_point_at_the_library_relative_dataset(self) -> None:
        """The default CSV must resolve from this library's root."""
        parsed = xgb_parser().parse_args([])

        csv_path: Path = parsed.csv
        assert csv_path == Path("tests") / "data" / "american_bankruptcy.csv"

    def test_accepts_multiple_seeds_and_budgets(self) -> None:
        """Both list arguments should collect every value given."""
        parsed = xgb_parser().parse_args(["--seeds", "1", "2", "3", "--leaf-budgets", "7", "9"])

        seeds: list[int] = parsed.seeds
        budgets: list[int] = parsed.leaf_budgets
        assert seeds == [1, 2, 3]
        assert budgets == [7, 9]

    def test_skip_anchors_defaults_off(self) -> None:
        """The anchors are measured unless explicitly skipped."""
        skip_anchors: bool = xgb_parser().parse_args([]).skip_anchors
        assert skip_anchors is False


class TestInstrumentMain:
    """The instrument script end to end."""

    def test_runs_the_xgb_arms_only_when_anchors_are_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Skipping anchors should leave exactly the depth-wise and leaf-wise arms."""
        csv_path = tmp_path / "american_bankruptcy.csv"
        write_bankruptcy_csv(csv_path, company_count=10)

        code = xgb_main([*_bankruptcy_args(csv_path), "--skip-anchors"])

        out = capsys.readouterr().out
        assert code == 0
        assert _arm_names(out, "american-bankruptcy:") == [
            "xgb depthwise d2",
            "xgb lossguide L4",
        ]

    def test_includes_the_anchors_by_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both anchors should appear alongside the instrument's arms."""
        csv_path = tmp_path / "american_bankruptcy.csv"
        write_bankruptcy_csv(csv_path, company_count=10)

        code = xgb_main(_bankruptcy_args(csv_path))

        out = capsys.readouterr().out
        assert code == 0
        assert _arm_names(out, "american-bankruptcy:") == [
            "xgb depthwise d2",
            "xgb lossguide L4",
            "lgb leafwise L31",
            "cleargbm depthwise d2",
        ]

    def test_writes_a_decodable_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The JSON written must round-trip through the package's own decoder."""
        csv_path = tmp_path / "american_bankruptcy.csv"
        write_bankruptcy_csv(csv_path, company_count=10)
        out_path = tmp_path / "nested" / "report.json"

        xgb_main([*_bankruptcy_args(csv_path), "--skip-anchors", "--out", str(out_path)])

        payload = require_mapping(load_json_str(out_path.read_text(encoding="utf-8")), "report")
        report = decode_growth_policy_report(payload)
        assert report["schema_version"] == REPORT_SCHEMA_VERSION
        assert report["dataset"]["name"] == "american-bankruptcy"
        assert [summary["arm"] for summary in report["summaries"]] == [
            "xgb depthwise d2",
            "xgb lossguide L4",
        ]
        assert _arm_names(capsys.readouterr().out, "american-bankruptcy:") == [
            "xgb depthwise d2",
            "xgb lossguide L4",
        ]

    def test_runs_as_a_module_entry_point(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The ``__main__`` guard should exit zero."""
        csv_path = tmp_path / "american_bankruptcy.csv"
        write_bankruptcy_csv(csv_path, company_count=10)
        module = "scripts.experiment_growth_policy_xgb_instrument"
        if module in sys.modules:
            del sys.modules[module]
        original = sys.argv
        sys.argv = [
            "experiment",
            *_bankruptcy_args(csv_path),
            "--skip-anchors",
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module, run_name="__main__")
        finally:
            sys.argv = original

        assert excinfo.value.code == 0
        assert _arm_names(capsys.readouterr().out, "american-bankruptcy:") == [
            "xgb depthwise d2",
            "xgb lossguide L4",
        ]


class TestMultiDatasetParser:
    """The multi-dataset script's argument surface."""

    def test_external_root_defaults_to_a_relative_path(self) -> None:
        """The default must be repository-relative, not an absolute machine path."""
        parsed = multi_parser().parse_args([])

        external_root: Path = parsed.external_root
        assert not external_root.is_absolute()
        assert external_root.parts[0] == ".."

    def test_dataset_paths_sit_beneath_the_root(self) -> None:
        """Both datasets are addressed relative to the external root."""
        assert Path("kaggle_taiwan_bankruptcy") / "data.csv" == TAIWAN_RELATIVE
        assert Path("german_credit") / "german.data" == GERMAN_RELATIVE


class TestMultiDatasetMain:
    """The multi-dataset script end to end."""

    def test_reports_both_datasets(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each dataset should get its own table."""
        root = tmp_path / "external"
        (root / TAIWAN_RELATIVE.parent).mkdir(parents=True)
        (root / GERMAN_RELATIVE.parent).mkdir(parents=True)
        write_taiwan_csv(root / TAIWAN_RELATIVE, row_count=60, feature_count=4)
        write_german_data(root / GERMAN_RELATIVE, row_count=60)

        code = multi_main(
            [
                "--external-root",
                str(root),
                "--seeds",
                "42",
                "--leaf-budgets",
                "4",
                "--max-depth",
                "2",
                *_FAST_ARGS,
            ]
        )

        out = capsys.readouterr().out
        assert code == 0
        expected = ["xgb depthwise d2", "xgb lossguide L4"]
        assert _arm_names(out, "taiwan-bankruptcy:") == expected
        assert _arm_names(out, "german-credit:") == expected

    def test_writes_one_report_per_dataset(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each dataset's JSON should be written and decodable."""
        root = tmp_path / "external"
        (root / TAIWAN_RELATIVE.parent).mkdir(parents=True)
        (root / GERMAN_RELATIVE.parent).mkdir(parents=True)
        write_taiwan_csv(root / TAIWAN_RELATIVE, row_count=60, feature_count=4)
        write_german_data(root / GERMAN_RELATIVE, row_count=60)
        out_dir = tmp_path / "reports"

        multi_main(
            [
                "--external-root",
                str(root),
                "--seeds",
                "42",
                "--leaf-budgets",
                "4",
                "--max-depth",
                "2",
                "--out-dir",
                str(out_dir),
                *_FAST_ARGS,
            ]
        )

        for name in ("taiwan-bankruptcy", "german-credit"):
            payload = require_mapping(
                load_json_str((out_dir / f"growth-policy-{name}.json").read_text(encoding="utf-8")),
                "report",
            )
            assert decode_growth_policy_report(payload)["dataset"]["name"] == name
        capsys.readouterr()

    def test_runs_as_a_module_entry_point(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The ``__main__`` guard should exit zero."""
        root = tmp_path / "external"
        (root / TAIWAN_RELATIVE.parent).mkdir(parents=True)
        (root / GERMAN_RELATIVE.parent).mkdir(parents=True)
        write_taiwan_csv(root / TAIWAN_RELATIVE, row_count=40, feature_count=3)
        write_german_data(root / GERMAN_RELATIVE, row_count=40)
        module = "scripts.experiment_growth_policy_multi_dataset"
        if module in sys.modules:
            del sys.modules[module]
        original = sys.argv
        sys.argv = [
            "experiment",
            "--external-root",
            str(root),
            "--seeds",
            "42",
            "--leaf-budgets",
            "4",
            "--max-depth",
            "2",
            *_FAST_ARGS,
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module, run_name="__main__")
        finally:
            sys.argv = original

        assert excinfo.value.code == 0
        assert _arm_names(capsys.readouterr().out, "taiwan-bankruptcy:") == [
            "xgb depthwise d2",
            "xgb lossguide L4",
        ]
