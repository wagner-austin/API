"""Tests for the regression benchmark command-line entry point.

Runs the real script with all three real learners on a small grouped
corpus, so the wiring between parser, runner, and manifest output is
exercised end to end.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)
from scripts.benchmark_cleargbm_regression import build_parser, main

from .test_regression_quality import write_rw_value_fixture


def _small_args(external_dir: Path, out: Path | None = None) -> list[str]:
    """Return CLI arguments for a fast run.

    Args:
        external_dir: Root holding the fixture corpus.
        out: Optional manifest output path.

    Returns:
        Argument list.
    """
    args = [
        "--dataset",
        "rw_value",
        "--external-dir",
        str(external_dir),
        "--trees",
        "20",
        "--max-depth",
        "3",
        "--num-leaves",
        "7",
        "--max-bins",
        "16",
        "--min-samples-leaf",
        "5",
        "--early-stopping",
        "10",
        "--seeds",
        "42",
    ]
    if out is not None:
        args.extend(["--out", str(out)])
    return args


class TestParser:
    """Defaults match the documented P1 protocol."""

    def test_defaults(self) -> None:
        """The default protocol is the P1 regression protocol."""
        parsed = build_parser().parse_args(["--dataset", "rw_value", "--external-dir", "x"])
        trees: int = parsed.trees
        learning_rate: float = parsed.learning_rate
        num_leaves: int = parsed.num_leaves
        seeds: list[int] = parsed.seeds
        out: Path | None = parsed.out
        assert trees == 300
        assert learning_rate == 0.05
        assert num_leaves == 31
        assert seeds == [42, 43, 44, 45, 46]
        assert out is None


class TestMain:
    """The entry point runs all four arms and writes the manifest."""

    def test_writes_a_manifest(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A full run writes a decodable manifest whose values match stdout.

        Rebuilds the entire expected stdout from the decoded manifest and
        asserts equality, so the report and the artifact cannot drift.
        """
        write_rw_value_fixture(tmp_path)
        out = tmp_path / "manifest.json"
        exit_code = main(_small_args(tmp_path, out))
        assert exit_code == 0
        decoded = narrow_json_to_dict(load_json_str(out.read_text(encoding="utf-8")))
        results = narrow_json_to_list(decoded["results"])
        assert len(results) == 4
        expected_lines = [
            "regression corpus: rw_value (grouped split), seeds [42]",
        ]
        for entry in results:
            record = narrow_json_to_dict(entry)
            quality = narrow_json_to_dict(record["quality"])
            model = narrow_json_to_str(record["model"])
            seed = narrow_json_to_int(record["seed"])
            rmse = narrow_json_to_float(quality["rmse"])
            mae = narrow_json_to_float(quality["mae"])
            r_squared = narrow_json_to_float(quality["r_squared"])
            fit_seconds = narrow_json_to_float(record["fit_seconds"])
            expected_lines.append(
                f"  {model:>18} seed={seed} rmse={rmse:.6f} mae={mae:.6f} "
                f"r2={r_squared:.6f} fit={fit_seconds:.3f}s"
            )
        expected_lines.append(f"manifest -> {out}")
        captured = capsys.readouterr()
        assert captured.out == "\n".join(expected_lines) + "\n"

    def test_runs_without_an_output_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Omitting --out reports to stdout without a manifest line."""
        write_rw_value_fixture(tmp_path)
        exit_code = main(_small_args(tmp_path))
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        assert len(lines) == 5
        assert lines[0] == "regression corpus: rw_value (grouped split), seeds [42]"

    def test_module_entry_point_raises_system_exit(self, tmp_path: Path) -> None:
        """Running as ``__main__`` exits through SystemExit(main())."""
        write_rw_value_fixture(tmp_path)
        argv = ["benchmark_cleargbm_regression", *_small_args(tmp_path)]
        saved = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("scripts.benchmark_cleargbm_regression", run_name="__main__")
            assert excinfo.value.code == 0
        finally:
            sys.argv = saved
