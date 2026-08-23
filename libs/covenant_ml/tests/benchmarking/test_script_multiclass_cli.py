"""Tests for the multiclass benchmark command-line entry point.

Runs the real script with both real learners on a small corpus, so the
wiring between parser, runner, and manifest output is exercised end to end.
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
from scripts.benchmark_cleargbm_multiclass import build_parser, main


def _small_args(out: Path | None = None) -> list[str]:
    """Return CLI arguments for a fast run.

    Args:
        out: Optional manifest output path.

    Returns:
        Argument list.
    """
    args = [
        "--samples",
        "400",
        "--features",
        "4",
        "--classes",
        "3",
        "--trees",
        "10",
        "--max-depth",
        "3",
        "--max-bins",
        "16",
        "--min-samples-leaf",
        "5",
        "--seeds",
        "42",
    ]
    if out is not None:
        args.extend(["--out", str(out)])
    return args


class TestParser:
    """Defaults match the documented benchmark shape."""

    def test_defaults(self) -> None:
        """The default corpus and seeds are the documented ones."""
        parsed = build_parser().parse_args([])
        samples: int = parsed.samples
        n_classes: int = parsed.classes
        seeds: list[int] = parsed.seeds
        out: Path | None = parsed.out
        assert samples == 6000
        assert n_classes == 5
        assert seeds == [42, 43, 44, 45]
        assert out is None


class TestMain:
    """The entry point runs both arms and writes the manifest."""

    def test_writes_a_manifest(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A full run writes a decodable manifest whose values match stdout.

        Rebuilds the entire expected stdout from the decoded manifest and
        asserts equality, so the report and the artifact cannot drift.
        """
        out = tmp_path / "manifest.json"
        exit_code = main(_small_args(out))
        assert exit_code == 0
        decoded = narrow_json_to_dict(load_json_str(out.read_text(encoding="utf-8")))
        results = narrow_json_to_list(decoded["results"])
        assert len(results) == 2
        expected_lines = [
            "multiclass corpus: 400 rows x 4 features, 3 classes, seeds [42]",
        ]
        for entry in results:
            record = narrow_json_to_dict(entry)
            quality = narrow_json_to_dict(record["quality"])
            model = narrow_json_to_str(record["model"])
            seed = narrow_json_to_int(record["seed"])
            log_loss = narrow_json_to_float(quality["log_loss"])
            accuracy = narrow_json_to_float(quality["accuracy"])
            expected_lines.append(
                f"  {model:>9} seed={seed} log_loss={log_loss:.6f} accuracy={accuracy:.4f}"
            )
        expected_lines.append(f"manifest -> {out}")
        captured = capsys.readouterr()
        assert captured.out == "\n".join(expected_lines) + "\n"

    def test_runs_without_an_output_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Omitting --out reports to stdout without a manifest line."""
        exit_code = main(_small_args())
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        assert len(lines) == 3
        assert lines[0] == "multiclass corpus: 400 rows x 4 features, 3 classes, seeds [42]"

    def test_module_entry_point_raises_system_exit(self, tmp_path: Path) -> None:
        """Running as ``__main__`` exits through SystemExit(main())."""
        argv = ["benchmark_cleargbm_multiclass", *_small_args()]
        saved = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("scripts.benchmark_cleargbm_multiclass", run_name="__main__")
            assert excinfo.value.code == 0
        finally:
            sys.argv = saved
