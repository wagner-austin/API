"""Tests for the multiclass benchmark command-line entry point.

Runs the real script with both real learners on a small corpus, so the
wiring between parser, runner, and manifest output is exercised end to end.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import CPU_STACK, NativeLibrariesAlreadyLoadedError
from platform_core.determinism_env import BLAS_THREAD_ENV_VARS, SINGLE_THREAD
from platform_core.determinism_record import DeterminismRecord, determinism_record
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)
from scripts.benchmark_cleargbm_multiclass import build_parser, main


def _stand_in_pin() -> DeterminismRecord:
    """Report the posture a production run would have, without pinning.

    The real pin refuses once numpy is loaded, and this is a numpy suite. The
    record returned is what `apply_cpu_determinism` produces at one thread, so
    assertions about the manifest see production's shape.

    Substituting this does not weaken the guarantee: whether the pin REFUSES
    when it cannot take is asserted by the entry-point test below.

    Returns:
        The single-thread CPU posture.
    """
    return determinism_record(CPU_STACK, dict.fromkeys(BLAS_THREAD_ENV_VARS, SINGLE_THREAD))


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
        exit_code = main(_small_args(out), pin=_stand_in_pin)
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
        exit_code = main(_small_args(), pin=_stand_in_pin)
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        assert len(lines) == 3
        assert lines[0] == "multiclass corpus: 400 rows x 4 features, 3 classes, seeds [42]"

    def test_the_entry_point_refuses_once_numpy_is_loaded(self) -> None:
        """Running as ``__main__`` in a numpy process hits the pin's refusal.

        This is the CORRECT outcome and it used to be `SystemExit(0)`. Before
        2026-08-27 this script pinned nothing, so `__main__` simply ran. It
        now pins first, and the pin refuses when a native numeric library is
        already loaded rather than reporting a posture the process does not
        have. A test asserting a clean exit here would be asserting that the
        pin does not work.
        """
        import numpy

        # Loading it IS the precondition, and asserting on it is how that
        # precondition stays visible rather than looking like a stray import.
        assert "numpy" in sys.modules
        assert numpy.__name__ == "numpy"

        saved = sys.argv
        sys.argv = ["benchmark_cleargbm_multiclass", *_small_args()]
        try:
            with pytest.raises(NativeLibrariesAlreadyLoadedError):
                runpy.run_module("scripts.benchmark_cleargbm_multiclass", run_name="__main__")
        finally:
            sys.argv = saved
