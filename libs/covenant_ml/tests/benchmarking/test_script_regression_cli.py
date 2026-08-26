"""Tests for the regression benchmark command-line entry point.

Runs the real script with all three real learners on a small grouped
corpus, so the wiring between parser, runner, and manifest output is
exercised end to end.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.comparability import decode_run_fingerprint
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
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
from scripts.benchmark_cleargbm_regression import build_parser, main

from .test_regression_quality import write_rw_value_fixture


def _stand_in_pin() -> DeterminismRecord:
    """Report the posture a production run would have, without pinning.

    The real pin refuses once numpy is loaded, and this is a numpy suite. The
    record returned is what `apply_cpu_determinism` produces at one thread, so
    assertions about the manifest see production's shape.

    Substituting this does not weaken the guarantee: whether the real pin is
    REACHABLE in production is asserted directly by
    `test_nothing_numeric_is_imported_at_module_scope`, and whether it TAKES
    is asserted by the subprocess test.

    Returns:
        The single-thread CPU posture.
    """
    from platform_core.determinism_cpu import CPU_STACK
    from platform_core.determinism_env import BLAS_THREAD_ENV_VARS

    return determinism_record(CPU_STACK, dict.fromkeys(BLAS_THREAD_ENV_VARS, SINGLE_THREAD))


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
        exit_code = main(_small_args(tmp_path, out), _stand_in_pin)
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
        exit_code = main(_small_args(tmp_path), _stand_in_pin)
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        assert len(lines) == 5
        assert lines[0] == "regression corpus: rw_value (grouped split), seeds [42]"

    def test_the_entry_point_runs_in_a_real_process_with_the_real_pin(self, tmp_path: Path) -> None:
        """A subprocess, because in-process is the one thing this cannot be.

        The real pin refuses once numpy is loaded, and this suite has numpy
        loaded before collection -- so `runpy` in-process would exercise the
        refusal rather than the entry point. A subprocess starts clean, which
        is what a production invocation is, and an exit code of 0 therefore
        proves the pin was REACHED and TOOK rather than merely that a stand-in
        was accepted.
        """
        write_rw_value_fixture(tmp_path)
        script = Path(__file__).parents[2] / "scripts" / "benchmark_cleargbm_regression.py"
        out = tmp_path / "manifest.json"

        completed = subprocess.run(
            [sys.executable, str(script), *_small_args(tmp_path, out)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr

        # Assert on the artifact, not on stdout. The posture in the manifest
        # is the thing under test: a subprocess starts with no numeric library
        # loaded, so the real pin was reached and took, and this record was
        # produced by it rather than by a stand-in.
        decoded = narrow_json_to_dict(load_json_str(out.read_text(encoding="utf-8")))
        fingerprint = decode_run_fingerprint(decoded["fingerprint"])
        assert dict(fingerprint["determinism"]["settings"]) == dict.fromkeys(
            BLAS_THREAD_ENV_VARS, SINGLE_THREAD
        )

    def test_a_pin_that_cannot_take_stops_the_run(self, tmp_path: Path) -> None:
        """The refusal reaches the entry point rather than being swallowed.

        Imports numpy first, then runs the module in-process, which is the
        shape of the bug this replaced: the pin ran, wrote variables nobody
        would read, and reported them as the run's posture.
        """
        write_rw_value_fixture(tmp_path)
        import numpy

        # Loading it IS the precondition, and asserting on it is how that
        # precondition stays visible rather than looking like a stray import.
        assert "numpy" in sys.modules
        assert numpy.__name__ == "numpy"

        saved = sys.argv
        sys.argv = ["benchmark_cleargbm_regression", *_small_args(tmp_path)]
        try:
            with pytest.raises(NativeLibrariesAlreadyLoadedError):
                runpy.run_module("scripts.benchmark_cleargbm_regression", run_name="__main__")
        finally:
            sys.argv = saved


class TestThePinIsReachable:
    """The test that would have caught the original defect.

    Between 8c3baa07 and its fix this script imported covenant_ml -- and so
    numpy -- at module scope, then pinned from `main`. Every gate passed:
    mypy, ruff, the guards, 2,564 tests at 100% branches. The manifest it
    wrote asserted OMP_NUM_THREADS=1 for a multi-threaded run.

    Nothing about the pin's own behaviour was wrong. What was wrong was the
    IMPORT ORDER of the file calling it, which no assertion about the pin can
    see. This asserts the order directly.
    """

    def test_nothing_numeric_is_imported_at_module_scope(self) -> None:
        """Importing the script must not load a native numeric library.

        Run in a subprocess so the answer is about the script rather than
        about whatever this pytest worker imported first.
        """
        script = Path(__file__).parents[2] / "scripts" / "benchmark_cleargbm_regression.py"
        probe = (
            "import importlib.util, sys;"
            f"spec = importlib.util.spec_from_file_location('s', r'{script}');"
            "mod = importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(mod);"
            "print(','.join(m for m in "
            "('numpy','scipy','sklearn','torch','pandas') if m in sys.modules))"
        )

        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "", (
            f"importing the script loaded {completed.stdout.strip()}, so the CPU "
            "determinism pin inside main() can never take effect"
        )
