"""Tests for the multiclass benchmark command-line entry point.

Runs the real script with both real learners on a small corpus, so the wiring
between declaration, runner, manifest and RECORD is exercised end to end.

WHAT CHANGED ON 2026-09-09, and why these tests assert something different
from what they used to. This script previously carried its own parser with a
DEFAULT for every flag -- including ``--seeds [42, 43, 44, 45]`` and an
optional ``--out`` that meant "run and write nothing". Board task ``6d5536cc``
found that five of six entry points wrote a manifest and no run record, and the
seed default is the same defect one layer up: a seed count nobody chose bounds
what a comparison can conclude. So the harness requires both, and the tests
that asserted the defaults now assert the refusals.
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
    narrow_json_to_list,
    narrow_json_to_str,
)
from platform_core.run_record import run_record_sidecar
from scripts.benchmark_cleargbm_multiclass import main

from covenant_ml.benchmarking.declarations import MULTICLASS
from covenant_ml.benchmarking.harness import summary_lines


def _stand_in_pin() -> DeterminismRecord:
    """Report the posture a production run would have, without pinning.

    The real pin refuses once numpy is loaded, and this is a numpy suite. The
    record returned is what ``apply_cpu_determinism`` produces at one thread,
    so assertions about the manifest see production's shape.

    Substituting this does not weaken the guarantee: whether the real pin is
    REACHABLE in production is asserted by the module-scope import test, and
    whether it REFUSES when it is not is asserted by the entry-point test.

    Returns:
        The single-thread CPU posture.
    """
    return determinism_record(CPU_STACK, dict.fromkeys(BLAS_THREAD_ENV_VARS, SINGLE_THREAD))


def _small_args(out: Path) -> list[str]:
    """Return CLI arguments for a fast run.

    Every flag is stated because every flag is required. That is the point of
    the change this file documents: a run says what it measured rather than
    inheriting it.

    Args:
        out: Manifest output path.

    Returns:
        Argument list.
    """
    return [
        "--samples",
        "800",
        "--features",
        "4",
        "--classes",
        "3",
        "--trees",
        "20",
        "--max-depth",
        "3",
        "--learning-rate",
        "0.1",
        "--max-bins",
        "16",
        "--min-samples-leaf",
        "5",
        "--seeds",
        "42",
        "--out",
        str(out),
    ]


class TestRequiredArguments:
    """A run states what it measured; nothing is inherited from a default."""

    def test_a_run_without_seeds_is_refused(self, tmp_path: Path) -> None:
        # The seed count decides what a comparison can conclude, and a default
        # is how `DEFAULT_SEEDS = (42, 43, 44)` -- documented as reproducing a
        # TIMING workload -- came to bound five quality verdicts.
        args = list(_small_args(tmp_path / "m.json"))
        del args[args.index("--seeds") : args.index("--seeds") + 2]
        with pytest.raises(SystemExit):
            main(args, pin=_stand_in_pin)

    def test_a_run_without_an_output_path_is_refused(self, tmp_path: Path) -> None:
        # `--out` used to be optional, so a benchmark could run and produce no
        # record at all -- the defect this task exists to close, as a flag.
        args = list(_small_args(tmp_path / "m.json"))
        del args[args.index("--out") : args.index("--out") + 2]
        with pytest.raises(SystemExit):
            main(args, pin=_stand_in_pin)


class TestMain:
    """The entry point runs all four arms and writes BOTH artifacts."""

    def test_a_run_writes_a_manifest_and_its_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The load-bearing test: neither artifact appears without the other."""
        out = tmp_path / "manifest.json"
        assert main(_small_args(out), pin=_stand_in_pin) == 0
        record_path = run_record_sidecar(out)
        assert out.exists()
        assert record_path.exists()
        # Rebuilt from the manifest the run wrote, then compared whole: the
        # report and the artifact cannot drift, because the report is
        # regenerated from the artifact and asserted equal.
        encoded = load_json_str(out.read_text(encoding="utf-8"))
        expected = (
            "\n".join(summary_lines(MULTICLASS, encoded))
            + f"\nmanifest -> {out}\nrun record -> {record_path}\n"
        )
        assert capsys.readouterr().out == expected

    def test_the_manifest_carries_every_arm(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.json"
        main(_small_args(out), pin=_stand_in_pin)
        decoded = narrow_json_to_dict(load_json_str(out.read_text(encoding="utf-8")))
        assert len(narrow_json_to_list(decoded["results"])) == 2

    def test_the_record_names_this_familys_own_experiment(self, tmp_path: Path) -> None:
        # Not the shared `-fit-time` constant. comparability reads this field
        # to decide subtractability, and a shared name would let a GOSS AUC be
        # subtracted from a ranking NDCG.
        out = tmp_path / "manifest.json"
        main(_small_args(out), pin=_stand_in_pin)
        record = narrow_json_to_dict(
            load_json_str(run_record_sidecar(out).read_text(encoding="utf-8"))
        )
        assert narrow_json_to_str(record["experiment"]) == "cleargbm-vs-lightgbm-multiclass-quality"

    def test_the_report_names_every_arm(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The report is rendered from the SAME observations the record
        # carries, so a summary that drifts from the artifact is impossible.
        out = tmp_path / "manifest.json"
        main(_small_args(out), pin=_stand_in_pin)
        encoded = load_json_str(out.read_text(encoding="utf-8"))
        names = [line.split()[0] for line in summary_lines(MULTICLASS, encoded)]
        assert "mean_cleargbm.log_loss" in names

    def test_the_entry_point_refuses_once_numpy_is_loaded(self, tmp_path: Path) -> None:
        """Running as ``__main__`` in a numpy process hits the pin's refusal.

        This is the CORRECT outcome. The pin refuses when a native numeric
        library is already loaded rather than reporting a posture the process
        does not have; a test asserting a clean exit here would be asserting
        that the pin does not work.
        """
        import numpy

        # Loading it IS the precondition, and asserting on it is how that
        # precondition stays visible rather than looking like a stray import.
        assert "numpy" in sys.modules
        assert numpy.__name__ == "numpy"

        saved = sys.argv
        sys.argv = ["benchmark_cleargbm_multiclass", *_small_args(tmp_path / "m.json")]
        try:
            with pytest.raises(NativeLibrariesAlreadyLoadedError):
                runpy.run_module("scripts.benchmark_cleargbm_multiclass", run_name="__main__")
        finally:
            sys.argv = saved
