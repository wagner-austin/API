"""Tests for the benchmark command-line entry point.

Runs the real script against a small generated dataset with both real
learners, so the wiring between parser, factory, runner and reporter is
exercised end to end.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import CPU_STACK, NativeLibrariesAlreadyLoadedError
from platform_core.determinism_env import BLAS_THREAD_ENV_VARS, SINGLE_THREAD
from platform_core.determinism_record import DeterminismRecord, determinism_record
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record, run_record_sidecar
from scripts.benchmark_cleargbm_vs_lightgbm import main

from covenant_ml.benchmarking.provenance import BENCHMARK_EXPERIMENT
from covenant_ml.benchmarking.types import (
    MANIFEST_SCHEMA_VERSION,
    BenchmarkManifest,
)
from covenant_ml.benchmarking.types_codec import decode_benchmark_manifest


def _stand_in_pin() -> DeterminismRecord:
    """Report the posture a production run would have, without pinning.

    The real pin refuses once numpy is loaded, and this is a numpy suite. The
    record returned is what `apply_cpu_determinism` produces at one thread, so
    assertions about the manifest see production's shape.

    Substituting this does not weaken the guarantee: whether the pin REFUSES
    when it cannot take is asserted by the entry-point test.

    Returns:
        The single-thread CPU posture.
    """
    return determinism_record(CPU_STACK, dict.fromkeys(BLAS_THREAD_ENV_VARS, SINGLE_THREAD))


def write_dataset(directory: Path, n_companies: int = 40) -> Path:
    """Write a small but learnable CSV.

    Args:
        directory: Destination directory.
        n_companies: Distinct companies to generate.

    Returns:
        Path to the written CSV.
    """
    lines = ["company_name,status_label,year,X1,X2"]
    for company in range(n_companies):
        for row in range(4):
            x1 = (company * 4 + row) / (n_companies * 4)
            label = "failed" if x1 > 0.7 else "alive"
            lines.append(f"C_{company},{label},{2000 + row},{x1:.4f},{1.0 - x1:.4f}")
    path = directory / "small.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cli_args(csv_path: Path, out_path: Path) -> list[str]:
    """Build a fast argument list for the script.

    Every flag is stated because every flag is required. ``--out`` in
    particular: it was optional until 2026-09-09, which meant a benchmark
    could run and emit no record at all -- the defect board task ``6d5536cc``
    exists to close, previously available as a flag.

    Args:
        csv_path: Input CSV.
        out_path: Manifest output path.

    Returns:
        Command-line arguments.
    """
    return [
        "--out",
        str(out_path),
        "--csv",
        str(csv_path),
        "--seeds",
        "42",
        "--repeats",
        "1",
        "--warmups",
        "0",
        "--trees",
        "3",
        "--max-depth",
        "2",
        "--max-bins",
        "8",
        "--num-leaves",
        "3",
    ]


def test_a_run_that_states_nothing_is_refused(tmp_path: Path) -> None:
    """There are no defaults any more, and that is the change.

    This asserted a bundled default CSV and `trees == 200` until 2026-09-09.
    A benchmark whose parameters come from defaults is one whose reader cannot
    tell what was measured from the invocation -- the same shape as
    `DEFAULT_SEEDS`, documented as reproducing a TIMING workload and then
    deciding what five quality comparisons could conclude.
    """
    with pytest.raises(SystemExit):
        main([], pin=_stand_in_pin)


def read_manifest(path: Path) -> BenchmarkManifest:
    """Decode a manifest written by the script.

    Args:
        path: Manifest path.

    Returns:
        The decoded manifest.
    """
    document = narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))
    return decode_benchmark_manifest(document)


def test_run_measures_every_reference_arm_at_the_requested_seed(tmp_path: Path) -> None:
    """Without ``--variants`` the run is the reference set and nothing else."""
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "manifest.json"
    exit_code = main(cli_args(csv_path, out_path), pin=_stand_in_pin)
    manifest = read_manifest(out_path)

    assert exit_code == 0
    measured = {(result["model"], result["seed"]) for result in manifest["results"]}
    assert measured == {("cleargbm", 42), ("lightgbm", 42), ("xgboost", 42)}


def test_variants_flag_adds_the_leaf_wise_arm(tmp_path: Path) -> None:
    """``--variants`` is what puts a ClearGBM variant in the manifest.

    Without this the flag could be wired to nothing and every run would look
    identical, which is exactly the mislabelled-arm failure the axis exists to
    prevent.
    """
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "manifest.json"
    exit_code = main([*cli_args(csv_path, out_path), "--variants"], pin=_stand_in_pin)
    manifest = read_manifest(out_path)

    assert exit_code == 0
    measured = {result["model"] for result in manifest["results"]}
    assert measured == {"cleargbm", "cleargbm@leaf_wise", "lightgbm", "xgboost"}


def test_run_records_exactly_one_leading_model_per_seed(tmp_path: Path) -> None:
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "manifest.json"
    main(cli_args(csv_path, out_path), pin=_stand_in_pin)
    manifest = read_manifest(out_path)

    leaders = [result for result in manifest["results"] if result["position"] == 0]
    assert len(leaders) == 1


def test_run_writes_a_decodable_manifest(tmp_path: Path) -> None:
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "nested" / "manifest.json"
    exit_code = main(cli_args(csv_path, out_path), pin=_stand_in_pin)

    assert exit_code == 0
    assert out_path.is_file()

    manifest = read_manifest(out_path)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["estimator"] == "median"
    assert manifest["seeds"] == [42]
    assert len(manifest["results"]) == 3


def test_run_applies_the_requested_hyperparameters(tmp_path: Path) -> None:
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "manifest.json"
    main(cli_args(csv_path, out_path), pin=_stand_in_pin)
    config = read_manifest(out_path)["config"]

    assert config["n_estimators"] == 3
    assert config["max_depth"] == 2
    assert config["max_bins"] == 8
    assert config["num_leaves"] == 3
    assert config["repeats"] == 1
    assert config["warmups"] == 0


def test_a_run_without_an_output_path_is_refused(tmp_path: Path) -> None:
    """`--out` was optional, which made emitting no record a supported mode."""
    csv_path = write_dataset(tmp_path)
    args = cli_args(csv_path, tmp_path / "m.json")
    del args[args.index("--out") : args.index("--out") + 2]
    with pytest.raises(SystemExit):
        main(args, pin=_stand_in_pin)


def test_the_entry_point_refuses_once_numpy_is_loaded(tmp_path: Path) -> None:
    """Executing the script as ``__main__`` in a numpy process hits the refusal.

    This is the CORRECT outcome and it used to be `SystemExit(0)`. Before
    2026-08-27 this script pinned nothing, so `__main__` simply ran -- and
    this is the entry point whose manifests carry the headline TIMING claim,
    so its fit times were taken at whatever thread count the shell inherited.
    It now pins first, and the pin refuses when a native numeric library is
    already loaded rather than reporting a posture the process does not have.

    A test asserting a clean exit here would be asserting that the pin does
    not work.
    """
    csv_path = write_dataset(tmp_path)
    import numpy

    # Loading it IS the precondition, and asserting on it is how that
    # precondition stays visible rather than looking like a stray import.
    assert "numpy" in sys.modules
    assert numpy.__name__ == "numpy"

    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_cleargbm_vs_lightgbm.py"
    original_argv = sys.argv
    sys.argv = [str(script), *cli_args(csv_path, tmp_path / "m.json")]
    module_name = "scripts.benchmark_cleargbm_vs_lightgbm"
    if module_name in sys.modules:
        del sys.modules[module_name]
    try:
        with pytest.raises(NativeLibrariesAlreadyLoadedError):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv


def test_run_writes_a_run_record_beside_the_manifest(tmp_path: Path) -> None:
    """A benchmark that emitted only its manifest is one no cross-experiment
    contrast can read.

    The manifest and the record are both written because neither contains the
    other: the manifest holds the per-seed detail, the record holds the claim
    in the vocabulary `platform_core.run_record` checks comparability in.
    """
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "manifest.json"
    exit_code = main(cli_args(csv_path, out_path), pin=_stand_in_pin)

    assert exit_code == 0
    sidecar = run_record_sidecar(out_path)
    assert sidecar.is_file()

    record = decode_run_record(
        narrow_json_to_dict(load_json_str(sidecar.read_text(encoding="utf-8")))
    )
    assert record["experiment"] == BENCHMARK_EXPERIMENT
    names = {observation["name"] for observation in record["observations"]}
    assert "normalized_ratio" in names
    assert "cleargbm.mean_fit_s" in names


def test_the_record_carries_the_same_fingerprint_the_manifest_does(tmp_path: Path) -> None:
    """Two files describing one run must not disagree about what ran it."""
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "manifest.json"
    main(cli_args(csv_path, out_path), pin=_stand_in_pin)

    record = decode_run_record(
        narrow_json_to_dict(load_json_str(run_record_sidecar(out_path).read_text(encoding="utf-8")))
    )
    assert record["fingerprint"] == read_manifest(out_path)["fingerprint"]


def test_the_record_is_always_written_beside_the_manifest(tmp_path: Path) -> None:
    """The inverse of what this test asserted before.

    It used to check that asking for nothing wrote nothing. There is no longer
    a way to ask for nothing, so the property worth pinning is the one that
    replaced it: a completed run leaves BOTH artifacts, every time.
    """
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "manifest.json"
    assert main(cli_args(csv_path, out_path), pin=_stand_in_pin) == 0
    assert out_path.exists()
    assert len(list(tmp_path.glob("*.runrecord.json"))) == 1
