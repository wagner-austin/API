"""Tests for the paired-evaluation entry point, run against real learners."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record, run_record_sidecar
from scripts.benchmark_paired_evaluation import main

from covenant_ml.benchmarking.declarations import PAIRED_EVALUATION
from covenant_ml.evaluation.codec_manifests import decode_evaluation_manifest
from covenant_ml.evaluation.types import EvaluationManifest, ResamplingUnit

from ..evaluation.test_wrds import write_extract
from ._cli_fixtures import interval_flags, model_flags, stand_in_pin, write_corpus


def _args(csv_path: Path, out_path: Path, unit: str = "cluster") -> list[str]:
    return [
        "--out",
        str(out_path),
        "--csv",
        str(csv_path),
        "--seeds",
        "1",
        "2",
        *model_flags(),
        "--reference",
        "lightgbm",
        *interval_flags(unit),
        "--true-alert-value",
        "10",
        "--false-alert-cost",
        "1",
        "--missed-cost",
        "5",
    ]


def _read(path: Path) -> EvaluationManifest:
    return decode_evaluation_manifest(load_json_str(path.read_text(encoding="utf-8")))


def test_a_run_writes_a_decodable_manifest_with_every_arm(tmp_path: Path) -> None:
    out_path = tmp_path / "nested" / "paired.json"
    assert main(_args(write_corpus(tmp_path), out_path), pin=stand_in_pin) == 0
    manifest = _read(out_path)
    assert {r["arm"] for r in manifest["results"]} == {"cleargbm", "lightgbm", "xgboost"}
    assert manifest["resampling"]["unit"] is ResamplingUnit.CLUSTER
    assert manifest["alert_costs"]["missed_cost"] == 5.0
    assert manifest["config"]["n_estimators"] == 4
    assert manifest["config"]["repeats"] == 1
    assert manifest["seeds"] == [1, 2]


def test_the_record_carries_the_pooled_claim_and_the_manifests_fingerprint(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "paired.json"
    main(_args(write_corpus(tmp_path), out_path), pin=stand_in_pin)
    record = decode_run_record(
        narrow_json_to_dict(load_json_str(run_record_sidecar(out_path).read_text("utf-8")))
    )
    names = {o["name"] for o in record["observations"]}
    assert record["experiment"] == PAIRED_EVALUATION["experiment"]
    assert "pooled_cleargbm_vs_lightgbm.auc.difference" in names
    assert "pooled_xgboost_vs_lightgbm.utility.p_value" in names
    assert "mean_policy_value.never_alert" in names
    assert "mean_lightgbm.auc_roc" in names
    assert record["fingerprint"] == _read(out_path)["fingerprint"]
    assert record["label"].endswith("-cluster-2seeds")


def test_an_unknown_unit_is_refused_before_any_fit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--unit"):
        main(_args(write_corpus(tmp_path), tmp_path / "p.json", unit="company"), pin=stand_in_pin)
    assert not (tmp_path / "p.json").exists()


def test_a_run_that_states_nothing_is_refused() -> None:
    with pytest.raises(SystemExit):
        main([], pin=stand_in_pin)


def test_wrds_uses_the_same_real_learners_and_record_writer(tmp_path: Path) -> None:
    path, _ = write_extract(tmp_path)
    output = tmp_path / "paired.json"
    args = _args(path, output, unit="block")
    args[args.index("--csv")] = "--wrds-manifest"
    args[args.index("--block-length") + 1] = "3"
    assert main(args, pin=stand_in_pin) == 0
    manifest = _read(output)
    assert manifest["dataset"]["n_rows"] == 159
    assert manifest["resampling"]["unit"] is ResamplingUnit.BLOCK
    assert {r["arm"] for r in manifest["results"]} == {"cleargbm", "lightgbm", "xgboost"}
    record = decode_run_record(
        narrow_json_to_dict(load_json_str(run_record_sidecar(output).read_text("utf-8")))
    )
    assert record["label"].startswith(manifest["dataset"]["sha256"][:12])
    assert record["fingerprint"] == manifest["fingerprint"]


def test_wrds_refuses_independent_row_inference(tmp_path: Path) -> None:
    path, _ = write_extract(tmp_path)
    output = tmp_path / "paired.json"
    args = _args(path, output, unit="row")
    args[args.index("--csv")] = "--wrds-manifest"
    with pytest.raises(ValueError, match="WRDS_DATA_UNIT"):
        main(args, pin=stand_in_pin)
    assert not output.exists()


def test_the_entry_point_refuses_once_numpy_is_loaded(tmp_path: Path) -> None:
    """Run as ``__main__`` in this numpy process, the real pin must refuse."""
    import numpy

    assert numpy.__name__ in sys.modules
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_paired_evaluation.py"
    original_argv = sys.argv
    sys.argv = [str(script), *_args(write_corpus(tmp_path), tmp_path / "p.json")]
    try:
        with pytest.raises(NativeLibrariesAlreadyLoadedError):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv
