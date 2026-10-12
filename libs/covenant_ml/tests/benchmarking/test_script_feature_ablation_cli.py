"""Tests for the feature-ablation entry point, run against a real learner."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record, run_record_sidecar
from scripts.benchmark_feature_ablation import main

from covenant_ml.benchmarking.declarations import FEATURE_ABLATION
from covenant_ml.evaluation.codec_manifests import decode_ablation_manifest
from covenant_ml.evaluation.types import AblationManifest

from ._cli_fixtures import interval_flags, model_flags, stand_in_pin, write_corpus


def _args(csv_path: Path, out_path: Path, model: str = "xgboost") -> list[str]:
    return [
        "--out",
        str(out_path),
        "--csv",
        str(csv_path),
        "--seeds",
        "1",
        "2",
        *model_flags(),
        "--model",
        model,
        "--groups",
        "X1;noise=X3",
        "--fractions",
        "0.5,1.0",
        *interval_flags("cluster"),
    ]


def _read(path: Path) -> AblationManifest:
    return decode_ablation_manifest(load_json_str(path.read_text(encoding="utf-8")))


def test_a_run_ablates_each_named_group_and_traces_the_curve(tmp_path: Path) -> None:
    out_path = tmp_path / "ablation.json"
    assert main(_args(write_corpus(tmp_path), out_path), pin=stand_in_pin) == 0
    manifest = _read(out_path)
    assert manifest["model"] == "xgboost"
    assert manifest["groups"] == [
        {"name": "X1", "features": ["X1"]},
        {"name": "noise", "features": ["X3"]},
    ]
    assert {r["arm"] for r in manifest["results"]} == {"full", "without:X1", "without:noise"}
    assert len(manifest["learning_curve"]) == 4


def test_the_record_names_each_groups_pooled_effect(tmp_path: Path) -> None:
    out_path = tmp_path / "ablation.json"
    main(_args(write_corpus(tmp_path), out_path), pin=stand_in_pin)
    record = decode_run_record(
        narrow_json_to_dict(load_json_str(run_record_sidecar(out_path).read_text("utf-8")))
    )
    names = {o["name"] for o in record["observations"]}
    assert record["experiment"] == FEATURE_ABLATION["experiment"]
    assert "pooled_without:X1_vs_full.auc.difference" in names
    assert "mean_full.brier" in names
    assert record["label"].endswith("-xgboost-cluster-2seeds")


def test_an_unknown_model_is_refused_before_any_fit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--model"):
        main(_args(write_corpus(tmp_path), tmp_path / "a.json", model="catboost"), pin=stand_in_pin)
    assert not (tmp_path / "a.json").exists()


def test_the_entry_point_refuses_once_numpy_is_loaded(tmp_path: Path) -> None:
    """Run as ``__main__`` in this numpy process, the real pin must refuse."""
    import numpy

    assert numpy.__name__ in sys.modules
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_feature_ablation.py"
    original_argv = sys.argv
    sys.argv = [str(script), *_args(write_corpus(tmp_path), tmp_path / "a.json")]
    try:
        with pytest.raises(NativeLibrariesAlreadyLoadedError):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv
