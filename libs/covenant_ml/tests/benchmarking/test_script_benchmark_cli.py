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
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from scripts.benchmark_cleargbm_vs_lightgbm import DEFAULT_CSV, build_parser, main

from covenant_ml.benchmarking.types import (
    MANIFEST_SCHEMA_VERSION,
    BenchmarkManifest,
    decode_benchmark_manifest,
)


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


def cli_args(csv_path: Path) -> list[str]:
    """Build a fast argument list for the script.

    Args:
        csv_path: Input CSV.

    Returns:
        Command-line arguments.
    """
    return [
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


def test_parser_defaults_point_at_the_bundled_dataset() -> None:
    parsed = build_parser().parse_args([])
    csv_path: Path = parsed.csv
    trees: int = parsed.trees
    max_depth: int = parsed.max_depth
    assert csv_path == DEFAULT_CSV
    assert trees == 200
    assert max_depth == 6


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
    exit_code = main([*cli_args(csv_path), "--out", str(out_path)])
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
    exit_code = main([*cli_args(csv_path), "--variants", "--out", str(out_path)])
    manifest = read_manifest(out_path)

    assert exit_code == 0
    measured = {result["model"] for result in manifest["results"]}
    assert measured == {"cleargbm", "cleargbm@leaf_wise", "lightgbm", "xgboost"}


def test_run_records_exactly_one_leading_model_per_seed(tmp_path: Path) -> None:
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "manifest.json"
    main([*cli_args(csv_path), "--out", str(out_path)])
    manifest = read_manifest(out_path)

    leaders = [result for result in manifest["results"] if result["position"] == 0]
    assert len(leaders) == 1


def test_run_writes_a_decodable_manifest(tmp_path: Path) -> None:
    csv_path = write_dataset(tmp_path)
    out_path = tmp_path / "nested" / "manifest.json"
    exit_code = main([*cli_args(csv_path), "--out", str(out_path)])

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
    main([*cli_args(csv_path), "--out", str(out_path)])
    config = read_manifest(out_path)["config"]

    assert config["n_estimators"] == 3
    assert config["max_depth"] == 2
    assert config["max_bins"] == 8
    assert config["num_leaves"] == 3
    assert config["repeats"] == 1
    assert config["warmups"] == 0


def test_run_without_out_writes_no_file(tmp_path: Path) -> None:
    csv_path = write_dataset(tmp_path)
    main(cli_args(csv_path))
    assert list(tmp_path.glob("*.json")) == []


def test_module_entrypoint_exits_zero(tmp_path: Path) -> None:
    """Executing the script as ``__main__`` propagates main's exit code."""
    csv_path = write_dataset(tmp_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_cleargbm_vs_lightgbm.py"
    original_argv = sys.argv
    sys.argv = [str(script), *cli_args(csv_path)]
    module_name = "scripts.benchmark_cleargbm_vs_lightgbm"
    if module_name in sys.modules:
        del sys.modules[module_name]
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    sys.argv = original_argv

    assert exit_info.value.code == 0
