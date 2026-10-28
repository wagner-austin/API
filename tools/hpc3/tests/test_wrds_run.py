"""Experiment boundaries reject ambiguous resources and mutable identities."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONValue

from hpc3.contracts.wrds_run import (
    absolute_path,
    decode_wrds_file,
    decode_wrds_run,
    decode_wrds_runtime,
    digest,
    encode_wrds_file,
    encode_wrds_run,
    encode_wrds_runtime,
    execution_document,
    relative_path,
    run_identity,
    safe_label,
)
from tests.wrds_support import example


def test_document_roundtrip_and_checkout_independent_identity(tmp_path: pathlib.Path) -> None:
    run = example(tmp_path)
    assert decode_wrds_run(encode_wrds_run(run)) == run
    assert decode_wrds_runtime(encode_wrds_runtime(run["runtime"])) == run["runtime"]
    assert decode_wrds_file(encode_wrds_file(run["code"][0])) == run["code"][0]
    identity = run_identity(run)
    run["code"][0]["path"] = "/another/checkout/main.py"
    assert run_identity(run) == identity
    assert execution_document(run)["code"] == [
        {"path": "main.py", "name": "main.py", "sha256": run["code"][0]["sha256"]}
    ]
    run["seed"] += 1
    assert run_identity(run) != identity


@pytest.mark.parametrize("name", ["", "..", "a/b", "a;rm", "a" * 65])
def test_unsafe_labels(name: str) -> None:
    with pytest.raises(ValueError, match="WRDS_LABEL"):
        safe_label(name)


@pytest.mark.parametrize(
    "name", ["", ".", "/x", "../x", "a/../b", "a//b", "a\nb", "a\rb", "a\x00b", "a\\b"]
)
def test_unsafe_relative_names(name: str) -> None:
    with pytest.raises(ValueError, match="WRDS_PATH"):
        relative_path(name)


def test_absolute_paths_digests_and_empty_file_source() -> None:
    assert absolute_path("/home/uci/name") == "/home/uci/name"
    assert relative_path("results/summary.json") == "results/summary.json"
    assert safe_label("repeat-2") == "repeat-2"
    assert digest("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="WRDS_PATH"):
        absolute_path("relative")
    with pytest.raises(ValueError, match="WRDS_DIGEST"):
        digest("invalid")
    with pytest.raises(ValueError, match="WRDS_PATH"):
        decode_wrds_file({"path": "", "name": "a", "sha256": "a" * 64})


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("slots", 0, "WRDS_RESOURCES"),
        ("slots", 9, "WRDS_RESOURCES"),
        ("memory_gb_per_slot", 0, "WRDS_RESOURCES"),
        ("memory_gb_per_slot", 49, "WRDS_RESOURCES"),
        ("seconds", 0, "WRDS_RESOURCES"),
        ("seconds", 604801, "WRDS_RESOURCES"),
        ("seed", -1, "WRDS_RESOURCES"),
        ("seed", 4294967296, "WRDS_RESOURCES"),
        ("root", "/tmp/wrds", "WRDS_ROOT"),
        ("code", [], "WRDS_FILES"),
        ("outputs", ["x.json"], "WRDS_FILES"),
        ("outputs", ["metrics.json", "metrics.json"], "WRDS_DUPLICATE"),
        ("outputs", ["metrics.json", "a", "a/b"], "WRDS_DUPLICATE"),
    ],
)
def test_invalid_run_fields(
    tmp_path: pathlib.Path, field: str, value: JSONValue, error: str
) -> None:
    raw = encode_wrds_run(example(tmp_path))
    raw[field] = value
    with pytest.raises(ValueError, match=error):
        decode_wrds_run(raw)


def test_unknown_fields_and_runtime_version(tmp_path: pathlib.Path) -> None:
    run = example(tmp_path)
    raw = encode_wrds_run(run)
    raw["silently_ignored"] = True
    with pytest.raises(ValueError, match="WRDS_FIELDS"):
        decode_wrds_run(raw)
    runtime = encode_wrds_runtime(run["runtime"])
    runtime["version"] = ""
    with pytest.raises(ValueError, match="WRDS_RUNTIME"):
        decode_wrds_runtime(runtime)
    raw = encode_wrds_run(run)
    raw["inputs"] = [{"path": "/wrds/table", "name": "table", "sha256": "a" * 64}]
    assert decode_wrds_run(raw)["inputs"][0]["path"] == "/wrds/table"
    raw["inputs"] = [{"path": "table", "name": "table", "sha256": "a" * 64}]
    with pytest.raises(ValueError, match="WRDS_PATH"):
        decode_wrds_run(raw)
