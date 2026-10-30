"""The shared queue polls final allocations through the existing private controller."""

import hashlib
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, require_dict

from fleet.core import _test_hooks, research_hpc
from tests.conftest import FakeRun, ok
from tests.test_research_hpc import _document


def _final_document(root: Path) -> JSONObject:
    document = _document(root)
    document["phase"] = "final"
    del document["native_attempt"]
    document["remote_manifest"] = "/pub/final/bundle.json"
    cluster = root / "cluster"
    cluster.write_text('{"workspace":"/shared/hpc3.json","report_task":"task"}')
    require_dict(document, "cluster")["sha256"] = hashlib.sha256(cluster.read_bytes()).hexdigest()
    return document


@pytest.mark.parametrize("started", [False, True])
def test_final_queue_submits_once_and_reconciles_retained_intent(
    tmp_path: Path, started: bool
) -> None:
    document = _final_document(tmp_path)
    if started:
        (tmp_path / "work/native").mkdir(parents=True)
    result: JSONObject = {"state": "running", "native_job_id": "123", "record_sha256": None}
    runner = FakeRun([ok(dump_json_str(result))])
    _test_hooks.run = runner
    assert research_hpc.poll(document, tmp_path / "transcript", "task") == result
    assert len(runner.calls) == 1
    assert runner.calls[0][6:] == (
        "reconcile-final" if started else "submit-final",
        "--trial",
        str(tmp_path / "trial"),
        "--cluster",
        str(tmp_path / "cluster"),
        "--directory",
        str(tmp_path / "work"),
        "--remote-manifest",
        "/pub/final/bundle.json",
    )
    assert runner.timeouts == [300]


def test_final_poll_refuses_a_different_report_task(tmp_path: Path) -> None:
    runner = FakeRun([])
    _test_hooks.run = runner
    with pytest.raises(ValueError, match="RESEARCH_HPC_FINAL_TASK"):
        research_hpc.poll(_final_document(tmp_path), tmp_path / "transcript", "different")
    assert runner.calls == []


def test_unknown_phase_refuses_before_controller_effects(tmp_path: Path) -> None:
    document = _document(tmp_path)
    document["phase"] = "holdout-preview"
    with pytest.raises(ValueError, match="RESEARCH_HPC_PHASE"):
        research_hpc.poll(document, tmp_path / "transcript", "task")
