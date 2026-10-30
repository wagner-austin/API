"""One-shot submission, crash recovery and measured terminal reconciliation."""

import sys
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, require_dict
from platform_core.stage_manifest import decode_stage_manifest

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.campaign_process import run_command
from covenant_ml.markets.research_cluster import (
    decode_research_cluster,
    encode_research_cluster,
    prepare_bundle,
)
from covenant_ml.markets.research_execution import reconcile_attempt, submit_attempt
from covenant_ml.markets.research_journal import admit_proposal, research_journal
from covenant_ml.markets.research_trial_records import decode_research_trial, encode_research_trial
from covenant_ml.markets.validation import read_document

from ._research_execution_fixtures import Scheduler, observation, setup


@pytest.fixture(autouse=True)
def restore_runner() -> Generator[None, None, None]:
    previous = _test_hooks.campaign_runner
    yield
    _test_hooks.campaign_runner = previous


def test_submit_complete_and_refuse_duplicate(tmp_path: Path) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    scheduler = Scheduler(trial)
    _test_hooks.campaign_runner = scheduler
    assert submit_attempt(path, trial, cluster, directory)["job_id"] == "123"
    assert len(scheduler.calls) == 2
    recipe = read_document(directory / "run.json")
    assert (recipe["cpus"], recipe["mem_gb"], recipe["minutes"], recipe["gpu"]) == (1, 4, 2, None)
    assert "timeout --signal=TERM --kill-after=5s 60s python" in str(recipe["command"])
    with pytest.raises(ValueError, match="invalid attempt transition"):
        submit_attempt(path, trial, cluster, directory)
    assert len(scheduler.calls) == 2
    row = reconcile_attempt(path, trial, cluster, directory)
    assert (row["status"], row["job_id"]) == ("complete", "123")
    assert research_journal(trial["policy"]).read(path) == (row,)
    with pytest.raises(ValueError, match="require submitting or running"):
        reconcile_attempt(path, trial, cluster, directory)
    assert len(scheduler.calls) == 3


@pytest.mark.parametrize(
    "state",
    [
        "PENDING",
        "RUNNING",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "PREEMPTED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
    ],
)
def test_recovery_after_ambiguous_submit_preserves_attempt(tmp_path: Path, state: str) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    scheduler = Scheduler(trial)
    scheduler.fail = "submit"
    _test_hooks.campaign_runner = scheduler
    with pytest.raises(RuntimeError, match="exited 7"):
        submit_attempt(path, trial, cluster, directory)
    assert research_journal(trial["policy"]).read(path)[0]["status"] == "submitting"
    scheduler.fail = ""
    scheduler.state = state
    row = reconcile_attempt(path, trial, cluster, directory)
    expected = (
        "running"
        if state in {"PENDING", "RUNNING"}
        else "timeout"
        if state == "TIMEOUT"
        else "failed"
    )
    assert (row["status"], row["job_id"]) == (expected, "123")
    assert len([call for call in scheduler.calls if call[2].endswith("submit")]) == 1


@pytest.mark.parametrize("change", [{"number": 2}, {"proposal": {"hypothesis": "changed"}}])
def test_altered_admission_is_refused_before_transport(tmp_path: Path, change: JSONObject) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    document = encode_research_trial(trial)
    if "proposal" in change:
        require_dict(document, "proposal")["hypothesis"] = "changed"
    else:
        document.update(change)
    with pytest.raises(ValueError, match="differs from admission"):
        submit_attempt(path, decode_research_trial(document), cluster, directory)
    assert not directory.exists()


def test_reordered_json_preserves_raw_admission_and_executes(tmp_path: Path) -> None:
    _path, trial, cluster, directory = setup(tmp_path)
    proposal = require_dict(encode_research_trial(trial), "proposal")
    config = require_dict(proposal, "config")
    proposal["config"] = dict(reversed(tuple(config.items())))
    reordered: JSONObject = dict(reversed(tuple(proposal.items())))
    path = tmp_path / "reordered.sqlite"
    journal = research_journal(trial["policy"])
    journal.initialize(path)
    admitted = admit_proposal(path, trial["policy"], reordered, 0)
    assert admitted["status"] == "admitted"
    assert admitted["request"] == dump_json_str(reordered)
    scheduler = Scheduler(trial)
    _test_hooks.campaign_runner = scheduler
    assert submit_attempt(path, trial, cluster, directory)["job_id"] == "123"
    row = reconcile_attempt(path, trial, cluster, directory)
    assert (row["status"], row["job_id"], row["request"]) == (
        "complete",
        "123",
        admitted["request"],
    )
    assert journal.read(path) == (row,)
    assert len(scheduler.calls) == 3


@pytest.mark.parametrize("field", ["job_id", "experiment", "artifact", "image_digest"])
def test_changed_receipt_never_binds_a_job(tmp_path: Path, field: str) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    scheduler = Scheduler(trial)
    scheduler.change[field] = {"research_attempt": "wrong"} if field == "experiment" else "wrong"
    _test_hooks.campaign_runner = scheduler
    with pytest.raises(ValueError, match="ledger differs"):
        submit_attempt(path, trial, cluster, directory)
    assert research_journal(trial["policy"]).read(path)[0]["job_id"] == ""


def test_changed_job_or_measurement_cannot_complete(tmp_path: Path) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    scheduler = Scheduler(trial)
    _test_hooks.campaign_runner = scheduler
    submit_attempt(path, trial, cluster, directory)
    report = observation(directory, trial)
    require_dict(report, "ledger")["job_id"] = "456"
    scheduler.change = report
    with pytest.raises(ValueError, match="observed job changed"):
        reconcile_attempt(path, trial, cluster, directory)
    scheduler.change = {"measurement": {"accounting": {"changed": True}}}
    with pytest.raises(ValueError, match="measured accounting differs"):
        reconcile_attempt(path, trial, cluster, directory)
    assert research_journal(trial["policy"]).read(path)[0]["status"] == "running"
    assert len(tuple(directory.glob("observe-*.log"))) == 2
    scheduler.change = {}
    scheduler.state = "UNRECOGNIZED"
    with pytest.raises(ValueError, match="unknown native scheduler state"):
        reconcile_attempt(path, trial, cluster, directory)


@pytest.mark.parametrize("changed", ["digest", "verified", "partition"])
def test_changed_dataset_cannot_complete(tmp_path: Path, changed: str) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    scheduler = Scheduler(trial)
    _test_hooks.campaign_runner = scheduler
    submit_attempt(path, trial, cluster, directory)
    report = observation(directory, trial)
    measurement = require_dict(report, "measurement")
    dataset = require_dict(measurement, "dataset")
    if changed == "digest":
        dataset["bundle_sha256"] = "0" * 64
    elif changed == "verified":
        require_dict(measurement, "verified_files")[trial["manifest"]] = "0" * 64
    else:
        require_dict(dataset, "bundle")["partition"] = "final"
    scheduler.change = report
    with pytest.raises(ValueError, match="measured dataset differs"):
        reconcile_attempt(path, trial, cluster, directory)
    assert research_journal(trial["policy"]).read(path)[0]["status"] == "running"


def test_cluster_codec_and_missing_pin(tmp_path: Path) -> None:
    _path, trial, cluster, directory = setup(tmp_path)
    assert decode_research_cluster(encode_research_cluster(cluster)) == cluster
    with pytest.raises(ValueError, match="exact fields"):
        decode_research_cluster({})
    document = encode_research_cluster(cluster)
    document["checksums"] = "wrong"
    with pytest.raises(ValueError, match="require frozen runtime"):
        prepare_bundle(directory, decode_research_cluster(document), trial)
    assert not directory.exists()


def test_bundle_manifest_identifies_exact_written_bytes(tmp_path: Path) -> None:
    _path, trial, cluster, directory = setup(tmp_path)
    prepare_bundle(directory, cluster, trial)
    manifest = decode_stage_manifest(read_document(directory / "stage.json"))
    assert [item["name"] for item in manifest["files"]] == ["request.json", "checksums.txt"]
    for item in manifest["files"]:
        payload = (directory / item["name"]).read_bytes()
        assert len(payload) == item["size_bytes"]
        assert sha256(payload).hexdigest() == item["sha256"]
    request_digest = sha256((directory / "request.json").read_bytes()).hexdigest()
    expected = (
        cluster["checksums"].rstrip("\n")
        + f"\n{request_digest}  {manifest['destination']}/request.json\n"
    ).encode("utf-8")
    assert (directory / "checksums.txt").read_bytes() == expected
    assert (directory / "expected.txt").read_text().splitlines() == [
        item["sha256"] for item in manifest["files"]
    ]


@pytest.mark.parametrize("field", ["python", "workspace", "root", "entrypoint", "image"])
@pytest.mark.parametrize("value", ["relative", "/pub/../escape"])
def test_cluster_refuses_relative_and_traversing_paths(
    tmp_path: Path, field: str, value: str
) -> None:
    _path, _trial, cluster, _directory = setup(tmp_path)
    document = encode_research_cluster(cluster)
    document[field] = value
    with pytest.raises(ValueError, match="absolute non-traversing"):
        decode_research_cluster(document)


@pytest.mark.parametrize("field", ["root", "entrypoint", "image"])
def test_remote_paths_refuse_windows_drive_paths(tmp_path: Path, field: str) -> None:
    _path, _trial, cluster, _directory = setup(tmp_path)
    document = encode_research_cluster(cluster)
    document[field] = "C:/research"
    with pytest.raises(ValueError, match="absolute non-traversing"):
        decode_research_cluster(document)


def test_real_transport_failure_leaves_durable_intent(tmp_path: Path) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    document = encode_research_cluster(cluster)
    document["python"] = sys.executable
    _test_hooks.campaign_runner = run_command
    with pytest.raises(RuntimeError, match="RESEARCH_EXECUTION_PROCESS"):
        submit_attempt(path, trial, decode_research_cluster(document), directory)
    assert research_journal(trial["policy"]).read(path)[0]["status"] == "submitting"
    assert (directory / "stage.log").read_text().strip() == (
        f"{sys.executable}: Error while finding module specification for 'hpc3.cli.stage' "
        "(ModuleNotFoundError: No module named 'hpc3')"
    )
