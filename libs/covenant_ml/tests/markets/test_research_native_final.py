"""Final submissions retain validation evidence and never restore spent authority."""

from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_str
from platform_core.run_record import Observation, encode_run_record, run_record
from platform_core.testing import sample_run_fingerprint
from scripts.research_proposals import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.campaign_process import run_command
from covenant_ml.markets.research_cluster import (
    ResearchCluster,
    decode_final_execution,
    decode_research_cluster,
    encode_research_cluster,
)
from covenant_ml.markets.research_delivery import deliver_final_grant
from covenant_ml.markets.research_journal import research_journal
from covenant_ml.markets.research_native_final import reconcile_final_attempt, submit_final_attempt
from covenant_ml.markets.research_records import encode_research_policy
from covenant_ml.markets.research_trial_records import ResearchTrial, encode_research_trial
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._research_execution_fixtures import receipt
from .test_research_delivery import Delivery, _released


@pytest.fixture(autouse=True)
def restore_runner() -> Generator[None, None, None]:
    previous = _test_hooks.campaign_runner
    yield
    _test_hooks.campaign_runner = previous


class Native:
    """Native receipt seam; real failed launches below leave a spent submission intent."""

    def __init__(self, trial: ResearchTrial, final: Path) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.state = "COMPLETED"
        self.change = ""
        self.trial, self.final = trial, final

    def __call__(
        self,
        command: tuple[str, ...],
        directory: Path,
        log: Path,
        *,
        stdin_bytes: bytes | None = None,
    ) -> int:
        assert stdin_bytes is None
        self.calls.append(command)
        if command[2].endswith("stage"):
            log.write_text("staged")
            return 0
        row = receipt(directory)
        if command[2].endswith("submit"):
            log.write_text(dump_json_str(row))
            return 0
        policy = self.trial["policy"]
        record = run_record(
            policy["protocol_digest"],
            f"final-{self.trial['number']}",
            sample_run_fingerprint(image_digest=policy["image_digest"]),
            (Observation(name="score", value=0.5),),
            "e" * 64,
        )
        accounting: JSONObject = {"job": {"state": self.state}, "exit_code": "0:0"}
        encoded = encode_run_record(record)
        if self.change == "label":
            encoded["label"] = "validation-1"
        elif self.change == "experiment":
            encoded["experiment"] = "different"
        elif self.change == "image":
            require_dict(encoded, "fingerprint")["image_digest"] = "e" * 64
        digest = sha256(self.final.read_bytes()).hexdigest()
        report: JSONObject = {
            "ledger": row,
            "accounting": accounting,
            "record": encoded,
            "measurement": {
                "accounting": accounting,
                "verified_files": {"/pub/final/bundle.json": digest},
                "dataset": {"bundle_sha256": digest, "bundle": read_document(self.final)},
            },
        }
        log.write_text(dump_json_str(report))
        return 0


def _native(root: Path) -> tuple[Path, ResearchTrial, ResearchCluster, Path, Native]:
    ledger, trial, final, grant, cluster = _released(root)
    _test_hooks.campaign_runner = Delivery(grant, cluster)
    deliver_final_grant(ledger, trial, cluster, grant, final, "/private/final")
    document = encode_research_cluster(cluster)
    document["checksums"] = (
        cluster["checksums"] + f"{sha256(final.read_bytes()).hexdigest()}  /pub/final/bundle.json\n"
    )
    cluster = decode_research_cluster(document)
    native = Native(trial, final)
    _test_hooks.campaign_runner = native
    return ledger, trial, cluster, grant, native


def test_final_submission_and_verified_result_leave_validation_unchanged(tmp_path: Path) -> None:
    ledger, trial, cluster, grant, native = _native(tmp_path)
    before = research_journal(trial["policy"]).read(ledger)
    assert submit_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json") == {
        "state": "running",
        "native_job_id": "123",
        "record_sha256": None,
    }
    recipe = read_document(grant / "native/run.json")
    assert (
        require_dict(recipe, "experiment")["research_attempt"]
        == require_str(
            require_dict(read_document(grant / "delivery.json"), "provenance"), "research_attempt"
        )
        + ":final"
    )
    command = recipe["command"]
    assert isinstance(command, str)
    assert "--final-grant /private/final" in command
    assert "--final-manifest /pub/final/bundle.json" in command
    assert "--access" not in command
    with pytest.raises(FileExistsError):
        submit_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")
    result = reconcile_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")
    observed = read_document(grant / "native/observe-0.log")
    assert result == {
        "state": "complete",
        "native_job_id": "123",
        "record_sha256": sha256(
            dump_json_str(require_dict(observed, "record")).encode()
        ).hexdigest(),
    }
    assert research_journal(trial["policy"]).read(ledger) == before
    assert len(native.calls) == 3


@pytest.mark.parametrize("case", ["provenance", "marker", "key", "receipt", "files"])
def test_changed_delivery_authority_never_submits(tmp_path: Path, case: str) -> None:
    ledger, trial, cluster, grant, native = _native(tmp_path)
    if case in {"provenance", "files"}:
        document = read_document(grant / "delivery.json")
        if case == "provenance":
            require_dict(document, "provenance")["request_sha256"] = "a" * 64
        else:
            document["files"] = [{"name": "different", "size_bytes": 1, "sha256": "a" * 64}]
        (grant / "delivery.json").write_text(dump_json_str(document))
    elif case == "marker":
        (grant / "consumed").write_bytes(b"different")
    elif case == "key":
        (grant / "final.agekey").write_bytes(b"must-not-exist")
    else:
        (grant / "delivery.log").write_text("transfer failed\n")
    with pytest.raises(ValueError, match="RESEARCH_FINAL_NATIVE"):
        submit_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")
    assert native.calls == []
    assert not (grant / "native").exists()


@pytest.mark.parametrize("change", ["label", "experiment", "image"])
def test_final_result_lineage_is_required(tmp_path: Path, change: str) -> None:
    ledger, trial, cluster, grant, native = _native(tmp_path)
    submit_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")
    native.change = change
    with pytest.raises(ValueError, match="RESEARCH_FINAL_NATIVE"):
        reconcile_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")


@pytest.mark.parametrize(
    "state, expected", [("PENDING", "running"), ("TIMEOUT", "failed"), ("ALIEN", "error")]
)
def test_native_states_never_restore_permission(tmp_path: Path, state: str, expected: str) -> None:
    ledger, trial, cluster, grant, native = _native(tmp_path)
    submit_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")
    native.state = state
    if expected == "error":
        with pytest.raises(ValueError, match="unknown scheduler state"):
            reconcile_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")
    else:
        assert reconcile_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json") == {
            "state": expected,
            "native_job_id": "123",
            "record_sha256": None,
        }
    assert len(native.calls) == 3


def test_real_launch_failure_keeps_submission_reserved(tmp_path: Path) -> None:
    ledger, trial, cluster, grant, _ = _native(tmp_path)
    document = encode_research_cluster(cluster)
    document["python"] = str(tmp_path / "absent-python")
    cluster = decode_research_cluster(document)
    _test_hooks.campaign_runner = run_command
    with pytest.raises(FileNotFoundError):
        submit_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")
    with pytest.raises(FileExistsError):
        submit_final_attempt(ledger, trial, cluster, grant, "/pub/final/bundle.json")
    assert (grant / "native/run.json").is_file()
    assert not (grant / "final.agekey").exists()


@pytest.mark.parametrize("case", ["fields", "relative", "traversal"])
def test_final_paths_are_strict(case: str) -> None:
    document: JSONObject = {
        "manifest": "/pub/final.json",
        "manifest_digest": "a" * 64,
        "grant_directory": "/private/grant",
    }
    if case == "fields":
        document["extra"] = True
    else:
        document["grant_directory"] = "relative" if case == "relative" else "/private/../escape"
    with pytest.raises(ValueError, match="RESEARCH_FINAL_EXECUTION"):
        decode_final_execution(document)


def test_final_cli_uses_the_same_controller(tmp_path: Path) -> None:
    ledger, trial, cluster, grant, native = _native(tmp_path)
    for name, document in (
        ("policy", encode_research_policy(trial["policy"])),
        ("trial", encode_research_trial(trial)),
        ("cluster", encode_research_cluster(cluster)),
    ):
        (tmp_path / f"{name}.json").write_text(dump_json_str(document))
    base = ["--policy", str(tmp_path / "policy.json"), "--ledger", str(ledger)]
    args = [
        "--trial",
        str(tmp_path / "trial.json"),
        "--cluster",
        str(tmp_path / "cluster.json"),
        "--directory",
        str(grant),
        "--remote-manifest",
        "/pub/final/bundle.json",
    ]
    assert main([*base, "submit-final", *args], stand_in_pin) == 0
    assert main([*base, "reconcile-final", *args], stand_in_pin) == 0
    assert len(native.calls) == 3
