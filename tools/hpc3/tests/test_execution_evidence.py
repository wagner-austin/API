"""Native accounting and independently verified artifact refusal boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, JSONValue, require_dict
from platform_core.run_record import Observation

from hpc3.contracts.execution_evidence import decode_execution_evidence, encode_execution_evidence
from hpc3.core import _test_hooks
from hpc3.core.execution_evidence import accounting, measure, parse_checksums, replay_report
from tests.conftest import FakeRun, cluster
from tests.execution_fixtures import (
    CHECKSUMS,
    END,
    IMAGE,
    INSTANT,
    START,
    accounting_row,
    documents,
    entry,
    serve,
)


@pytest.mark.parametrize(
    "text", ["", "wrong", f"{IMAGE}  relative", f"{IMAGE}  /a/../b", f"{IMAGE}  /a\n{IMAGE}  /a"]
)
def test_bad_checksum_manifest_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="HPC_EVIDENCE_INPUTS"):
        parse_checksums(text)


def test_checksum_manifest_preserves_paths_and_accepts_binary_marker() -> None:
    assert parse_checksums(f"{IMAGE}  /pub/a space\n{'b' * 64} */pub/b\n") == {
        "/pub/a space": IMAGE,
        "/pub/b": "b" * 64,
    }


def test_accounting_is_queried_in_utc_with_full_job_name(fake_run: FakeRun) -> None:
    value = entry()
    fake_run.add("sacct", stdout=accounting_row(value))
    result = accounting(value, cluster())
    assert result["job"]["job_id"] == "101"
    assert result["job"]["elapsed_seconds"] == 11
    assert (result["exit_code"], result["started_at"], result["ended_at"]) == ("0:0", START, END)
    assert decode_execution_evidence(encode_execution_evidence(result), cluster()) == result
    assert fake_run.commands()[0].startswith("env TZ=UTC sacct ")
    assert "JobName%200" in fake_run.commands()[0]


@pytest.mark.parametrize("row", ["", "x|y", "a\nb"])
def test_missing_partial_or_multiple_accounting_rows_fail(fake_run: FakeRun, row: str) -> None:
    fake_run.add("sacct", stdout=row)
    with pytest.raises(ValueError, match="HPC_EVIDENCE_ACCOUNTING"):
        accounting(entry(), cluster())


def test_job_id_cannot_inject_a_scheduler_option() -> None:
    with pytest.raises(ValueError, match="numeric allocation"):
        accounting(entry("101;true"), cluster())


@pytest.mark.parametrize(
    "changes",
    [
        {"job": "102"},
        {"name": "another"},
        {"partition": "free"},
        {"state": "RUNNING"},
        {"exit": "1:0"},
        {"node": ""},
        {"start": END, "end": START},
        {"start": START + "+00:00"},
        {"end": END + "+00:00"},
    ],
)
def test_inconsistent_or_incomplete_accounting_fails(
    fake_run: FakeRun, changes: dict[str, str]
) -> None:
    fake_run.add("sacct", stdout=accounting_row(entry(), **changes))
    with pytest.raises(ValueError, match="HPC_EVIDENCE_ACCOUNTING"):
        accounting(entry(), cluster())


def test_unknown_scheduler_timestamp_is_not_a_completion(fake_run: FakeRun) -> None:
    fake_run.add("sacct", stdout=accounting_row(entry(), start="Unknown"))
    with pytest.raises(ValueError, match="Invalid isoformat"):
        accounting(entry(), cluster())


def test_evidence_codec_requires_exact_fields() -> None:
    with pytest.raises(ValueError, match="exact evidence fields"):
        decode_execution_evidence({}, cluster())


@pytest.mark.parametrize(
    "changes", [{"artifact": None}, {"image_digest": ""}, {"image_digest": "b" * 64}]
)
def test_missing_artifact_or_unpinned_runtime_fails(changes: dict[str, JSONValue]) -> None:
    with pytest.raises(ValueError, match="HPC_EVIDENCE_IDENTITY"):
        measure(entry(**changes), cluster(), CHECKSUMS)


def test_measured_report_binds_scheduler_and_verified_result(fake_run: FakeRun) -> None:
    value = entry()
    manifest, record = documents()
    serve(fake_run, value, manifest, record)
    measured = measure(value, cluster(), CHECKSUMS)
    assert measured.record["observations"] == (Observation(name="score", value=0.25),)
    assert measured.report["verified_files"] == CHECKSUMS
    assert measured.report["compute_checksum_lines"] == 1
    assert measured.report["result_mtime"] == INSTANT
    assert measured.report["dataset"] == manifest["dataset"]


@pytest.mark.parametrize("field", ["runtime", "payload", "fingerprint"])
def test_results_must_bind_the_record_and_runtime(fake_run: FakeRun, field: str) -> None:
    manifest, record = documents(image="b" * 64 if field == "runtime" else IMAGE)
    if field == "payload":
        record["payload_digest"] = "0" * 64
    if field == "fingerprint":
        require_dict(manifest, "fingerprint")["driver_version"] = "changed"
    serve(fake_run, entry(), manifest, record)
    with pytest.raises(ValueError, match="HPC_EVIDENCE_RESULT"):
        measure(entry(), cluster(), CHECKSUMS)


@pytest.mark.parametrize("instant", [INSTANT - 6, INSTANT + 7])
def test_stale_or_later_result_is_not_this_jobs_result(fake_run: FakeRun, instant: int) -> None:
    serve(fake_run, entry(), *documents(), instant=instant)
    with pytest.raises(ValueError, match="HPC_EVIDENCE_TIME"):
        measure(entry(), cluster(), CHECKSUMS)


def test_compute_log_must_confirm_frozen_inputs(fake_run: FakeRun) -> None:
    serve(fake_run, entry(), *documents(), checksum_log="finished\n")
    with pytest.raises(ValueError, match="required compute checksum"):
        measure(entry(), cluster(), CHECKSUMS)


@pytest.mark.parametrize(
    "changes", [{"job_id": "101"}, {"host": "other"}, {"experiment": {"arm": "C"}}]
)
def test_replay_must_be_distinct_jobs_of_same_experiment(changes: JSONObject) -> None:
    with pytest.raises(ValueError, match="HPC_EVIDENCE_PAIR"):
        replay_report(entry(), entry("102", **changes), cluster(), CHECKSUMS)


def test_repetition_alone_is_not_experiment_identity() -> None:
    with pytest.raises(ValueError, match="HPC_EVIDENCE_PAIR"):
        replay_report(
            entry(experiment={"repetition": "a"}),
            entry("102", experiment={"repetition": "b"}),
            cluster(),
            CHECKSUMS,
        )


def test_changed_environment_is_not_comparable(fake_run: FakeRun) -> None:
    serve(fake_run, entry(), *documents())
    serve(fake_run, entry("102"), *documents(driver="new"))
    with pytest.raises(ValueError, match="not comparable"):
        replay_report(entry(), entry("102"), cluster(), CHECKSUMS)


@pytest.mark.parametrize("variant", ["same", "score", "empty", "missing", "payload", "bytes"])
def test_replay_requires_complete_agreement(fake_run: FakeRun, variant: str) -> None:
    left_manifest, left_record = documents(observations=() if variant == "empty" else None)
    right_manifest, right_record = documents(score=0.5 if variant in {"score", "payload"} else 0.25)
    if variant == "missing":
        right_record["observations"] = []
    if variant == "payload":
        right_record["observations"] = left_record["observations"]
    if variant == "bytes":
        right_manifest["extra"] = "changed"
    serve(fake_run, entry(), left_manifest, left_record)
    serve(fake_run, entry("102"), right_manifest, right_record)
    report = replay_report(entry(), entry("102"), cluster(), CHECKSUMS)
    assert report["comparison"] == ("identical" if variant == "same" else "different")
    assert report["kind"] == "HPC3_REPLAY_EVIDENCE"


def test_remote_failure_real_process_twin() -> None:
    """Execute the production process runner and propagate a real nonzero exit."""
    real_run = _test_hooks.run

    def failed_transport(
        argv: Sequence[str], *, stdin_bytes: bytes | None = None
    ) -> _test_hooks.CommandResult:
        assert argv[0] == "ssh"
        return real_run(
            ["sh", "-c", "echo accounting-unavailable >&2; exit 7"], stdin_bytes=stdin_bytes
        )

    _test_hooks.run = failed_transport
    with pytest.raises(AppError, match="accounting-unavailable"):
        accounting(entry(), cluster())
