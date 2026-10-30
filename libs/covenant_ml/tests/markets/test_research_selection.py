"""Evaluator selection seals a real journal and consumes exactly one release."""

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict

from covenant_ml.markets.research_journal import (
    admit_proposal,
    decode_research_attempt,
    encode_research_attempt,
    research_journal,
)

from ._research_fixtures import completion, policy, proposal


def _completed(path: Path) -> JSONObject:
    """Complete two distinct real validation attempts and return a selection."""
    journal = research_journal(policy())
    journal.initialize(path)
    candidate = ""
    for number in (1, 2):
        request = proposal()
        require_dict(request, "config")["n_estimators"] = 100 * number
        row = admit_proposal(path, policy(), request, 1000)
        candidate = row["candidate"]
        submit: JSONObject = {"action": "submit", "number": number, "observed_ms": 1000}
        journal.transact(path, lambda _, submit=submit: submit)
        job = f"slurm:{number}"
        scheduled: JSONObject = {
            "action": "scheduled",
            "number": number,
            "observed_ms": 1001,
            "job_id": job,
        }
        journal.transact(path, lambda _, scheduled=scheduled: scheduled)
        event = completion(number)
        event["job_id"] = job
        journal.transact(path, lambda _, event=event: event)
    return {
        "action": "select",
        "number": 2,
        "observed_ms": 2001,
        "candidate": candidate,
        "validation_payload_digest": "e" * 64,
        "final_manifest_digest": "f" * 64,
    }


def test_selection_preserves_validation_and_consumes_one_final_release(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    event = _completed(path)
    journal = research_journal(policy())
    before = journal.read(path)
    selected = journal.transact(path, lambda _: event)
    assert selected[0] == before[0]
    assert selected[1]["status"] == "selected"
    assert selected[1]["final_manifest_digest"] == "f" * 64
    assert selected[1]["record"] == before[1]["record"]
    assert selected[1]["request"] == before[1]["request"]
    assert decode_research_attempt(encode_research_attempt(selected[1])) == selected[1]
    release: JSONObject = {**event, "action": "release_final", "observed_ms": 2002}
    released = journal.transact(path, lambda _: release)
    assert released[1]["status"] == "final_released"
    assert released[1]["final_manifest_digest"] == "f" * 64
    assert released[1]["candidate"] == before[1]["candidate"]
    assert released[1]["record"] == before[1]["record"]
    # A different reader after a process restart sees the consumed release.
    assert research_journal(policy()).read(path) == released
    for repeated in (event, release):
        repeated["observed_ms"] = 2003
        with pytest.raises(ValueError, match="RESEARCH-006"):
            journal.transact(path, lambda _, repeated=repeated: repeated)
        assert journal.read(path) == released


@pytest.mark.parametrize("released", [False, True])
def test_sealed_study_records_refusals_and_cannot_select_another_candidate(
    tmp_path: Path, released: bool
) -> None:
    path = tmp_path / "study.sqlite"
    event = _completed(path)
    journal = research_journal(policy())
    journal.transact(path, lambda _: event)
    if released:
        release: JSONObject = {**event, "action": "release_final", "observed_ms": 2002}
        journal.transact(path, lambda _: release)
    request = proposal()
    request["seed"] = 23
    refused = admit_proposal(path, policy(), request, 2003)
    assert (refused["status"], refused["reason"], refused["candidate"]) == (
        "refused",
        "study_sealed",
        "",
    )
    before = journal.read(path)
    alternative: JSONObject = {
        **event,
        "number": 1,
        "candidate": before[0]["candidate"],
        "observed_ms": 2004,
    }
    with pytest.raises(ValueError, match="RESEARCH-008"):
        journal.transact(path, lambda _: alternative)
    assert journal.read(path) == before


@pytest.mark.parametrize("active", ["admitted", "submitting", "running"])
def test_selection_waits_for_all_admitted_work(tmp_path: Path, active: str) -> None:
    path = tmp_path / "study.sqlite"
    event = _completed(path)
    journal = research_journal(policy())
    request = proposal()
    request["seed"] = 23
    row = admit_proposal(path, policy(), request, 2001)
    assert row["status"] == "admitted"
    if active != "admitted":
        journal.transact(path, lambda _: {"action": "submit", "number": 3, "observed_ms": 2001})
    if active == "running":
        journal.transact(
            path,
            lambda _: {
                "action": "scheduled",
                "number": 3,
                "observed_ms": 2001,
                "job_id": "slurm:3",
            },
        )
    before = journal.read(path)
    assert before[2]["status"] == active
    with pytest.raises(ValueError, match="RESEARCH-008"):
        journal.transact(path, lambda _: event)
    assert journal.read(path) == before


@pytest.mark.parametrize("action", ["select", "release_final"])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("partition", "holdout", "RESEARCH-009"),
        ("candidate", "a" * 64, "RESEARCH-009"),
        ("validation_payload_digest", "b" * 64, "RESEARCH-009"),
        ("final_manifest_digest", "malformed", "CAMPAIGN-001"),
    ],
)
def test_selection_and_release_refuse_changed_identity_atomically(
    tmp_path: Path, action: str, field: str, value: JSONValue, error: str
) -> None:
    path = tmp_path / "study.sqlite"
    event = _completed(path)
    journal = research_journal(policy())
    if action == "release_final":
        journal.transact(path, lambda _: event)
    changed: JSONObject = {**event, "action": action, field: value}
    before = journal.read(path)
    with pytest.raises(ValueError, match=error):
        journal.transact(path, lambda _: changed)
    assert journal.read(path) == before


def test_release_requires_prior_selection_and_its_exact_manifest(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    selection = _completed(path)
    journal = research_journal(policy())
    release: JSONObject = {**selection, "action": "release_final"}
    before = journal.read(path)
    with pytest.raises(ValueError, match="RESEARCH-006"):
        journal.transact(path, lambda _: release)
    assert journal.read(path) == before
    selected = journal.transact(path, lambda _: selection)
    release["final_manifest_digest"] = "a" * 64
    with pytest.raises(ValueError, match=r"RESEARCH-009.*manifest"):
        journal.transact(path, lambda _: release)
    assert journal.read(path) == selected


def test_attempt_codec_cannot_attach_final_evidence_to_an_unselected_row(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    _completed(path)
    row = encode_research_attempt(research_journal(policy()).read(path)[0])
    row["final_manifest_digest"] = "f" * 64
    with pytest.raises(ValueError, match="RESEARCH-004"):
        decode_research_attempt(row)
