"""Scheduler lineage and failures using a real on-disk research journal."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, require_dict

from covenant_ml.markets.research_journal import admit_proposal, research_journal

from ._research_fixtures import completion, policy, proposal


def _running(path: Path) -> None:
    journal = research_journal(policy())
    journal.initialize(path)
    admit_proposal(path, policy(), proposal(), 1000)
    journal.transact(path, lambda _: {"action": "submit", "number": 1, "observed_ms": 1000})
    journal.transact(
        path,
        lambda _: {
            "action": "scheduled",
            "number": 1,
            "observed_ms": 1001,
            "job_id": "slurm:123",
        },
    )


def test_negative_result_is_preserved_and_final_is_terminal(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    _running(path)
    journal = research_journal(policy())
    event = completion()
    final = journal.transact(path, lambda _: event)
    assert final[0]["status"] == "complete"
    assert final[0]["record"] == dump_json_str(event["record"])
    assert journal.read(path) == final
    with pytest.raises(ValueError, match="RESEARCH-006"):
        journal.transact(path, lambda _: event)
    assert journal.read(path) == final


@pytest.mark.parametrize(
    "field",
    [
        "job_id",
        "dataset_digest",
        "code_digest",
        "image_digest",
        "label",
        "experiment",
        "payload_digest",
    ],
)
def test_completion_must_match_pinned_lineage(tmp_path: Path, field: str) -> None:
    path = tmp_path / "study.sqlite"
    _running(path)
    event = completion()
    record = require_dict(event, "record")
    if field == "image_digest":
        require_dict(record, "fingerprint")[field] = "f" * 64
    elif field in {"label", "experiment"}:
        record[field] = "wrong"
    elif field == "payload_digest":
        record[field] = ""
    else:
        event[field] = "f" * 64
    journal = research_journal(policy())
    before = journal.read(path)
    with pytest.raises(ValueError, match="RESEARCH-007"):
        journal.transact(path, lambda _: event)
    assert journal.read(path) == before


@pytest.mark.parametrize("action", ["failed", "timeout", "interrupted"])
def test_terminal_failures_retain_job_and_consume_trial(tmp_path: Path, action: str) -> None:
    path = tmp_path / "study.sqlite"
    _running(path)
    journal = research_journal(policy())
    final = journal.transact(
        path,
        lambda _: {
            "action": action,
            "number": 1,
            "observed_ms": 2000,
            "reason": "worker stopped",
        },
    )
    assert (final[0]["status"], final[0]["job_id"], final[0]["reason"]) == (
        action,
        "slurm:123",
        "worker stopped",
    )
    assert admit_proposal(path, policy(), proposal(), 2001)["reason"] == "candidate_seed_consumed"


@pytest.mark.parametrize(
    "event",
    [
        {"action": "submit", "number": 0, "observed_ms": 1000},
        {"action": "submit", "number": 2, "observed_ms": 1000},
        {"action": "submit", "number": 1, "observed_ms": 999},
        {"action": "scheduled", "number": 1, "observed_ms": 1000, "job_id": "123"},
        {"action": "complete", "number": 1, "observed_ms": 1000},
        {"action": "invented", "number": 1, "observed_ms": 1000},
    ],
)
def test_invalid_transitions_roll_back(tmp_path: Path, event: JSONObject) -> None:
    path = tmp_path / "study.sqlite"
    journal = research_journal(policy())
    journal.initialize(path)
    original = admit_proposal(path, policy(), proposal(), 1000)
    with pytest.raises(ValueError, match=r"RESEARCH-00[56]"):
        journal.transact(path, lambda _: event)
    assert journal.read(path) == (original,)


def test_duplicate_scheduler_id_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    _running(path)
    request = proposal()
    request["seed"] = 23
    admit_proposal(path, policy(), request, 2000)
    journal = research_journal(policy())
    journal.transact(path, lambda _: {"action": "submit", "number": 2, "observed_ms": 2000})
    before = journal.read(path)
    with pytest.raises(ValueError, match="RESEARCH-006"):
        journal.transact(
            path,
            lambda _: {
                "action": "scheduled",
                "number": 2,
                "observed_ms": 2001,
                "job_id": "slurm:123",
            },
        )
    assert journal.read(path) == before


def test_real_writer_lock_prevents_unrecorded_admission(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    journal = research_journal(policy())
    journal.initialize(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            admit_proposal(path, policy(), proposal(), 1000)
    assert journal.read(path) == ()
    assert admit_proposal(path, policy(), proposal(), 1001)["status"] == "admitted"
