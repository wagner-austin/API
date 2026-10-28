"""Admission, permanent trial budgets and raw refused-request preservation."""

from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue, require_dict

from covenant_ml.markets.research_journal import (
    admit_proposal,
    decode_research_attempt,
    encode_research_attempt,
    research_journal,
    research_policy_digest,
)
from covenant_ml.markets.research_records import (
    candidate_digest,
    decode_research_policy,
    decode_research_proposal,
    encode_research_policy,
    encode_research_proposal,
)

from ._research_fixtures import policy, proposal


def test_codecs_and_immutable_policy_identity(tmp_path: Path) -> None:
    frozen = policy()
    assert decode_research_policy(encode_research_policy(frozen)) == frozen
    request = decode_research_proposal(proposal())
    assert encode_research_proposal(request) == proposal()
    reversed_features = proposal()
    reversed_features["features"] = ["volume_60", "return_1"]
    original = decode_research_proposal(reversed_features)
    reversed_features["features"] = ["return_1", "volume_60"]
    assert candidate_digest(original) == candidate_digest(
        decode_research_proposal(reversed_features)
    )
    assert len(research_policy_digest(frozen)) == 64
    changed = encode_research_policy(frozen)
    changed["max_candidates"] = 1
    other = decode_research_policy(changed)
    assert research_policy_digest(other) != research_policy_digest(frozen)
    path = tmp_path / "study.sqlite"
    journal = research_journal(frozen)
    assert journal.initialize(path) == ()
    admitted = admit_proposal(path, frozen, proposal(), 1000)
    assert admitted["status"] == "admitted"
    assert decode_research_attempt(encode_research_attempt(admitted)) == admitted
    with pytest.raises(ValueError, match="JOURNAL-001"):
        research_journal(other).read(path)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("protocol_digest", "f" * 64, "protocol_changed"),
        ("partition", "final", "holdout_forbidden"),
        ("features", ["future_return"], "features_forbidden"),
        ("seed", 999, "seed_forbidden"),
        ("seconds", 61, "time_budget"),
        ("parent", "unknown", "parent_unknown"),
    ],
)
def test_policy_refusals_are_durable(
    tmp_path: Path, field: str, value: JSONValue, reason: str
) -> None:
    path = tmp_path / "study.sqlite"
    frozen = policy()
    research_journal(frozen).initialize(path)
    request = proposal()
    request[field] = value
    row = admit_proposal(path, frozen, request, 1000)
    assert (row["status"], row["reason"], row["candidate"]) == ("refused", reason, "")
    assert research_journal(frozen).read(path) == (row,)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("n_estimators", 0, "model_budget"),
        ("n_estimators", 201, "model_budget"),
        ("num_leaves", 1, "model_budget"),
        ("num_leaves", 32, "model_budget"),
        ("max_depth", 0, "model_budget"),
        ("max_depth", 17, "model_budget"),
        ("max_bins", 1, "model_budget"),
        ("max_bins", 256, "model_budget"),
        ("n_jobs", 2, "fixed_configuration_changed"),
        ("repeats", 2, "fixed_configuration_changed"),
        ("warmups", 1, "fixed_configuration_changed"),
        ("learning_rate", 0.1, "fixed_configuration_changed"),
    ],
)
def test_model_limits(tmp_path: Path, field: str, value: JSONValue, reason: str) -> None:
    path = tmp_path / "study.sqlite"
    research_journal(policy()).initialize(path)
    request = proposal()
    require_dict(request, "config")[field] = value
    row = admit_proposal(path, policy(), request, 1000)
    assert (row["status"], row["reason"]) == ("refused", reason)


def test_failures_and_renaming_do_not_reset_budget(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    journal = research_journal(policy())
    journal.initialize(path)
    first = admit_proposal(path, policy(), proposal(), 1000)
    journal.transact(
        path,
        lambda _: {
            "action": "failed",
            "number": 1,
            "observed_ms": 1001,
            "reason": "scheduler refused",
        },
    )
    renamed = proposal()
    renamed["hypothesis"] = "A different title cannot buy another trial."
    duplicate = admit_proposal(path, policy(), renamed, 1002)
    assert duplicate["reason"] == "candidate_seed_consumed"
    renamed["seed"] = 23
    assert admit_proposal(path, policy(), renamed, 1003)["status"] == "admitted"
    second = proposal()
    second["parent"] = first["candidate"]
    require_dict(second, "config")["n_estimators"] = 100
    assert admit_proposal(path, policy(), second, 1004)["status"] == "admitted"
    third = proposal()
    require_dict(third, "config")["n_estimators"] = 50
    assert admit_proposal(path, policy(), third, 1005)["reason"] == "candidate_budget"
    history = journal.read(path)
    assert tuple(row["status"] for row in history) == (
        "failed",
        "refused",
        "admitted",
        "admitted",
        "refused",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extra", "command"),
        ("features", []),
        ("features", ["a", "a"]),
        ("seed", "wrong"),
        ("protocol_digest", "bad"),
    ],
)
def test_malformed_input_is_preserved(tmp_path: Path, field: str, value: JSONValue) -> None:
    path = tmp_path / "study.sqlite"
    journal = research_journal(policy())
    journal.initialize(path)
    request = proposal()
    request[field] = value
    with pytest.raises((ValueError, TypeError)):
        admit_proposal(path, policy(), request, 1000)
    rows = journal.read(path)
    assert len(rows) == 1
    assert (rows[0]["status"], rows[0]["reason"]) == ("refused", "not_admitted")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extra", 1),
        ("seeds", []),
        ("seeds", [11, 11]),
        ("features", []),
        ("features", ["a", "a"]),
        ("max_candidates", 21),
    ],
)
def test_invalid_frozen_policy(field: str, value: JSONValue) -> None:
    document = encode_research_policy(policy())
    document[field] = value
    with pytest.raises(ValueError):
        decode_research_policy(document)


def test_invalid_nested_config_and_attempt_codec(tmp_path: Path) -> None:
    request = proposal()
    require_dict(request, "config")["shell"] = "command"
    with pytest.raises(ValueError, match="RESEARCH-003"):
        decode_research_proposal(request)
    path = tmp_path / "study.sqlite"
    research_journal(policy()).initialize(path)
    row = encode_research_attempt(admit_proposal(path, policy(), proposal(), 1000))
    row["extra"] = 1
    with pytest.raises(ValueError, match="RESEARCH-004"):
        decode_research_attempt(row)
    del row["extra"]
    row["status"] = "invented"
    with pytest.raises(ValueError, match="RESEARCH-004"):
        decode_research_attempt(row)
