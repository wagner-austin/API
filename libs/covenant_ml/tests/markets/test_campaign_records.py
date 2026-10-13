"""Manifest, schedule and attempt invariants including DST and frozen input refusal."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict, require_list

from covenant_ml.markets.campaign_execution import load_campaign
from covenant_ml.markets.campaign_planning import persist_campaign, plan_campaign
from covenant_ml.markets.campaign_records import (
    Attempt,
    AttemptState,
    decode_attempt,
    decode_campaign,
    decode_frozen_model,
    encode_attempt,
    encode_campaign,
    encode_frozen_model,
)
from covenant_ml.markets.provenance import implementation_digest

from ._campaign_fixtures import single_campaign


def test_complete_manifest_and_daylight_saving_transition(tmp_path: Path) -> None:
    protocol, model = tmp_path / "protocol.json", tmp_path / "model.json"
    protocol.write_text('{"purpose":"fixed prospective evaluation"}', encoding="utf-8")
    model.write_text("{}", encoding="utf-8")
    campaign = plan_campaign(
        "autumn",
        date(2026, 11, 1),
        date(2026, 11, 2),
        "2026-10-01T00:00:00Z",
        protocol,
        (model, model),
        implementation_digest(),
    )
    assert decode_campaign(encode_campaign(campaign)) == campaign
    assert len(campaign["plans"]) == 12
    assert {p["window_start"] for p in campaign["plans"]} == {
        "2026-10-31T20:00:00+00:00",
        "2026-11-01T21:00:00+00:00",
    }
    assert {p["window_end"] for p in campaign["plans"]} == {
        "2026-10-31T20:15:00+00:00",
        "2026-11-01T21:15:00+00:00",
    }
    assert {(p["series"], p["forecast_hour"]) for p in campaign["plans"]} == {
        (city, hour) for city in ("KXHIGHTBOS", "KXHIGHPHIL", "KXHIGHTDC") for hour in (0, 12)
    }
    root = tmp_path / "frozen"
    digest = persist_campaign(campaign, root)
    assert load_campaign(root, digest) == campaign
    with pytest.raises(FileExistsError):
        persist_campaign(campaign, root)


@pytest.mark.parametrize("last", [date(2026, 10, 31), date(2028, 1, 1)])
def test_invalid_date_range(tmp_path: Path, last: date) -> None:
    with pytest.raises(ValueError, match="CAMPAIGN-006"):
        plan_campaign(
            "bad",
            date(2026, 11, 1),
            last,
            "2026-10-01T00:00:00Z",
            tmp_path,
            (tmp_path, tmp_path),
            "a" * 64,
        )


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("hour", 6, "003"),
        ("path", "relative.json", "002"),
        ("digest", "z" * 64, "001"),
        ("digest", "abc", "001"),
    ],
)
def test_invalid_frozen_model(tmp_path: Path, key: str, value: JSONValue, code: str) -> None:
    document: JSONObject = {"hour": 0, "path": str(tmp_path / "model.json"), "digest": "a" * 64}
    decoded = decode_frozen_model(document)
    assert decode_frozen_model(encode_frozen_model(decoded)) == decoded
    changed = encode_frozen_model(decoded)
    changed[key] = value
    with pytest.raises(ValueError, match="CAMPAIGN-" + code):
        decode_frozen_model(changed)


@pytest.mark.parametrize(
    "change", ["models", "empty", "duplicate", "name", "hour", "model", "protocol", "past"]
)
def test_invalid_campaign(tmp_path: Path, change: str) -> None:
    campaign, _, _ = single_campaign(tmp_path)
    document = encode_campaign(campaign)
    plans = require_list(document, "plans")
    plan = narrow_json_to_dict(plans[0])
    if change == "models":
        document["models"] = []
    elif change == "empty":
        document["plans"] = []
    elif change == "duplicate":
        plans.append(plans[0])
    elif change == "name":
        plan["name"] = "../escape"
    elif change == "hour":
        plan["forecast_hour"] = 6
        day = campaign["plans"][0]["window_start"][:10]
        plan["window_start"] = day + "T19:00:00Z"
        plan["window_end"] = day + "T19:15:00Z"
    elif change == "model":
        plan["model_digest"] = "c" * 64
    elif change == "protocol":
        plan["protocol_digest"] = "c" * 64
    else:
        document["created_at"] = plan["window_start"]
    with pytest.raises(ValueError, match="CAMPAIGN-004"):
        decode_campaign(document)


@pytest.mark.parametrize("state", tuple(AttemptState))
def test_attempt_roundtrips(state: AttemptState) -> None:
    row = Attempt(
        name="run",
        operation="capture",
        state=state,
        observed_at="2026-10-13T00:00:00Z",
        returncode=0
        if state is AttemptState.COMPLETE
        else 7
        if state is AttemptState.FAILED
        else None,
        report_digest="a" * 64 if state is AttemptState.COMPLETE else None,
    )
    assert decode_attempt(encode_attempt(row)) == row


@pytest.mark.parametrize(
    "state,code,digest,operation",
    [
        ("complete", None, "a" * 64, "capture"),
        ("complete", 0, None, "capture"),
        ("failed", 0, None, "capture"),
        ("failed", None, None, "capture"),
        ("failed", 7, "a" * 64, "capture"),
        ("missed", 0, None, "capture"),
        ("missed", None, "a" * 64, "capture"),
        ("missed", None, None, "unknown"),
    ],
)
def test_inconsistent_attempt(
    state: str, code: int | None, digest: str | None, operation: str
) -> None:
    with pytest.raises(ValueError, match="CAMPAIGN-005"):
        decode_attempt(
            {
                "name": "run",
                "state": state,
                "operation": operation,
                "observed_at": "2026-10-13T00:00:00Z",
                "returncode": code,
                "report_digest": digest,
            }
        )
