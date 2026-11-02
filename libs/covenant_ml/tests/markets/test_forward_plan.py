"""The committed forward plan's pinned bytes, its codec and every refusal."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict

from covenant_ml.markets.forecast_policy import QUOTED_FEE_BPS, QUOTED_HURDLE, QUOTED_STRESS
from covenant_ml.markets.forward_plan import (
    decode_forward_plan,
    encode_forward_plan,
    load_forward_plan,
)
from covenant_ml.markets.validation import read_document

from ._forward_fixtures import FROZEN_PLAN, FROZEN_PLAN_SHA256, write_plan


def test_committed_plan_is_the_board_recorded_plan() -> None:
    plan = load_forward_plan(FROZEN_PLAN, FROZEN_PLAN_SHA256)
    assert encode_forward_plan(plan) == read_document(FROZEN_PLAN)
    first_dates: tuple[str, ...] = ("2026-11-03", "2026-11-04")
    assert plan["event_dates"][:2] == first_dates
    assert (len(plan["event_dates"]), plan["event_dates"][-1]) == (40, "2026-12-12")
    assert (plan["model"]["arm"], plan["reference_arm"], plan["model"]["forecast_hour"]) == (
        "market_plus_forecast",
        "raw_market",
        12,
    )
    assert (plan["fee_coefficient_bps"], plan["edge_hurdle"], plan["stress_per_trade"]) == (
        700,
        200,
        100,
    )
    assert (QUOTED_FEE_BPS, QUOTED_HURDLE, QUOTED_STRESS) == (700, 200, 100)
    assert plan["campaign"]["manifest_sha256"] == (
        "f57a191b3a7a1f289d01e5b9826a9f9d49e6222133a06382e2a92330a122cc85"
    )


def test_changed_bytes_are_refused(tmp_path: Path) -> None:
    copy = tmp_path / "plan.json"
    copy.write_bytes(FROZEN_PLAN.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="FORWARD-001"):
        load_forward_plan(copy, FROZEN_PLAN_SHA256)


def test_a_rewritten_plan_loads_under_its_own_digest(tmp_path: Path) -> None:
    document = read_document(FROZEN_PLAN)
    document["name"] = "renamed"
    path = tmp_path / "plan.json"
    assert load_forward_plan(path, write_plan(path, document))["name"] == "renamed"


def _changed(field: str, value: JSONValue, inner: str = "") -> JSONObject:
    document = read_document(FROZEN_PLAN)
    if inner:
        require_dict(document, inner)[field] = value
    else:
        document[field] = value
    return document


@pytest.mark.parametrize(
    "document",
    [
        _changed("schema_version", 2),
        _changed("reference_arm", "market_plus_forecast"),
        _changed("reference_arm", "other"),
        _changed("series", []),
        _changed("series", ["KXHIGHTBOS", "KXHIGHTBOS"]),
        _changed("series", ["KXHIGHNY"]),
        _changed("event_dates", []),
        _changed("event_dates", ["2026-11-04", "2026-11-03"]),
        _changed("event_dates", ["2026-11-03", "2026-11-03"]),
        _changed("event_date_count", 39),
        _changed("confidence", 1.0),
        _changed("alpha", 0.0),
        _changed("smallest_effect_of_interest", 0.0),
        _changed("arm", "other", "model"),
        _changed("forecast_hour", 6, "model"),
        _changed("features", [], "model"),
        _changed("report_sha256", "A" * 64, "model"),
        _changed("report_sha256", "a" * 63, "model"),
        _changed("manifest_sha256", "g" * 64, "campaign"),
    ],
)
def test_invalid_declarations_are_refused(document: JSONObject) -> None:
    with pytest.raises(ValueError, match="FORWARD-002"):
        decode_forward_plan(document)


@pytest.mark.parametrize(
    "field,value",
    [("fee_coefficient_bps", 701), ("edge_hurdle", 100), ("stress_per_trade", 0)],
)
def test_costs_the_policy_does_not_implement_are_refused(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="FORWARD-003"):
        decode_forward_plan(_changed(field, value))


def test_noncanonical_dates_and_empty_text_are_refused() -> None:
    with pytest.raises(ValueError, match="MARKET-004"):
        decode_forward_plan(_changed("event_dates", ["20261103"]))
    with pytest.raises(ValueError, match="MARKET-004"):
        decode_forward_plan(_changed("frozen_at", "20261102"))
    with pytest.raises(ValueError, match=r"MARKET-003\] scope must not be blank"):
        decode_forward_plan(_changed("scope", " "))
