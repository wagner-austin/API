"""Forward scoring of real captured and settled campaign evidence, fully offline."""

from __future__ import annotations

import math
from contextlib import closing
from pathlib import Path
from statistics import stdev

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_list,
)
from platform_core.power_distributions import t_critical

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.campaign_records import Campaign, FrozenModel
from covenant_ml.markets.forward_plan import ForwardPlan, decode_forward_plan
from covenant_ml.markets.forward_scoring import (
    decode_forward_contract,
    encode_forward_contract,
    replay_capture,
    score_forward_plan,
)
from covenant_ml.markets.validation import read_document

from ._forward_fixtures import forward_campaign, plan_document
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def score(plan: ForwardPlan, root: Path, observed: str, url: str) -> JSONObject:
    with closing(_test_hooks.client_factory(5.0, None)) as client:

        def archive(path: Path) -> ResponseArchive:
            return ResponseArchive(path, url, client, _test_hooks.clock, _test_hooks.pause)

        report, sources = score_forward_plan(plan, "c" * 64, root, observed, archive, archive)
    assert root / "campaign.json" in sources
    return report


def days(campaign: Campaign) -> list[str]:
    return [plan["event_day"] for plan in campaign["plans"]]


def saved_model_probability(root: Path, name: str) -> float:
    capture = read_document(root / "reports" / (name + "-capture.json"))
    prediction = narrow_json_to_dict(require_list(capture, "predictions")[0])
    return require_float(prediction, "market_plus_forecast")


def test_declared_dates_score_from_archives_without_requests(
    api_server: ApiServer, tmp_path: Path
) -> None:
    outcomes = [
        ("binary", 1),
        ("binary", 0),
        ("binary", 1),
        ("pending", 1),
        ("scalar", 1),
        ("unsettled", 1),
    ]
    campaign, root, digest, observed = forward_campaign(api_server, tmp_path, outcomes)
    requests = len(api_server.requests)
    plan = decode_forward_plan(plan_document(campaign, digest, days(campaign)))
    report = score(plan, root, observed, api_server.url)
    assert len(api_server.requests) == requests
    assert (report["declared_dates"], report["scored_dates"], report["eligible_contracts"]) == (
        6,
        3,
        3,
    )
    dates = [narrow_json_to_dict(value) for value in require_list(report, "dates")]
    assert [date["event_day"] for date in dates] == days(campaign)
    dispositions = [
        narrow_json_to_dict(require_list(date, "plans")[0])["settlement"] for date in dates
    ]
    assert dispositions == ["complete"] * 5 + ["future"]
    assert [date["contracts"] for date in dates] == [1, 1, 1, 0, 0, 0]
    assert [date["paired_brier_difference"] for date in dates[3:]] == [None, None, None]
    differences: list[float] = []
    for frozen, date, (_, yes) in zip(campaign["plans"][:3], dates[:3], outcomes, strict=True):
        probability = saved_model_probability(root, frozen["name"])
        model, reference = (probability - yes) ** 2, (0.5 - yes) ** 2
        assert (date["model_brier"], date["reference_brier"]) == pytest.approx((model, reference))
        differences.append(model - reference)
    mean = sum(differences) / 3
    assert report["mean_paired_brier_difference"] == pytest.approx(mean)
    comparison = require_dict(report, "comparison")
    interval = require_dict(comparison, "interval")
    assert (interval["estimate"], interval["n_units"]) == (pytest.approx(mean), 3)
    power = require_dict(comparison, "power")
    expected_mde = t_critical(2, 0.05) * stdev(differences) / math.sqrt(3)
    assert (power["replicates"], power["minimum_detectable_effect"]) == (
        3,
        pytest.approx(expected_mde),
    )
    assert power["smallest_effect_of_interest"] == pytest.approx(0.005)
    probability = saved_model_probability(root, campaign["plans"][0]["name"])
    edge = max(probability, 1 - probability) - 0.64
    side = 1 if probability > 0.5 else 0
    pnl = [((yes == side) * 10000 - 6200) if edge > 0 else 0 for _, yes in outcomes[:3]]
    policy = require_dict(report, "policy")
    trades = 3 if edge > 0 else 0
    assert (policy["trades"], policy["pnl"], policy["extra_cent_pnl"]) == (
        trades,
        sum(pnl),
        sum(pnl) - 100 * trades,
    )
    assert require_dict(report, "criteria") == {
        "mean_below_zero": mean < 0,
        "upper_bound_below_zero": require_float(interval, "ci_high") < 0,
        "stressed_pnl_positive": sum(pnl) - 100 * trades > 0,
        "forward_lead": mean < 0
        and require_float(interval, "ci_high") < 0
        and sum(pnl) - 100 * trades > 0,
        "reason": "scored",
    }
    contracts = [
        decode_forward_contract(narrow_json_to_dict(value))
        for value in require_list(report, "contracts")
    ]
    assert [(row["yes"], row["bid"], row["ask"], row["reference"]) for row in contracts] == [
        (1, 4000, 6000, 0.5),
        (0, 4000, 6000, 0.5),
        (1, 4000, 6000, 0.5),
    ]
    assert [encode_forward_contract(row) for row in contracts] == require_list(report, "contracts")
    assert report["plan_sha256"] == "c" * 64


def test_too_few_scored_dates_fail_every_criterion(api_server: ApiServer, tmp_path: Path) -> None:
    campaign, root, digest, observed = forward_campaign(
        api_server, tmp_path, [("binary", 1), ("binary", 0)]
    )
    report = score(
        decode_forward_plan(plan_document(campaign, digest, days(campaign))),
        root,
        observed,
        api_server.url,
    )
    assert report["scored_dates"] == 2
    assert (report["mean_paired_brier_difference"], report["comparison"], report["policy"]) == (
        None,
        None,
        None,
    )
    assert require_dict(report, "criteria") == {
        "mean_below_zero": False,
        "upper_bound_below_zero": False,
        "stressed_pnl_positive": False,
        "forward_lead": False,
        "reason": "fewer than 3 scored dates",
    }


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("campaign", "name", "other-campaign"),
        ("campaign", "collector_implementation_sha256", "d" * 64),
        ("campaign", "protocol_sha256", "d" * 64),
        ("model", "report_sha256", "d" * 64),
        ("model", "features", ["market_logit"]),
        ("", "event_dates", ["2030-01-01"]),
    ],
)
def test_a_plan_the_campaign_does_not_answer_is_refused(
    api_server: ApiServer, tmp_path: Path, section: str, field: str, value: JSONValue
) -> None:
    campaign, root, digest, observed = forward_campaign(api_server, tmp_path, [("binary", 1)])
    document = plan_document(campaign, digest, days(campaign))
    target = require_dict(document, section) if section else document
    target[field] = value
    document["event_date_count"] = len(require_list(document, "event_dates"))
    with pytest.raises(ValueError, match="FORWARD-004"):
        score(decode_forward_plan(document), root, observed, api_server.url)


def test_a_capture_its_archives_do_not_reproduce_is_refused(
    api_server: ApiServer, tmp_path: Path
) -> None:
    campaign, root, _, _ = forward_campaign(api_server, tmp_path, [("binary", 1)])
    model = read_document(Path(campaign["models"][0]["path"]))
    fitted = require_dict(require_dict(model, "models"), "market_plus_forecast")
    fitted["intercept"] = require_float(fitted, "intercept") + 1.0
    changed = tmp_path / "changed-model.json"
    changed.write_text(dump_json_str(model), encoding="utf-8")
    drifted = Campaign(
        name=campaign["name"],
        created_at=campaign["created_at"],
        implementation_digest=campaign["implementation_digest"],
        protocol_path=campaign["protocol_path"],
        protocol_digest=campaign["protocol_digest"],
        models=tuple(
            FrozenModel(hour=entry["hour"], path=str(changed), digest=entry["digest"])
            for entry in campaign["models"]
        ),
        plans=campaign["plans"],
    )
    plan = campaign["plans"][0]
    with closing(_test_hooks.client_factory(5.0, None)) as client:

        def archive(path: Path) -> ResponseArchive:
            return ResponseArchive(
                path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
            )

        assert replay_capture(root, campaign, plan, archive, archive)[0]["schema_version"] == 1
        with pytest.raises(ValueError, match="FORWARD-005"):
            replay_capture(root, drifted, plan, archive, archive)


@pytest.mark.parametrize(
    "field,value",
    [("yes", 2), ("bid", 7000), ("ask", 10001), ("model", 1.5), ("reference", -0.1)],
)
def test_an_invalid_contract_is_refused(field: str, value: float) -> None:
    document: JSONObject = {
        "ticker": "T",
        "series": "KXHIGHTBOS",
        "event_day": "2026-11-03",
        "bid": 4000,
        "ask": 6000,
        "yes": 1,
        "model": 0.4,
        "reference": 0.5,
    }
    document[field] = value
    with pytest.raises(ValueError, match="FORWARD-006"):
        decode_forward_contract(document)
