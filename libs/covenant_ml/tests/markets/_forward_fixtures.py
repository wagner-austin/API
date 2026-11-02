"""A real multi-day campaign and a forward plan bound to it, for forward scoring tests."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, require_dict

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.campaign_records import Campaign
from covenant_ml.markets.forecast_model import WEATHER_FEATURES
from covenant_ml.markets.validation import read_document

from ._campaign_evaluation_fixtures import capture_plan, settle_plan
from ._campaign_fixtures import single_campaign
from ._server import ApiServer

#: The plan frozen on the board for task 269fd5c0, as committed.
FROZEN_PLAN = Path(__file__).resolve().parents[2] / "docs" / "forward_weather_plan_20261103.json"
#: Its SHA-256 as recorded on that task's thread before its first contract settled.
FROZEN_PLAN_SHA256 = "bfdb77a0694048bc950df34029f6e82f49689acfe0c42a69108a481b8a9bd33c"


@pytest.fixture()
def restore_clock() -> Generator[None, None, None]:
    """Put the real clock back after a test that fixes it."""
    original = _test_hooks.clock
    yield
    _test_hooks.clock = original


def fixed_clock(instant: str) -> Callable[[], str]:
    """A clock that always reads one instant."""

    def read() -> str:
        return instant

    return read


def forward_campaign(
    server: ApiServer, destination: Path, outcomes: Sequence[tuple[str, int]]
) -> tuple[Campaign, Path, str, str]:
    """Capture one plan per outcome on consecutive days and settle each one named.

    Each capture runs with the clock fixed at its own plan's window start, since
    every later plan's window lies days ahead; a test calling this requests
    ``restore_clock``.

    Args:
        server: Local origin serving native exchange and forecast documents.
        destination: Empty directory.
        outcomes: Per day, a settlement kind (``binary``, ``pending``, ``scalar``,
            ``changed``) and the payout, or kind ``unsettled`` for no settlement.

    Returns:
        Campaign, its directory, manifest digest and the latest settlement instant.
    """
    campaign, root, digest = single_campaign(destination, len(outcomes))
    observed = campaign["created_at"]
    for plan, (kind, yes) in zip(campaign["plans"], outcomes, strict=True):
        _test_hooks.clock = fixed_clock(plan["window_start"])
        ticker = capture_plan(server, root, campaign, plan)
        if kind != "unsettled":
            observed = settle_plan(server, root, plan, ticker, digest, kind, yes)
    return campaign, root, digest, observed


def plan_document(campaign: Campaign, digest: str, days: Sequence[str]) -> JSONObject:
    """The committed frozen plan, rebound to a test campaign and its event days."""
    document = read_document(FROZEN_PLAN)
    document["campaign"] = {
        "name": campaign["name"],
        "manifest_sha256": digest,
        "collector_implementation_sha256": campaign["implementation_digest"],
        "protocol_sha256": campaign["protocol_digest"],
    }
    features: list[JSONValue] = list(WEATHER_FEATURES)
    dates: list[JSONValue] = list(days)
    document["model"] = {
        **require_dict(document, "model"),
        "forecast_hour": 0,
        "report_sha256": campaign["models"][0]["digest"],
        "features": features,
    }
    document["series"] = ["KXHIGHTBOS"]
    document["event_dates"] = dates
    document["event_date_count"] = len(days)
    return document


def write_plan(path: Path, document: JSONObject) -> str:
    """Write a plan file and return the SHA-256 of its exact bytes."""
    path.write_text(dump_json_str(document), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


__all__ = [
    "FROZEN_PLAN",
    "FROZEN_PLAN_SHA256",
    "fixed_clock",
    "forward_campaign",
    "plan_document",
    "restore_clock",
    "write_plan",
]
