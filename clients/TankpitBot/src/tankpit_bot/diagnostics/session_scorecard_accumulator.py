"""Scorecard accumulation: fold raw event records into one accumulator.

The ingest half of the scorecard concern -- :func:`route_scorecard_record`
and its per-channel routers decide what each runtime event contributes.
The accumulator's shape and factory are
:mod:`tankpit_bot.diagnostics.session_scorecard_types`; the
DIAGNOSTIC-kind arms are
:mod:`tankpit_bot.diagnostics.session_scorecard_routes`; distillation
into a :class:`SessionScorecardDict` is
:mod:`tankpit_bot.diagnostics.session_scorecard`; rendering is
:mod:`tankpit_bot.diagnostics.session_scorecard_render`.
"""

from __future__ import annotations

import re

from tankpit_bot.diagnostics.session_scorecard_routes import route_scorecard_diagnostic
from tankpit_bot.diagnostics.session_scorecard_types import (
    ScorecardAccumulatorDict,
    optional_str_field,
)
from tankpit_bot.runtime_records import RuntimeEventRecordDict

# WORLD-channel fuel transition receipts, e.g. "Fuel: 1090 -> 823 (-267)".
_WORLD_FUEL_PATTERN = re.compile(r"^Fuel: (-?\d+) -> (-?\d+) ")


def route_scorecard_record(
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Route one event record into the scorecard accumulator.

    Every record contributes to the session duration bounds; only the
    scorecard-relevant channels and diagnostic kinds populate the
    other buckets.

    Args:
        record: Decoded runtime event record.
        accumulator: Scorecard accumulator to update in place.
    """
    if not accumulator["first_timestamp"]:
        accumulator["first_timestamp"] = record["timestamp"]
    accumulator["last_timestamp"] = record["timestamp"]
    channel = record["channel"]
    if channel == "STATE":
        accumulator["state_transitions"].append((record["timestamp"], record["message"]))
        return
    if channel == "WIRE" and record["message"].startswith("shoot("):
        accumulator["shots"] += 1
        return
    if channel == "WORLD":
        _route_world_fuel_receipt(record, accumulator)
        return
    if channel != "DIAGNOSTIC":
        return
    route_scorecard_diagnostic(record, accumulator)


def _route_world_fuel_receipt(
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Attribute an in-flight teleport fuel debit to its bot state.

    WORLD-channel fuel transitions are the per-receipt view of the
    fuel book; a debit billed while ``in_flight_action_kind`` is
    ``teleport`` is teleport spend, attributed to the ambient
    ``bot_state``. Measured on run 20260729-105325: 15592 across 104
    receipts, inside the ledger's 11993..19290 feasibility bound
    (sample-to-sample fuel deltas undercounted at 10972 because
    pickup credits mask debits inside one tick window).

    Args:
        record: Decoded WORLD-channel event record.
        accumulator: Scorecard accumulator to update in place.
    """
    if optional_str_field(record["fields"], "in_flight_action_kind", "none") != "teleport":
        return
    match = _WORLD_FUEL_PATTERN.match(record["message"])
    if match is None:
        return
    delta = int(match.group(2)) - int(match.group(1))
    if delta >= 0:
        return
    state = optional_str_field(record["fields"], "bot_state", "")
    accumulator["teleport_spend_fuel"][state] = (
        accumulator["teleport_spend_fuel"].get(state, 0) - delta
    )
    accumulator["teleport_spend_drops"][state] = (
        accumulator["teleport_spend_drops"].get(state, 0) + 1
    )


__all__ = [
    "route_scorecard_record",
]
