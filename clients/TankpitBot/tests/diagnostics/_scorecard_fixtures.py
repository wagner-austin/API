"""Shared record builders for the scorecard test suites.

One home for the helpers the accumulator, render, and kill-attribution
suites all need (consolidated 2026-08-20 — three copies had grown, one
per split file, and a fixture forked three ways drifts three ways).
"""

from __future__ import annotations

from tankpit_bot.diagnostics.session_scorecard_accumulator import route_scorecard_record
from tankpit_bot.diagnostics.session_scorecard_types import (
    ScorecardAccumulatorDict,
    new_scorecard_accumulator,
)
from tankpit_bot.runtime_records import RuntimeEventRecordDict


def _record(
    *,
    channel: str,
    message: str = "",
    timestamp: str = "2026-06-12T06:25:00",
    fields: dict[str, str | int | float | bool] | None = None,
) -> RuntimeEventRecordDict:
    """Build a runtime event record for routing tests.

    Args:
        channel: Event channel name.
        message: Event message text.
        timestamp: ISO timestamp.
        fields: Structured payload fields.

    Returns:
        Runtime event record.
    """
    return RuntimeEventRecordDict(
        timestamp=timestamp,
        level="INFO",
        logger="tankpit_bot.runtime.events",
        mode="bot",
        channel=channel,
        message=message,
        fields=fields if fields is not None else {},
    )


def _routed(records: list[RuntimeEventRecordDict]) -> ScorecardAccumulatorDict:
    """Route every record into a fresh accumulator.

    Args:
        records: Records in stream order.

    Returns:
        Routed accumulator.
    """
    accumulator = new_scorecard_accumulator()
    for record in records:
        route_scorecard_record(record, accumulator)
    return accumulator


def _fuel_sample_record(
    *,
    fuel: int,
    timestamp: str,
    bot_state: str = "HUNT/ENGAGE",
    in_flight: str = "shoot",
) -> RuntimeEventRecordDict:
    """Build a context-stamped ``self_alignment_sample`` record.

    Args:
        fuel: ``belief_fuel`` value.
        timestamp: ISO timestamp.
        bot_state: Ambient bot-state context.
        in_flight: Ambient in-flight action kind.

    Returns:
        Runtime event record.
    """
    return _record(
        channel="DIAGNOSTIC",
        timestamp=timestamp,
        fields={
            "diagnostic_kind": "self_alignment_sample",
            "belief_fuel": fuel,
            "bot_state": bot_state,
            "in_flight_action_kind": in_flight,
        },
    )
