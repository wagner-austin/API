"""Shared bot builders for the completion-event tests."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import (
    ActionKind,
    BotStateDataDict,
    InFlightActionDict,
    StateName,
    make_in_flight_action,
    make_initial_state_data,
)
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    decode_runtime_event_record,
)


def _decode_action_outcome_lines(jsonl: str) -> list[RuntimeEventRecordDict]:
    """Return every ``action_outcome`` event decoded from a JSONL artifact.

    Args:
        jsonl: Raw newline-delimited JSONL artifact body.

    Returns:
        Decoded :class:`RuntimeEventRecordDict` instances whose
        ``diagnostic_kind`` is ``action_outcome``. Other records
        (``STATE``, ``WIRE``, unrelated diagnostics) are filtered out
        so completion-site assertions are not coupled to unrelated
        emissions on the same path.
    """
    records: list[RuntimeEventRecordDict] = []
    for line in jsonl.strip().splitlines():
        raw: JSONObject = narrow_json_to_dict(load_json_str(line))
        record = decode_runtime_event_record(raw)
        if record["fields"].get("diagnostic_kind") == "action_outcome":
            records.append(record)
    return records


def _make_bot_with_in_flight(
    *,
    state: StateName,
    action_kind: ActionKind,
    target_x: int,
    target_y: int,
    started_ms: int,
) -> Bot:
    """Build a :class:`Bot` with a pre-configured in-flight action.

    Args:
        state: HFSM state name to install.
        action_kind: Kind for the in-flight action record.
        target_x: Target X coordinate stamped on the action record.
        target_y: Target Y coordinate stamped on the action record.
        started_ms: Dispatch timestamp the bot would have recorded.

    Returns:
        Bot instance whose ``_state_data`` carries the configured
        in-flight action and HFSM state.
    """
    bot = Bot("https://test.tankpit.com/", headless=True)
    base_data: BotStateDataDict = make_initial_state_data()
    action: InFlightActionDict = make_in_flight_action(
        action_kind,
        target_x,
        target_y,
        started_ms,
    )
    bot._state_data = BotStateDataDict(
        state=state,
        in_flight_action=action,
        fuel_threshold=base_data["fuel_threshold"],
    )
    return bot
