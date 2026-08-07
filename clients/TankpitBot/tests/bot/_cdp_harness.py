"""Shared builders for the CDP test modules.

``test_cdp.py`` was 2,818 lines. It is now seven test modules over
these three builders: a page-client snapshot, a stubbed bot-action
result, and the health-gate snapshot variant.
"""

from __future__ import annotations

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.bot.states import (
    ActionKind,
    BotStateDataDict,
    StateName,
    make_in_flight_action,
)
from tankpit_bot.browser import get_current_time_ms


def _make_snapshot(*, map_visible: bool = False) -> PageClientSnapshotDict:
    """Return a healthy live-client snapshot for dispatch_command tests."""
    return PageClientSnapshotDict(
        timestamp_ms=1000,
        client_present=True,
        map_visible=map_visible,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=50,
        last_page_client_send_age_ms=100,
        last_bot_send_age_ms=100,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )


def _sba(
    sd: BotStateDataDict,
    state: StateName,
    kind: ActionKind,
    tx: int,
    ty: int,
    started_ms: int = -1,
) -> BotStateDataDict:
    """Build new BotStateDataDict with state and in-flight action.

    Args:
        sd: Current state data.
        state: New state name.
        kind: Action kind.
        tx: Target X.
        ty: Target Y.
        started_ms: Action start time. Defaults to current time
            so the action doesn't immediately stall. Pass 0 to
            test the "no timestamp" stall-guard path.
    """
    ts = get_current_time_ms() if started_ms < 0 else started_ms
    return BotStateDataDict(
        state=state,
        fuel_threshold=sd["fuel_threshold"],
        in_flight_action=make_in_flight_action(kind, tx, ty, ts),
    )


def _snapshot_for_health(
    *,
    client_present: bool = True,
    ws_ready_state: int | None = 1,
) -> PageClientSnapshotDict:
    """Build a snapshot with only the health-relevant fields parameterized."""
    return PageClientSnapshotDict(
        timestamp_ms=1000,
        client_present=client_present,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=50,
        last_page_client_send_age_ms=100,
        last_bot_send_age_ms=100,
        ws_ready_state=ws_ready_state,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )
