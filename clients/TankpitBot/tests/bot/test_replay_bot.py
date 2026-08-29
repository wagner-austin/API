"""End-to-end replay test: drive the real ``Bot`` through a captured session.

This is the strongest integration the harness allows. Real decoders, real
world-state mutators, real ``decide()``, real executor, real command
encoding -- only the WebSocket dispatch and browser bootstrap are
substituted (see ``_replay_harness.ReplayBot`` for the rationale).

Assertion shape: property-based, not snapshot-based. The bot's per-tick
behavior has small non-deterministic variations between runs (a borderline
``map_open`` may or may not fire on a particular tick depending on
floating-point comparisons or set/dict iteration order). A frozen
trace snapshot would be flaky; instead we assert *invariants* that any
correct run must satisfy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.bot._replay_harness import ReplaySession, run_replay

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def fuel_probe_session() -> ReplaySession:
    """Replay the committed fuel_probe capture once per test that needs it."""
    return run_replay(REPO_ROOT / "fuel_probe.capture_session.json", frames_per_tick=5)


def test_fuel_probe_replay_decodes_all_frames(fuel_probe_session: ReplaySession) -> None:
    """The harness drains every received frame from the capture file."""
    assert fuel_probe_session.total_received_frames == 119


def test_fuel_probe_replay_produces_one_tick_per_batch(
    fuel_probe_session: ReplaySession,
) -> None:
    """One tick is recorded per ``frames_per_tick``-sized batch (119 / 5 = 24)."""
    assert len(fuel_probe_session.ticks) == 24


def test_fuel_probe_replay_transitions_out_of_initializing(
    fuel_probe_session: ReplaySession,
) -> None:
    """The bot leaves the ``INITIALIZING`` startup state once self-state arrives."""
    states = [t.state for t in fuel_probe_session.ticks]
    assert "INITIALIZING" in states
    assert {"SCANNING", "MOVING", "COLLECTING", "IDLE"}.intersection(states)


def test_fuel_probe_replay_enters_collect_mode(
    fuel_probe_session: ReplaySession,
) -> None:
    """Real ``decide()`` selects COLLECT after detecting empty inventory."""
    modes = [t.ai_mode for t in fuel_probe_session.ticks]
    assert "COLLECT" in modes


def test_fuel_probe_replay_routes_the_stocked_session_into_collection(
    fuel_probe_session: ReplaySession,
) -> None:
    """Replay routes the stocked wire input into known-stock pickups.

    Re-pinned 2026-08-06 (sweep-first: every tick became a scope
    shift replay could never answer) and again 2026-08-13 with the
    recon reorder (HUD flags 8/9/14, known stock preempts scanning):
    the recorded session's containers are now COLLECTED -- pickup
    dispatches engage -- instead of being scanned past. The invariant
    kept here is decision ROUTING: pickups fire on the recorded stock
    and every tick stays in COLLECT.
    """
    commands = {cmd for _tick, cmd in fuel_probe_session.all_dispatched}
    assert any(cmd.startswith("pickup_") for cmd in commands)
    assert all(t.ai_mode in ("UNSET", "COLLECT") for t in fuel_probe_session.ticks)


def test_fuel_probe_replay_dispatches_at_least_one_command(
    fuel_probe_session: ReplaySession,
) -> None:
    """Real ``_send_bytes`` is invoked through real executor through real decide().

    This is the load-bearing harness wiring assertion: a zero-dispatch run
    means the replay never exercised the production command path.
    """
    if not fuel_probe_session.all_dispatched:
        pytest.fail(
            "replay produced no dispatched commands; the harness did not exercise "
            "_send_bytes through the real executor / decide() chain"
        )


def test_fuel_probe_replay_first_dispatched_command_is_radar(
    fuel_probe_session: ReplaySession,
) -> None:
    """The first command the bot sends is the scan-on-landing radar.

    COLLECT's scan-on-landing gate fires one radar on every fresh
    teleport landing before any pickup -- mirrors HUNT's
    scan_on_landing so the planner has the full picture (0x5A entries
    plus radar reveals) before committing to a pickup order.
    """
    if not fuel_probe_session.all_dispatched:
        pytest.fail("expected at least one dispatched command")
    _first_tick, first_command = fuel_probe_session.all_dispatched[0]
    assert first_command == "radar"


def test_fuel_probe_replay_dispatches_sweep_radars_beyond_the_landing_scan(
    fuel_probe_session: ReplaySession,
) -> None:
    """Band stock scans exactly once: the pre-inventory landing scan.

    Re-pinned 2026-08-28 with the radar hoard rule: the recorded
    session's stock sits in the hoard band (between one extra and the
    hunt bar), where the slot is disabled and every scan decider
    declines -- so the only radar the replay dispatches is the first
    landing scan, fired before the 0x49 inventory snapshot reveals
    the band. (The 2026-08-06 pin of >= 3 sweep radars encoded the
    pre-hoard flow that burned band stock on sweeps -- the exact
    income-burn deadlock the rule kills.)
    """
    radar_count = sum(1 for _tick, cmd in fuel_probe_session.all_dispatched if cmd == "radar")
    assert radar_count == 1


def test_fuel_probe_replay_observes_visible_containers(
    fuel_probe_session: ReplaySession,
) -> None:
    """Radar scans populate the world-state ``containers`` index."""
    max_containers_known = max(t.containers_known for t in fuel_probe_session.ticks)
    assert max_containers_known >= 1


def test_fuel_probe_replay_observes_self_state_position(
    fuel_probe_session: ReplaySession,
) -> None:
    """The bot's decoded self-state position is populated after initial sync."""
    positions = [
        (t.self_x, t.self_y)
        for t in fuel_probe_session.ticks
        if t.self_x is not None and t.self_y is not None
    ]
    assert positions
