"""Replay-based integration tests for the full decoder pipeline.

Drives every received frame from real captured WebSocket sessions through the
production ``process_received_message`` entry point — exercising real XOR
decoding, real length-based dispatch, real container subtype routing, and
real world-state mutators. Assertions are anchored on state transitions
observed by replaying the captures (not on mock-built expectations), so any
regression in the decoder stack surfaces as a real protocol-state failure.

Distinct from the synthetic-bytes unit tests under tests/protocol/ and
tests/container/, which exercise individual decoders in isolation. These
tests exercise the WHOLE chain: a recorded byte stream produces the same
world-state mutations a live session would.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.inventory import InventoryState
from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.world_state import get_world_service, get_world_state, reset_world_state
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.sniffer.xor import build_global_xor_table
from tankpit_bot.types import CaptureSession, decode_capture_session

REPO_ROOT = Path(__file__).resolve().parents[2]


def total_inventory_count(inv: InventoryState) -> int:
    """Sum all item counts in an inventory state."""
    return (
        inv["armor_shields"]["count"]
        + inv["dual_shots"]["count"]
        + inv["missile_shots"]["count"]
        + inv["homing_shots"]["count"]
        + inv["extra_radars"]["count"]
    )


def _load(path: Path) -> CaptureSession:
    text = core_hooks.read_text(path)
    return decode_capture_session(narrow_json_to_dict(load_json_str(text)))


def _replay_all_received(session: CaptureSession) -> int:
    magic = session["magic"]
    if magic is None:
        raise RuntimeError("capture has no magic key — cannot replay binary frames")
    reset_world_state()
    build_global_xor_table(magic)
    count = 0
    for msg in session["messages"]:
        if msg["direction"] != "received":
            continue
        process_received_message(msg["payload"])
        count += 1
    return count


def _replay_through(session: CaptureSession, *, stop_at_index: int) -> int:
    """Replay received frames up to (not including) the absolute message index."""
    magic = session["magic"]
    if magic is None:
        raise RuntimeError("capture has no magic key")
    reset_world_state()
    build_global_xor_table(magic)
    received = 0
    for i, msg in enumerate(session["messages"]):
        if i >= stop_at_index:
            break
        if msg["direction"] != "received":
            continue
        process_received_message(msg["payload"])
        received += 1
    return received


@pytest.fixture()
def _isolate_world_state() -> Generator[None, None, None]:
    """Each test gets a clean global world-state and a fresh teardown."""
    reset_world_state()
    yield
    reset_world_state()


@pytest.mark.usefixtures("_isolate_world_state")
def test_fuel_probe_capture_replays_to_observed_terminal_state() -> None:
    """The full fuel_probe capture produces the recorded terminal state.

    Ground truth was captured by replaying the file through the real pipeline
    once and recording the resulting state. Any decoder-stack regression that
    changes how tanks, containers, mines, viewports, inventory, or self-state
    are mutated will fail this assertion.
    """
    session = _load(REPO_ROOT / "fuel_probe.capture_session.json")
    received = _replay_all_received(session)

    world = get_world_state()
    inv = get_inventory_state(get_world_service())
    self_state = world["self_state"]

    assert received == 119

    if self_state is None:
        pytest.fail("replay did not populate self_state — 0x3E tank_status decoder broken")
    assert self_state["tank_id"] == 1301
    assert self_state["x"] == 146
    assert self_state["y"] == 110
    assert self_state["fuel"] == 934
    assert self_state["rank"] == 1

    viewport = world["viewport"]
    assert viewport["left"] == 138
    assert viewport["top"] == 102
    assert viewport["width"] == 16
    assert viewport["height"] == 16

    assert len(world["tanks"]) == 37
    assert "1301" in world["tanks"]
    assert "500" in world["tanks"]

    assert len(world["containers"]) == 23
    assert len(world["mines"]) == 8
    assert len(world["scanned_viewports"]) == 4

    assert total_inventory_count(inv) == 107
    assert inv["armor_shields"]["count"] == 25
    assert inv["dual_shots"]["count"] == 25
    assert inv["missile_shots"]["count"] == 25
    assert inv["homing_shots"]["count"] == 25
    assert inv["extra_radars"]["count"] == 7


@pytest.mark.usefixtures("_isolate_world_state")
def test_fuel_probe_inventory_zero_until_first_sync_frame() -> None:
    """Real ``_inventory_state`` stays empty until the first 0x49 frame.

    Replays frames before message index 51 (the first received frame to grow
    inventory total — discovered separately by scanning the capture). Proves
    the inventory tracker is not credited by any earlier frame and is only
    populated by the explicit absolute-sync message.
    """
    session = _load(REPO_ROOT / "fuel_probe.capture_session.json")
    received = _replay_through(session, stop_at_index=51)

    assert received > 0
    assert total_inventory_count(get_inventory_state(get_world_service())) == 0


@pytest.mark.usefixtures("_isolate_world_state")
def test_fuel_probe_inventory_jumps_to_112_after_first_sync_frame() -> None:
    """The first 0x49 absolute-inventory frame credits all five slots at once.

    Replays through and including the first inventory-growing frame, asserts
    the real tracker now reports the recorded post-sync totals.
    """
    session = _load(REPO_ROOT / "fuel_probe.capture_session.json")
    _replay_through(session, stop_at_index=52)

    inv = get_inventory_state(get_world_service())
    assert total_inventory_count(inv) == 112
    assert inv["armor_shields"]["count"] == 25
    assert inv["dual_shots"]["count"] == 25
    assert inv["missile_shots"]["count"] == 25
    assert inv["homing_shots"]["count"] == 25
    assert inv["extra_radars"]["count"] == 12


@pytest.mark.usefixtures("_isolate_world_state")
def test_teleport_probe_capture_replays_to_observed_terminal_state() -> None:
    """The teleport probe capture exercises teleport-landed signals end to end.

    No radar/pickup activity in this session, so containers/mines/scanned all
    stay empty — but the teleport sequencing produced a final landing at
    (134, 69) and 68 received frames worth of tank-registry updates.
    """
    session = _load(REPO_ROOT / "teleport_probe.capture_session.json")
    received = _replay_all_received(session)

    world = get_world_state()
    self_state = world["self_state"]

    assert received == 68
    if self_state is None:
        pytest.fail("replay did not populate self_state")
    assert self_state["tank_id"] == 1301
    assert (self_state["x"], self_state["y"]) == (134, 69)
    assert self_state["fuel"] == 897
    assert len(world["tanks"]) == 37
    assert len(world["containers"]) == 0
    assert len(world["mines"]) == 0
    assert total_inventory_count(get_inventory_state(get_world_service())) == 124


@pytest.mark.usefixtures("_isolate_world_state")
def test_enemy_teleport_probe_capture_replays_to_observed_terminal_state() -> None:
    """The enemy-teleport capture lands on an enemy and decrements radar slots."""
    session = _load(REPO_ROOT / "enemy_teleport_probe.capture_session.json")
    received = _replay_all_received(session)

    world = get_world_state()
    self_state = world["self_state"]
    inv = get_inventory_state(get_world_service())

    assert received == 71
    if self_state is None:
        pytest.fail("replay did not populate self_state")
    assert self_state["tank_id"] == 1301
    assert (self_state["x"], self_state["y"]) == (100, 167)
    assert self_state["fuel"] == 373
    assert len(world["tanks"]) == 37
    assert total_inventory_count(inv) == 119
    assert inv["extra_radars"]["count"] == 19


@pytest.mark.usefixtures("_isolate_world_state")
def test_movement_probe_capture_replays_to_observed_terminal_state() -> None:
    """The movement probe capture exercises pure walk-to-target sequencing.

    No radar, no pickup, no teleport — inventory stays at the original 125.
    Proves the movement_response / position_update decoders update self
    coordinates correctly when no inventory-affecting frames arrive.
    """
    session = _load(REPO_ROOT / "movement_probe.capture_session.json")
    received = _replay_all_received(session)

    world = get_world_state()
    self_state = world["self_state"]

    assert received == 62
    if self_state is None:
        pytest.fail("replay did not populate self_state")
    assert (self_state["x"], self_state["y"]) == (131, 118)
    assert self_state["fuel"] == 1076
    assert len(world["containers"]) == 0
    assert total_inventory_count(get_inventory_state(get_world_service())) == 125


@pytest.mark.usefixtures("_isolate_world_state")
def test_root_capture_session_replays_to_observed_terminal_state() -> None:
    """The unnamed root capture (capture_session.json) has mixed activity.

    Some radar (3 scanned viewports, 7 containers discovered) and no radar
    consumption (inventory radar slot still 25). Validates the cross-cutting
    case where multiple decoder paths fire across the same session.
    """
    session = _load(REPO_ROOT / "capture_session.json")
    received = _replay_all_received(session)

    world = get_world_state()
    self_state = world["self_state"]
    inv = get_inventory_state(get_world_service())

    assert received == 118
    if self_state is None:
        pytest.fail("replay did not populate self_state")
    assert (self_state["x"], self_state["y"]) == (81, 85)
    assert self_state["fuel"] == 1100
    assert len(world["tanks"]) == 37
    assert len(world["containers"]) == 7
    assert len(world["mines"]) == 0
    assert len(world["scanned_viewports"]) == 3
    assert total_inventory_count(inv) == 125
    assert inv["extra_radars"]["count"] == 25


@pytest.mark.usefixtures("_isolate_world_state")
def test_fuel_probe_radar_uses_decrement_extra_radars_individually() -> None:
    """Each radar use processed during replay reduces the radar slot by exactly 1.

    Five radar-use frames (indices 67, 80, 97, 114, 130 in the capture)
    incrementally decrement ``extra_radars`` from 12 -> 7 — the only slot that
    changes between the initial sync and end of capture.
    """
    session = _load(REPO_ROOT / "fuel_probe.capture_session.json")

    radar_use_indices = [67, 80, 97, 114, 130]
    expected_radar_totals = [11, 10, 9, 8, 7]

    magic = session["magic"]
    if magic is None:
        pytest.fail("capture session must have a magic key")

    for stop_index, expected_radar in zip(radar_use_indices, expected_radar_totals, strict=True):
        reset_world_state()
        build_global_xor_table(magic)
        for i, msg in enumerate(session["messages"]):
            if i > stop_index:
                break
            if msg["direction"] == "received":
                process_received_message(msg["payload"])

        inv = get_inventory_state(get_world_service())
        assert inv["extra_radars"]["count"] == expected_radar, (
            f"after frame {stop_index} expected radar={expected_radar} "
            f"got {inv['extra_radars']['count']}"
        )
        assert inv["armor_shields"]["count"] == 25
        assert inv["dual_shots"]["count"] == 25
        assert inv["missile_shots"]["count"] == 25
        assert inv["homing_shots"]["count"] == 25
