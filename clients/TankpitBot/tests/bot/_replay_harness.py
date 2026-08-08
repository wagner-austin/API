"""Replay harness — drive the real ``Bot`` through a captured WS session.

The strongest integration test architecture available: every line of game
logic, every decoder, every world-state mutator, every ``decide()`` call,
every executor validation, and every command-encoding step runs FOR REAL.
The ONLY substitutions are at the OS-level boundary that has no equivalent
in CI:

* browser bootstrap (`_navigate_and_login`, `_setup_*`, `_wait_for_game_ready`,
  `_gather_intel`, `_cleanup`) — there is no Chromium in tests
* the WebSocket ``_send_bytes`` dispatch — there is no live game server to
  receive bytes; we capture them as the observable side-effect instead

Everything above those boundaries is the production ``Bot`` class. The XOR
table is built from the recorded session magic. Received frames from the
capture are pushed into ``_cdp_message_buffer`` — the same list the live
CDP listener writes to — so ``drain_messages`` sees them via the real path.

Iteration plan (per discussion 2026-05-26): emit a per-tick trace, the user
reviews decisions and locks in correct ones. Snapshot tests are then valid
specifications of correctness, not just frozen current behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.inventory import InventoryState
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.types import CaptureSession, decode_capture_session
from tests.fakes import FakeCDPSession


@dataclass
class TickRecord:
    """One tick's observable outputs.

    Captures bot state machine, AI mode, the dispatched command (if any),
    and a small slice of world state. The trace file emits one of these per
    tick — diffs against the trace become the regression signal.
    """

    tick: int
    drained: int
    state: str
    ai_mode: str
    ai_mode_state: str
    combat_target_id: int
    resource_target: str
    in_flight_kind: str
    self_x: int | None
    self_y: int | None
    fuel: int | None
    inv_total: int
    tanks_known: int
    containers_known: int
    mines_known: int
    dispatched: list[str] = field(default_factory=list)


@dataclass
class ReplaySession:
    """All ticks plus dispatched-command timeline from one replay run."""

    capture_path: Path
    total_received_frames: int
    ticks: list[TickRecord]
    all_dispatched: list[tuple[int, str]]  # (tick, cmd_name)


class ReplayBot(Bot):
    """Bot subclass with browser bootstrap stubbed and dispatch captured.

    Initialization avoids touching Playwright; the harness manually seeds
    ``_magic``, the XOR table, and the global inventory/world-state singletons
    before stepping ticks.
    """

    def __init__(self, target_url: str = "https://tankpit.com/play") -> None:
        super().__init__(target_url, headless=True, prefer_account=False)
        self.dispatched_commands: list[str] = []

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """Capture dispatched commands instead of writing to a WebSocket.

        Returning ``True`` matches the real success path — the bot's state
        machine then transitions to the post-dispatch state (e.g. ``MOVING``)
        exactly as it would in a live run.
        """
        _ = data
        self.dispatched_commands.append(cmd_name)
        return True

    def take_dispatched(self) -> list[str]:
        """Drain the dispatched-command buffer and return what was captured."""
        out = self.dispatched_commands[:]
        self.dispatched_commands.clear()
        return out


def load_capture(path: Path) -> CaptureSession:
    """Load and validate a captured session via the real codec."""
    text = core_hooks.read_text(path)
    return decode_capture_session(narrow_json_to_dict(load_json_str(text)))


def _received_payloads(session: CaptureSession) -> list[str]:
    """Filter the message log to received frames in order."""
    return [m["payload"] for m in session["messages"] if m["direction"] == "received"]


def run_replay(
    capture_path: Path,
    *,
    frames_per_tick: int = 5,
    mode: Literal["batch"] = "batch",
) -> ReplaySession:
    """Replay a capture through ReplayBot, one tick per batch of frames.

    ``frames_per_tick`` is a deliberately simple grouping. The capture's
    original timing is not preserved (no per-frame timestamp tick markers
    exist in the protocol); a future iteration could group frames by
    inter-arrival gap to better mimic live cadence.
    """
    _ = mode
    session = load_capture(capture_path)
    magic = session["magic"]
    if magic is None:
        raise RuntimeError(f"capture {capture_path.name} has no magic key")

    bot = ReplayBot()
    bot._magic = magic
    bot.xor_table = build_session_xor_table(magic)
    bot._cdp = FakeCDPSession(emit_runtime_frames=False)

    payloads = _received_payloads(session)
    ticks: list[TickRecord] = []
    all_dispatched: list[tuple[int, str]] = []
    tick_num = 0
    cursor = 0
    while cursor < len(payloads):
        batch = payloads[cursor : cursor + frames_per_tick]
        cursor += len(batch)
        bot._cdp_message_buffer.extend(batch)
        _tick_once(bot)
        dispatched = bot.take_dispatched()
        ticks.append(_record_tick(bot, tick_num, drained=len(batch), dispatched=dispatched))
        for cmd in dispatched:
            all_dispatched.append((tick_num, cmd))
        tick_num += 1

    return ReplaySession(
        capture_path=capture_path,
        total_received_frames=len(payloads),
        ticks=ticks,
        all_dispatched=all_dispatched,
    )


def _record_tick(
    bot: ReplayBot, tick_num: int, *, drained: int, dispatched: list[str]
) -> TickRecord:
    """Snapshot the observable bot + world state at the end of one tick."""
    ws = bot.world
    world = ws.get_world_state()
    self_state = world["self_state"]
    inv = get_inventory_state(ws)
    state_data = bot.get_state_data()
    ai = bot._ai_state
    resource_target = (
        f"{ai['resource_target_kind']}@{ai['resource_target_x']},{ai['resource_target_y']}"
        if ai["resource_target_kind"]
        else "-"
    )
    return TickRecord(
        tick=tick_num,
        drained=drained,
        state=state_data["state"],
        ai_mode=ai["mode"],
        ai_mode_state=ai["mode_state"],
        combat_target_id=ai["combat_target_id"],
        resource_target=resource_target,
        in_flight_kind=state_data["in_flight_action"]["kind"],
        self_x=self_state["x"] if self_state is not None else None,
        self_y=self_state["y"] if self_state is not None else None,
        fuel=self_state["fuel"] if self_state is not None else None,
        inv_total=_inv_total(inv),
        tanks_known=len(world["tanks"]),
        containers_known=len(world["containers"]),
        mines_known=len(world["mines"]),
        dispatched=dispatched,
    )


def _inv_total(inv: InventoryState) -> int:
    """Sum all five inventory slot counts using the real ``count`` field."""
    return (
        inv["armor_shields"]["count"]
        + inv["dual_shots"]["count"]
        + inv["missile_shots"]["count"]
        + inv["homing_shots"]["count"]
        + inv["extra_radars"]["count"]
    )


def format_trace(session: ReplaySession) -> str:
    """Render the per-tick trace as a human-readable diffable text block."""
    lines = [
        f"# Replay trace for {session.capture_path.name}",
        f"# received_frames={session.total_received_frames} ticks={len(session.ticks)}",
        f"# dispatched_commands={len(session.all_dispatched)}",
        "",
        (
            "tick drn state              mode      mstate         tgt rsrc"
            "            in_flight  self      fuel  inv  tnks cont mines  dispatched"
        ),
    ]
    for t in session.ticks:
        pos = f"({t.self_x},{t.self_y})" if t.self_x is not None else "(none)"
        fuel = f"{t.fuel:>5}" if t.fuel is not None else "  -  "
        tgt = str(t.combat_target_id) if t.combat_target_id != -1 else "-"
        dispatched = ",".join(t.dispatched) if t.dispatched else "-"
        lines.append(
            f"{t.tick:>4} "
            f"{t.drained:>3} "
            f"{t.state:<18} "
            f"{t.ai_mode:<9} "
            f"{t.ai_mode_state:<14} "
            f"{tgt:>3} "
            f"{t.resource_target:<15} "
            f"{t.in_flight_kind:<10} "
            f"{pos:<10} "
            f"{fuel} "
            f"{t.inv_total:>4} "
            f"{t.tanks_known:>4} "
            f"{t.containers_known:>4} "
            f"{t.mines_known:>5}  "
            f"{dispatched}"
        )
    return "\n".join(lines) + "\n"
