"""Live viewport/autoscroll probe: edge walks under both toggle states.

Measures the muddiest corner of the model ([[viewport-shift-protocol]],
open questions from the 2026-07-25 density runs):

* what the server's 0x5A window does as a tank walks toward and past
  the visible edge — with autoscroll OFF and then ON;
* the command-acceptance boundary — single move commands at growing
  offsets until the server's 0x52 rejection, mapping what bounds a
  legal target (the last 0x5A window, a radius, or something else).

Protocol per phase: teleport to a probe anchor near the tank (small,
funded hop; map opened first — teleports need the map-open state;
the landing is verified from the position echo before walking), then
walk terrain-aware single-tile steps toward the window's EAST edge
column, attempt ONE crossing step past the edge (the decisive
experiment: does the 0x5A window shift, and only under autoscroll
ON?), then fire single long moves at ``_LONG_OFFSETS`` east of
wherever the tank stands. Every accept (0x47 echo), reject (0x52),
0x5A origin, and 0x3D position lands in the capture for offline
analysis (``analysis_scripts/analyze_viewport_probe.py``).

The first live run (20260725-190352) proved the blind version of the
walk useless for the ON phase: a step fired before the anchor echo
landed, the server pathfound the tank west around water, and the
edge was never reached. Hence the landing verification, the echo
wait after every step, and the GIF-terrain routing here.

Autoscroll is flipped between phases with a physical ``a`` key press
(the key-probe machinery) and VERIFIED from the wire: the server
echoes the short 0x41 autoscroll ack carrying the new enabled flag.
The probe expects to find autoscroll OFF (the user's standing
config), runs phase A as-found, toggles ON for phase B, restores OFF
— verified — and quits to the lobby. A fuel floor guards every
phase: below it the probe skips ahead to restore + quit rather than
stranding the tank (the 2026-07-25 sitting-duck rule).
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.probe_base import ProbeBase, ProbeError
from tankpit_bot.action_lab.probe_entrypoint import run_and_save_standard_probe_session
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.types_codecs import encode_teleport_startup_timing
from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import try_decode_plaintext_ack
from tankpit_bot.sniffer.world_state import get_terrain_map

log = get_logger(__name__)

_SETTLE_MS = 1500
_TOGGLE_SETTLE_MS = 2000
_WALK_STEPS = 16
_LONG_OFFSETS: tuple[int, ...] = (6, 10, 14, 18, 24)
_ANCHOR_HOP = 6
_FUEL_FLOOR = 120
_POLL_MS = 500
_STEP_POLLS = 4
_CROSS_ROW_SCAN = 3


class ViewportProbeSessionDict(TypedDict):
    """One live viewport/autoscroll probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    walk_steps_per_phase: int
    long_offsets: list[int]
    walks_sent_off: int
    longs_sent_off: int
    walks_sent_on: int
    longs_sent_on: int
    toggles_sent: int
    ack_states: list[bool]
    fuel_before: int
    fuel_after: int


def encode_viewport_probe_session(session: ViewportProbeSessionDict) -> JSONObject:
    """Encode a viewport-probe session to a JSON object.

    Args:
        session: Session to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "walk_steps_per_phase": session["walk_steps_per_phase"],
        "long_offsets": list(session["long_offsets"]),
        "walks_sent_off": session["walks_sent_off"],
        "longs_sent_off": session["longs_sent_off"],
        "walks_sent_on": session["walks_sent_on"],
        "longs_sent_on": session["longs_sent_on"],
        "toggles_sent": session["toggles_sent"],
        "ack_states": list(session["ack_states"]),
        "fuel_before": session["fuel_before"],
        "fuel_after": session["fuel_after"],
    }


def format_viewport_probe_summary(session: ViewportProbeSessionDict) -> str:
    """Format a compact human-readable summary for a viewport session.

    Args:
        session: Session to summarize.

    Returns:
        One-line summary string.
    """
    return (
        f"Viewport probe complete: walks off/on="
        f"{session['walks_sent_off']}/{session['walks_sent_on']} "
        f"longs off/on={session['longs_sent_off']}/{session['longs_sent_on']} "
        f"toggles={session['toggles_sent']} acks={session['ack_states']} "
        f"fuel {session['fuel_before']}->{session['fuel_after']}"
    )


class ViewportProbe(ProbeBase):
    """Edge-walk and boundary-move sampler under both autoscroll states."""

    def _current_fuel(self) -> tuple[int, int, int]:
        """Return the tank's live (fuel, x, y).

        Raises:
            ProbeError: If self state is unavailable.
        """
        state = self.get_self_state()
        if state is None:
            raise ProbeError("self state unavailable mid-probe")
        return state["fuel"], state["x"], state["y"]

    def _read_autoscroll_ack(self, start_index: int) -> bool:
        """Read the autoscroll ack from frames captured after a press.

        The ack is the server's PLAINTEXT two-byte echo of the toggle
        (raw ``"A0"``/``"A1"``, un-XORed) — it must be read from the
        raw frame body, never through the XOR decode path.

        Args:
            start_index: Capture index at the moment of the key press.

        Returns:
            The acked enabled flag.

        Raises:
            ProbeError: If no autoscroll ack arrived — the toggle is
                unverified and the probe must not continue guessing.
        """
        for captured in self.messages[start_index:]:
            if captured["direction"] != "received":
                continue
            data = decode_base64_safe(captured["payload"])
            if not data:
                continue
            offset = 0
            while offset + 2 < len(data):
                length = data[offset] | (data[offset + 1] << 8)
                offset += 2
                if length == 0 or offset + length > len(data):
                    break
                body = data[offset : offset + length]
                offset += length
                ack = try_decode_plaintext_ack(body)
                if ack is not None and ack["msg_type"] == "autoscroll_ack":
                    return ack["enabled"]
        raise ProbeError("no autoscroll ack after the 'a' press; toggle unverified")

    def _toggle_autoscroll(self) -> bool:
        """Press ``a`` and return the wire-verified new state.

        Returns:
            The acked enabled flag after the toggle.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._page
        if page is None:
            raise ProbeError("page is unavailable")
        start_index = len(self.messages)
        page.keyboard.press("a")
        page.wait_for_timeout(float(_TOGGLE_SETTLE_MS))
        action_hooks.drain_buffered_messages(self)
        enabled = self._read_autoscroll_ack(start_index)
        log.info("Viewport probe: autoscroll ack enabled=%s", enabled)
        return enabled

    def _window(self) -> tuple[int, int, int, int]:
        """Return the last-known 0x5A window as (left, top, width, height)."""
        viewport = self.get_world_state()["viewport"]
        return (viewport["left"], viewport["top"], viewport["width"], viewport["height"])

    def _terrain_map(self) -> TerrainMapProtocol:
        """Return the loaded GIF terrain map.

        Raises:
            ProbeError: If no terrain map is loaded — the edge walk
                cannot be routed blind (run 20260725-190352 walked
                into water and never reached the edge).
        """
        terrain = get_terrain_map()
        if terrain is None:
            raise ProbeError("terrain map unavailable; cannot route the edge walk")
        return terrain

    def _await_position_change(self, x: int, y: int) -> bool:
        """Poll the wire until the tank's echoed position leaves (x, y).

        Args:
            x: Position X before the command.
            y: Position Y before the command.

        Returns:
            True once the echoed position differs, False on timeout.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        for _ in range(_STEP_POLLS):
            page.wait_for_timeout(float(_POLL_MS))
            action_hooks.drain_buffered_messages(self)
            _, new_x, new_y = self._current_fuel()
            if (new_x, new_y) != (x, y):
                return True
        return False

    def _anchor(self) -> bool:
        """Hop a few tiles east so the viewport re-centers on the tank.

        A teleport is the one guaranteed re-center (autoscroll-
        independent), giving each phase a clean centered baseline.
        The map is opened first — teleports require the map-open
        state — and the landing is verified from the position echo
        before the phase walks (run 20260725-190352 fired its first
        step before the echo landed and walked from a stale position).

        Returns:
            True when the hop landed, False when the phase must be
            skipped (unfunded or the teleport never echoed).
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        fuel, x, y = self._current_fuel()
        if fuel < _FUEL_FLOOR:
            log.info("Viewport probe: fuel %d below floor - skipping anchor", fuel)
            return False
        self.open_map()
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self)
        self.teleport_to(x + _ANCHOR_HOP, y)
        if not self._await_position_change(x, y):
            log.info("Viewport probe: anchor teleport never echoed - skipping phase")
            return False
        return True

    def _pick_step(
        self,
        x: int,
        y: int,
        top: int,
        height: int,
        visited: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Pick the next walkable one-tile step toward the east edge.

        Callers only step from strictly inside the window, so every
        eastward candidate stays at or inside the edge column.

        Args:
            x: Current position X.
            y: Current position Y.
            top: Window top row.
            height: Window height in tiles.
            visited: Tiles already stood on this walk (loop guard).

        Returns:
            The chosen in-window step, or None when nothing walkable
            and unvisited remains.
        """
        terrain = self._terrain_map()
        candidates = ((x + 1, y), (x + 1, y - 1), (x + 1, y + 1), (x, y - 1), (x, y + 1))
        for step_x, step_y in candidates:
            if not top <= step_y < top + height:
                continue
            if (step_x, step_y) in visited:
                continue
            if terrain.is_passable(step_x, step_y):
                return step_x, step_y
        return None

    def _walk_to_edge(self) -> int:
        """Walk terrain-aware steps until the tank stands on the east edge.

        Returns:
            Steps sent (stops early at the fuel floor, when no
            walkable step remains, or when a step never echoes).
        """
        steps = 0
        visited: set[tuple[int, int]] = set()
        left, top, width, height = self._window()
        edge_x = left + width - 1
        for _ in range(_WALK_STEPS):
            fuel, x, y = self._current_fuel()
            if fuel < _FUEL_FLOOR:
                log.info("Viewport probe: fuel %d below floor - ending walk", fuel)
                break
            if x >= edge_x:
                break
            visited.add((x, y))
            step = self._pick_step(x, y, top, height, visited)
            if step is None:
                log.info("Viewport probe: no walkable unvisited step toward the edge")
                break
            self.move_to(step[0], step[1])
            steps += 1
            if not self._await_position_change(x, y):
                log.info("Viewport probe: step to (%d,%d) never echoed", step[0], step[1])
                break
        return steps

    def _cross_edge(self) -> None:
        """Attempt one step past the east edge and record the window's answer.

        The decisive experiment: with the tank ON the edge column, a
        move one column beyond it either rejects (static window) or
        walks — and if it walks, the capture shows whether a 0x5A
        shift came with it. Skipped when the walk never reached the
        edge or no passable tile exists just past it.
        """
        left, top, width, _height = self._window()
        edge_x = left + width - 1
        fuel, x, y = self._current_fuel()
        if fuel < _FUEL_FLOOR or x < edge_x:
            log.info("Viewport probe: crossing skipped (fuel=%d x=%d edge=%d)", fuel, x, edge_x)
            return
        terrain = self._terrain_map()
        target_y: int | None = None
        for delta in range(_CROSS_ROW_SCAN + 1):
            for row in (y - delta, y + delta):
                if terrain.is_passable(edge_x + 1, row):
                    target_y = row
                    break
            if target_y is not None:
                break
        if target_y is None:
            log.info("Viewport probe: no passable tile past the edge near row %d", y)
            return
        self.move_to(edge_x + 1, target_y)
        self._await_position_change(x, y)
        new_left, new_top, _, _ = self._window()
        log.info(
            "Viewport probe: edge crossing to (%d,%d) - window (%d,%d) -> (%d,%d)",
            edge_x + 1,
            target_y,
            left,
            top,
            new_left,
            new_top,
        )

    def _long_moves(self) -> int:
        """Fire single moves at growing eastward offsets.

        Returns:
            Long moves sent (stops early at the fuel floor).
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        sent = 0
        for offset in _LONG_OFFSETS:
            fuel, x, y = self._current_fuel()
            if fuel < _FUEL_FLOOR:
                log.info("Viewport probe: fuel %d below floor - ending long moves", fuel)
                break
            self.move_to(x + offset, y)
            sent += 1
            page.wait_for_timeout(float(_SETTLE_MS))
            action_hooks.drain_buffered_messages(self)
        return sent

    def _run_phase(self) -> tuple[int, int]:
        """Anchor, edge-walk, edge-cross, and boundary-probe one phase.

        Returns:
            Tuple of (walk steps sent, long moves sent).
        """
        if not self._anchor():
            return 0, 0
        walks = self._walk_to_edge()
        self._cross_edge()
        return walks, self._long_moves()

    def execute_probe(
        self,
        *,
        initial_sync_timeout_ms: int,
    ) -> ViewportProbeSessionDict:
        """Run the live viewport/autoscroll probe session."""

        def _run_ready_session(context: ProbeCommandReadyContextDict) -> ViewportProbeSessionDict:
            from tankpit_bot.action_lab import _test_hooks as action_hooks

            fuel_before, _, _ = self._current_fuel()
            phase_started_ms = action_hooks.get_current_time_ms()
            ack_states: list[bool] = []
            toggles = 1
            # An aborted probe must still leave the room — an
            # unattended tank is a target in a PvP world (the
            # 2026-07-25 density-probe tank was killed and the
            # account lost a rank).
            try:
                # Normalize to OFF. The initial state is unknowable without
                # a press (no query carries the flag), and the first live
                # run proved it cannot be assumed: a fresh browser session
                # started ON despite the user's own client showing OFF —
                # the flag looks client-local, not server-persisted. The
                # first press both reveals and flips the state.
                state = self._toggle_autoscroll()
                ack_states.append(state)
                if state:
                    state = self._toggle_autoscroll()
                    ack_states.append(state)
                    toggles += 1
                    if state:
                        raise ProbeError("autoscroll still enabled after the normalization press")
                walks_off, longs_off = self._run_phase()
                enabled = self._toggle_autoscroll()
                ack_states.append(enabled)
                toggles += 1
                if not enabled:
                    raise ProbeError("autoscroll acked DISABLED when switching to the ON phase")
                walks_on, longs_on = self._run_phase()
                restored = self._toggle_autoscroll()
                ack_states.append(restored)
                toggles += 1
                if restored:
                    raise ProbeError("autoscroll still enabled after the restore press")
            except ProbeError:
                self.quit_to_lobby()
                raise
            self.quit_to_lobby()
            fuel_after, _, _ = self._current_fuel()
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=phase_started_ms,
            )
            return ViewportProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=envelope.startup_timing,
                walk_steps_per_phase=_WALK_STEPS,
                long_offsets=list(_LONG_OFFSETS),
                walks_sent_off=walks_off,
                longs_sent_off=longs_off,
                walks_sent_on=walks_on,
                longs_sent_on=longs_on,
                toggles_sent=toggles,
                ack_states=ack_states,
                fuel_before=fuel_before,
                fuel_after=fuel_after,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def _create_viewport_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> ViewportProbe:
    """Factory for ViewportProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        ViewportProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, ViewportProbe)
    return probe


def run_viewport_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = True,
    initial_sync_timeout_ms: int = 10000,
) -> ViewportProbeSessionDict:
    """Run a live viewport probe and save the session JSON."""

    def _run_session(probe: ViewportProbe) -> ViewportProbeSessionDict:
        return probe.execute_probe(initial_sync_timeout_ms=initial_sync_timeout_ms)

    return run_and_save_standard_probe_session(
        probe_factory=_create_viewport_probe,
        run_session=_run_session,
        encoder=encode_viewport_probe_session,
        summary_formatter=format_viewport_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "ViewportProbe",
    "ViewportProbeSessionDict",
    "encode_viewport_probe_session",
    "format_viewport_probe_summary",
    "run_viewport_probe",
]
