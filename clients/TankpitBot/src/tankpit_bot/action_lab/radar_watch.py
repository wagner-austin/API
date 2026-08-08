"""Live radar-watch probe: free built-in scans of one spot, for spawn law.

Runs on the ACCOUNT with extra radars TOGGLED OFF (user directive,
2026-07-24: no guest login — disable slot 5 so every scan uses the
free built-in radar). The stock-protection chain is all verified
mechanics: the toggle command is ``build_toggle_equipment_command(5)``
(0x72 'r' hotkey, ASCII '5'), the enabled state is wire-visible
(0x74 push + 0x49 bit-7 flags, both decoded), and with extras off the
scan debit clamps to ``min(10, fuel)`` — free once fuel reaches zero.

The watch itself: one radar scan per ``scan_interval_ms``, one free
map open per ``map_poll_interval_ms``, and a 1-tile walk shuffle per
beat — the first session proved a never-playing client is
DISCONNECTED ~12 minutes after join (wiki log 2026-07-24), so the
watch must genuinely play, exactly like the bot-watch dwell. 0x4F
responses are DIFFS (unchanged already-visible entities are never
re-sent), so after baseline coverage every reveal IS a fresh event;
the map polls give the global fuel baseline for the
near-player-clustering cross-check. Analysis is offline from the
capture.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger

from tankpit_bot.action_lab.probe_base import ProbeBase, ProbeError
from tankpit_bot.action_lab.probe_entrypoint import run_and_save_standard_probe_session
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.types_codecs import encode_teleport_startup_timing
from tankpit_bot.protocol.commands import build_toggle_equipment_command
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state

log = get_logger(__name__)

_RADAR_SLOT = 5
_TOGGLE_SETTLE_MS = 1500


class RadarWatchSessionDict(TypedDict):
    """One live radar-watch session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    duration_ms: int
    scan_interval_ms: int
    map_poll_interval_ms: int
    walks_sent: int
    extras_before: int
    extras_enabled_at_start: bool
    toggles_sent: int
    scans_sent: int
    map_polls_sent: int
    extras_after: int


def encode_radar_watch_session(session: RadarWatchSessionDict) -> JSONObject:
    """Encode a radar-watch session to a JSON object.

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
        "duration_ms": session["duration_ms"],
        "scan_interval_ms": session["scan_interval_ms"],
        "map_poll_interval_ms": session["map_poll_interval_ms"],
        "walks_sent": session["walks_sent"],
        "extras_before": session["extras_before"],
        "extras_enabled_at_start": session["extras_enabled_at_start"],
        "toggles_sent": session["toggles_sent"],
        "scans_sent": session["scans_sent"],
        "map_polls_sent": session["map_polls_sent"],
        "extras_after": session["extras_after"],
    }


def format_radar_watch_summary(session: RadarWatchSessionDict) -> str:
    """Format a compact human-readable summary for a radar-watch session.

    Args:
        session: Session to summarize.

    Returns:
        One-line summary string.
    """
    return (
        f"Radar watch complete: scans={session['scans_sent']} "
        f"map_polls={session['map_polls_sent']} walks={session['walks_sent']} "
        f"toggles={session['toggles_sent']} "
        f"extras {session['extras_before']}->{session['extras_after']} "
        f"duration_ms={session['duration_ms']}"
    )


class RadarWatchProbe(ProbeBase):
    """Stationary free-radar watcher for container spawn dynamics."""

    def toggle_equipment_slot(self, slot: int) -> bool:
        """Send one equipment-slot toggle (the 0x72 'r' hotkey command).

        Args:
            slot: Equipment slot (1-5); 5 is extra radars.

        Returns:
            True if the command was sent.
        """
        return self._send_bytes(
            build_toggle_equipment_command(slot),
            f"toggle_equipment({slot})",
        )

    def _read_extras(self) -> tuple[int, bool]:
        """Query inventory and read the extra-radar count and flag.

        Returns:
            Pair of (count, enabled) for the extra-radar slot.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        wait_page = self._require_page()
        self.request_inventory()
        wait_page.wait_for_timeout(float(_TOGGLE_SETTLE_MS))
        action_hooks.drain_buffered_messages(self, self.world)
        state = get_inventory_state(self.world)
        radars = state["extra_radars"]
        return radars["count"], radars["enabled"]

    def _ensure_extras_disabled(self) -> tuple[int, bool, int]:
        """Disable the extra-radar slot, verifying via the wire state.

        Returns:
            Tuple of (extras count, initially-enabled flag, toggles sent).

        Raises:
            ProbeError: If the slot still reads enabled after toggling.
        """
        count, enabled = self._read_extras()
        log.info("Radar watch: extras=%d enabled=%s at start", count, enabled)
        if not enabled:
            return count, False, 0
        self.toggle_equipment_slot(_RADAR_SLOT)
        count_after, still_enabled = self._read_extras()
        if still_enabled:
            raise ProbeError("extra radars still enabled after toggle; refusing to scan")
        log.info("Radar watch: extras disabled (stock %d preserved)", count_after)
        return count_after, True, 1

    def _watch_loop(
        self,
        duration_ms: int,
        scan_interval_ms: int,
        map_poll_interval_ms: int,
    ) -> tuple[int, int, int]:
        """Scan, map-poll, and walk-shuffle until the duration elapses.

        Each beat walks one tile (east/west alternating) BEFORE the
        scan — a never-playing client is disconnected ~12 minutes
        after join, so the watch must take real actions to survive.

        Args:
            duration_ms: Total watch duration.
            scan_interval_ms: Time between radar scans.
            map_poll_interval_ms: Time between free map opens
                (0 disables map polling entirely — the 2026-07-24
                sessions suggest idling in the map-open state for
                ~12 minutes disconnects the client).

        Returns:
            Tuple of (scans sent, map polls sent, walks sent).
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        started_ms = action_hooks.get_current_time_ms()
        next_map_poll_ms = started_ms
        scans = 0
        map_polls = 0
        walks = 0
        beat = 0
        while action_hooks.get_current_time_ms() - started_ms < duration_ms:
            now_ms = action_hooks.get_current_time_ms()
            if map_poll_interval_ms > 0 and now_ms >= next_map_poll_ms:
                self.open_map()
                map_polls += 1
                next_map_poll_ms = now_ms + map_poll_interval_ms
            self_state = self.get_self_state()
            if self_state is not None:
                step = 1 if beat % 2 == 0 else -1
                self.move_to(self_state["x"] + step, self_state["y"])
                walks += 1
            beat += 1
            self.use_radar()
            scans += 1
            page.wait_for_timeout(float(scan_interval_ms))
            action_hooks.drain_buffered_messages(self, self.world)
        return scans, map_polls, walks

    def execute_probe(
        self,
        *,
        duration_ms: int,
        scan_interval_ms: int,
        map_poll_interval_ms: int,
        initial_sync_timeout_ms: int,
    ) -> RadarWatchSessionDict:
        """Run the live radar-watch session."""
        if duration_ms <= 0:
            raise ProbeError("duration_ms must be positive")
        if scan_interval_ms <= 0:
            raise ProbeError("scan_interval_ms must be positive")

        def _run_ready_session(context: ProbeCommandReadyContextDict) -> RadarWatchSessionDict:
            from tankpit_bot.action_lab import _test_hooks as action_hooks

            extras_before, was_enabled, toggles = self._ensure_extras_disabled()
            watch_started_ms = action_hooks.get_current_time_ms()
            scans, map_polls, walks = self._watch_loop(
                duration_ms,
                scan_interval_ms,
                map_poll_interval_ms,
            )
            extras_after, _ = self._read_extras()
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=watch_started_ms,
            )
            return RadarWatchSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=envelope.startup_timing,
                duration_ms=duration_ms,
                scan_interval_ms=scan_interval_ms,
                map_poll_interval_ms=map_poll_interval_ms,
                walks_sent=walks,
                extras_before=extras_before,
                extras_enabled_at_start=was_enabled,
                toggles_sent=toggles,
                scans_sent=scans,
                map_polls_sent=map_polls,
                extras_after=extras_after,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def _create_radar_watch_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> RadarWatchProbe:
    """Factory for RadarWatchProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        RadarWatchProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, RadarWatchProbe)
    return probe


def run_radar_watch_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = True,
    duration_ms: int = 1800000,
    scan_interval_ms: int = 15000,
    map_poll_interval_ms: int = 30000,
    initial_sync_timeout_ms: int = 10000,
) -> RadarWatchSessionDict:
    """Run a live radar-watch probe and save the session JSON."""

    def _run_session(probe: RadarWatchProbe) -> RadarWatchSessionDict:
        return probe.execute_probe(
            duration_ms=duration_ms,
            scan_interval_ms=scan_interval_ms,
            map_poll_interval_ms=map_poll_interval_ms,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_radar_watch_probe,
        run_session=_run_session,
        encoder=encode_radar_watch_session,
        summary_formatter=format_radar_watch_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "RadarWatchProbe",
    "RadarWatchSessionDict",
    "encode_radar_watch_session",
    "format_radar_watch_summary",
    "run_radar_watch_probe",
]
