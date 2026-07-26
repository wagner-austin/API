"""Live container-density probe: extra-radar sweeps of grid-spread sites.

Measures the field's TRUE hidden-container density — the one number
the 2026-07-25 exposure-law work left as a calibrated assumption
([[game-economy]]): dots are exposure memory, most large fuel is
hidden, and only a full-viewport sweep of FRESH ground samples the
hidden population without bias.

Design: teleport to each of a fixed map-spread site grid and fire ONE
extra radar there (a free radar covers only 5x5 around the tank and
the fixed viewport makes multi-viewport free sweeps impossible —
teleports are not fuel-clamped, so a fuel-0 free-radar probe is
pinned to one viewport forever). Each extra reveals the full 16x16
viewport: the 0x4F response is the DELTA of not-yet-visible entities,
i.e. exactly the hidden population of 256 fresh tiles. Sites are grid
coordinates, NOT fuel dots — sampling at dots would bias density
toward fuel-rich ground; refuel hops to dots happen only between
samples and are never scanned.

Stock etiquette (user rule): extras are budgeted (``max_extras``),
slot 5 is toggled on only for the probe, and the slot's enable state
is RESTORED to what it was at start. Analysis is offline from the
capture (``analysis_scripts/analyze_density_probe.py``).
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
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.protocol.commands import build_quit_command, build_toggle_equipment_command
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state

log = get_logger(__name__)

_RADAR_SLOT = 5
_SETTLE_MS = 2000
_PICKUP_SETTLE_MS = 4000
_FUEL_RESERVE = 250
_MAX_REFUEL_HOPS_PER_SITE = 3
_MAX_BOOTSTRAP_PICKUPS = 12
_LANDING_TOLERANCE = 6

DENSITY_SITES: tuple[tuple[int, int], ...] = tuple(
    (x, y) for y in (40, 96, 152, 208) for x in (40, 96, 152, 208)
)
"""Sixteen map-spread sample sites (a 4x4 grid over the interior).

Grid coordinates, deliberately independent of fuel dots and of any
prior session's ground — a teleport's landing displacement scatters
each visit a little, and the extra radar covers the full viewport
around wherever the tank actually lands.
"""


class DensityProbeSessionDict(TypedDict):
    """One live density-probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    max_extras: int
    sites_planned: int
    sites_scanned: int
    sites_skipped: int
    refuel_hops: int
    bootstrap_pickups: int
    extras_before: int
    extras_enabled_at_start: bool
    toggles_sent: int
    extras_after: int
    fuel_before: int
    fuel_after: int


def encode_density_probe_session(session: DensityProbeSessionDict) -> JSONObject:
    """Encode a density-probe session to a JSON object.

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
        "max_extras": session["max_extras"],
        "sites_planned": session["sites_planned"],
        "sites_scanned": session["sites_scanned"],
        "sites_skipped": session["sites_skipped"],
        "refuel_hops": session["refuel_hops"],
        "bootstrap_pickups": session["bootstrap_pickups"],
        "extras_before": session["extras_before"],
        "extras_enabled_at_start": session["extras_enabled_at_start"],
        "toggles_sent": session["toggles_sent"],
        "extras_after": session["extras_after"],
        "fuel_before": session["fuel_before"],
        "fuel_after": session["fuel_after"],
    }


def format_density_probe_summary(session: DensityProbeSessionDict) -> str:
    """Format a compact human-readable summary for a density session.

    Args:
        session: Session to summarize.

    Returns:
        One-line summary string.
    """
    return (
        f"Density probe complete: sites={session['sites_scanned']}/"
        f"{session['sites_planned']} skipped={session['sites_skipped']} "
        f"refuels={session['refuel_hops']} pickups={session['bootstrap_pickups']} "
        f"toggles={session['toggles_sent']} "
        f"extras {session['extras_before']}->{session['extras_after']} "
        f"fuel {session['fuel_before']}->{session['fuel_after']}"
    )


class DensityProbe(ProbeBase):
    """Grid-sweep extra-radar sampler for the hidden container density."""

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
        wait_page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self)
        state = get_inventory_state(get_world_service())
        radars = state["extra_radars"]
        return radars["count"], radars["enabled"]

    def _ensure_extras_enabled(self) -> tuple[int, bool, int]:
        """Enable the extra-radar slot, verifying via the wire state.

        Returns:
            Tuple of (extras count, initially-enabled flag, toggles sent).

        Raises:
            ProbeError: If the slot still reads disabled after toggling,
                or the stock is empty — a "density scan" with the free
                5x5 would silently sample 25 tiles instead of 256.
        """
        count, enabled = self._read_extras()
        log.info("Density probe: extras=%d enabled=%s at start", count, enabled)
        if count <= 0:
            raise ProbeError("no extra radars in stock; a density sweep needs full-viewport scans")
        if enabled:
            return count, True, 0
        self.toggle_equipment_slot(_RADAR_SLOT)
        count_after, now_enabled = self._read_extras()
        if not now_enabled:
            raise ProbeError("extra radars still disabled after toggle; refusing to scan")
        return count_after, False, 1

    def _restore_extras_state(self, was_enabled: bool) -> int:
        """Put the slot-5 enable flag back the way the account had it.

        Args:
            was_enabled: The slot state read at probe start.

        Returns:
            Toggles sent (0 or 1).

        Raises:
            ProbeError: If the restore toggle does not verify.
        """
        if was_enabled:
            return 0
        self.toggle_equipment_slot(_RADAR_SLOT)
        _, still_enabled = self._read_extras()
        if still_enabled:
            raise ProbeError("extra radars still enabled after restore toggle")
        log.info("Density probe: slot 5 restored to disabled")
        return 1

    def _quit_to_lobby(self) -> None:
        """Send the graceful quit so the tank never lingers in-world.

        Standing rule from the 2026-07-25 incident: an unattended
        probe tank is a target in a PvP world (player 2596 killed the
        immobilized tank and the user lost a rank). Every probe end —
        normal or marooned — exits the room deliberately instead of
        leaving the tank standing until the socket drops.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        self._send_bytes(build_quit_command(), "quit_game")
        page = self._require_page()
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self)
        log.info("Density probe: quit to lobby sent")

    def _current_fuel(self) -> tuple[int, int, int]:
        """Return the tank's live (fuel, x, y).

        Raises:
            ProbeError: If self state is unavailable.
        """
        state = self.get_self_state()
        if state is None:
            raise ProbeError("self state unavailable mid-probe")
        return state["fuel"], state["x"], state["y"]

    def _bootstrap_fuel(self, needed: int) -> int:
        """Walk-collect visible viewport fuel until ``needed`` is met.

        The first live run (2026-07-25 15:15) burned 12 extras at fuel
        0 because every teleport was rejected — teleports are NOT
        fuel-clamped (now live-confirmed), so a broke probe must fund
        its first hop from the ground it stands on: walking and
        pickups work at any fuel level.

        Args:
            needed: Target fuel level.

        Returns:
            Pickup attempts sent.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        attempts = 0
        tried: set[tuple[int, int]] = set()
        while attempts < _MAX_BOOTSTRAP_PICKUPS:
            fuel, x, y = self._current_fuel()
            if fuel >= needed:
                return attempts
            candidates = [
                (abs(c["x"] - x) + abs(c["y"] - y), c["x"], c["y"])
                for c in self.get_world_state()["containers"].values()
                if c["is_fuel"] and c["volume"] > 0 and (c["x"], c["y"]) not in tried
            ]
            if candidates:
                _, cx, cy = min(candidates)
            else:
                # Blind dot-walk: no visible fuel anywhere in the fixed
                # viewport (the marooned-at-0 state the first live runs
                # hit). Walking is free AND instant at fuel 0
                # ([[walk-mechanics]] + the clamp law), and the 0x4C
                # atlas gives known fuel coordinates map-wide — walk to
                # the nearest untried dot and attempt the pickup there.
                self.open_map()
                page.wait_for_timeout(float(_SETTLE_MS))
                action_hooks.drain_buffered_messages(self)
                dots = [
                    (abs(dot[0] - x) + abs(dot[1] - y), dot[0], dot[1])
                    for dot in get_world_service().map_fuel_dots
                    if dot not in tried
                ]
                if not dots:
                    log.info("Density probe: no untried fuel anywhere to bootstrap from")
                    return attempts
                _, cx, cy = min(dots)
                self.move_to(cx, cy)
                page.wait_for_timeout(float(_PICKUP_SETTLE_MS))
                action_hooks.drain_buffered_messages(self)
            tried.add((cx, cy))
            self.pickup_fuel(cx, cy)
            attempts += 1
            page.wait_for_timeout(float(_PICKUP_SETTLE_MS))
            action_hooks.drain_buffered_messages(self)
        return attempts

    def _refuel_toward(self, site_x: int, site_y: int) -> int:
        """Hop to nearest fuel dots until the site teleport is funded.

        Landing on a dot auto-picks whatever it still holds (~40% of
        live dots). Up to ``_MAX_REFUEL_HOPS_PER_SITE`` hops; a dry
        streak is tolerated — the site teleport simply gets attempted
        with whatever fuel remains, and a rejection is itself capture
        evidence.

        Args:
            site_x: Next sample site X.
            site_y: Next sample site Y.

        Returns:
            Refuel hops sent.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        hops = 0
        visited: set[tuple[int, int]] = set()
        while hops < _MAX_REFUEL_HOPS_PER_SITE:
            fuel, x, y = self._current_fuel()
            needed = teleport_cost(x, y, site_x, site_y) + _FUEL_RESERVE
            if fuel >= needed:
                return hops
            self.open_map()
            page.wait_for_timeout(float(_SETTLE_MS))
            action_hooks.drain_buffered_messages(self)
            dots = [
                dot
                for dot in get_world_service().map_fuel_dots
                if dot not in visited
                and dot != (x, y)
                and teleport_cost(x, y, dot[0], dot[1]) <= fuel
            ]
            if not dots:
                log.info("Density probe: no affordable unvisited fuel dots to refuel from")
                return hops
            target = dots[0]
            best_cost = teleport_cost(x, y, target[0], target[1])
            for dot in dots[1:]:
                cost = teleport_cost(x, y, dot[0], dot[1])
                if cost < best_cost:
                    target, best_cost = dot, cost
            visited.add(target)
            self.teleport_to(target[0], target[1])
            hops += 1
            page.wait_for_timeout(float(_SETTLE_MS))
            action_hooks.drain_buffered_messages(self)
        return hops

    def _reach_site(self, site_x: int, site_y: int) -> tuple[bool, int, int]:
        """Fund and attempt the teleport to one site, verifying landing.

        Funding order: walk-pickups from visible viewport fuel first
        (works at any fuel level), then affordable dot hops. The
        landing is VERIFIED from self position before the caller
        spends an extra — the first live run burned its whole budget
        re-scanning one viewport because every broke teleport was
        silently rejected.

        Args:
            site_x: Sample site X.
            site_y: Sample site Y.

        Returns:
            Tuple of (landed, refuel hops, bootstrap pickups).
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        fuel, x, y = self._current_fuel()
        needed = teleport_cost(x, y, site_x, site_y) + _FUEL_RESERVE
        pickups = self._bootstrap_fuel(needed) if fuel < needed else 0
        hops = self._refuel_toward(site_x, site_y)
        self.teleport_to(site_x, site_y)
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self)
        _, landed_x, landed_y = self._current_fuel()
        landed = max(abs(landed_x - site_x), abs(landed_y - site_y)) <= _LANDING_TOLERANCE
        if not landed:
            log.info(
                "Density probe: site (%d,%d) not reached (at %d,%d) - extra preserved",
                site_x,
                site_y,
                landed_x,
                landed_y,
            )
        return landed, hops, pickups

    def _sweep_sites(self, max_extras: int) -> tuple[int, int, int, int]:
        """Teleport the site grid, firing one extra radar per landed site.

        Stops when the extras budget is spent or the stock runs out
        (never falls back to the free 5x5 — that would silently change
        the instrument mid-measurement). A site whose teleport did not
        verifiably land is SKIPPED without spending an extra.

        Args:
            max_extras: Extra radars the probe may consume.

        Returns:
            Tuple of (sites scanned, refuel hops, bootstrap pickups,
            sites skipped).
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        scanned = 0
        refuels = 0
        pickups = 0
        skipped = 0
        for site_x, site_y in DENSITY_SITES:
            if scanned >= max_extras:
                break
            count, _ = self._read_extras()
            if count <= 0:
                log.info("Density probe: extras exhausted after %d sites", scanned)
                break
            landed, hops, picks = self._reach_site(site_x, site_y)
            refuels += hops
            pickups += picks
            if not landed:
                skipped += 1
                fuel, _, _ = self._current_fuel()
                if fuel < _FUEL_RESERVE:
                    # Marooned: unreachable site AND broke after every
                    # funding path. Grinding on is exactly how the
                    # 2026-07-25 tank became a sitting duck — stop the
                    # sweep now; the caller quits to the lobby.
                    log.info(
                        "Density probe: marooned at fuel %d after %d sites - aborting sweep",
                        fuel,
                        scanned,
                    )
                    break
                continue
            self.use_radar()
            scanned += 1
            page.wait_for_timeout(float(_SETTLE_MS))
            action_hooks.drain_buffered_messages(self)
        return scanned, refuels, pickups, skipped

    def execute_probe(
        self,
        *,
        max_extras: int,
        initial_sync_timeout_ms: int,
    ) -> DensityProbeSessionDict:
        """Run the live density-probe session."""
        if max_extras <= 0:
            raise ProbeError("max_extras must be positive")

        def _run_ready_session(context: ProbeCommandReadyContextDict) -> DensityProbeSessionDict:
            from tankpit_bot.action_lab import _test_hooks as action_hooks

            fuel_before, _, _ = self._current_fuel()
            extras_before, was_enabled, toggles = self._ensure_extras_enabled()
            sweep_started_ms = action_hooks.get_current_time_ms()
            scanned, refuels, pickups, skipped = self._sweep_sites(max_extras)
            toggles += self._restore_extras_state(was_enabled)
            extras_after, _ = self._read_extras()
            fuel_after, _, _ = self._current_fuel()
            self._quit_to_lobby()
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=sweep_started_ms,
            )
            return DensityProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=envelope.startup_timing,
                max_extras=max_extras,
                sites_planned=len(DENSITY_SITES),
                sites_scanned=scanned,
                sites_skipped=skipped,
                refuel_hops=refuels,
                bootstrap_pickups=pickups,
                extras_before=extras_before,
                extras_enabled_at_start=was_enabled,
                toggles_sent=toggles,
                extras_after=extras_after,
                fuel_before=fuel_before,
                fuel_after=fuel_after,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def _create_density_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> DensityProbe:
    """Factory for DensityProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        DensityProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, DensityProbe)
    return probe


def run_density_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = True,
    max_extras: int = 12,
    initial_sync_timeout_ms: int = 10000,
) -> DensityProbeSessionDict:
    """Run a live density probe and save the session JSON."""

    def _run_session(probe: DensityProbe) -> DensityProbeSessionDict:
        return probe.execute_probe(
            max_extras=max_extras,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_density_probe,
        run_session=_run_session,
        encoder=encode_density_probe_session,
        summary_formatter=format_density_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "DENSITY_SITES",
    "DensityProbe",
    "DensityProbeSessionDict",
    "encode_density_probe_session",
    "format_density_probe_summary",
    "run_density_probe",
]
