"""Live larder-gate probe: own-tile equipment pickup vs the adjacent law.

The larder plan ([[larder-plan]]) wants harvest teleports that land ON
a verified equipment container and pick it up with the programmatic
``pickup_equipment`` command — the same wire action the human
long-press dispatches ([[client-commands]]). The single recorded
own-tile sample (capture 2026-06-21 16:54:26) failed silently, so the
plan's §Probe gate requires a deliberate live answer before any
bot-loop code: stand ON a verified equipment container, try the
pickup from the tank's own tile, then step off one cardinal tile and
try again from adjacency as the control.

Search reuses the density probe's funded site hops and extras
etiquette (budgeted, slot-5 state restored at exit): each landed site
spends one extra radar to reveal the full viewport, and every
equipment container it exposes becomes an attempt candidate.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger

from tankpit_bot.action_lab.density_probe import DENSITY_SITES, DensityProbe
from tankpit_bot.action_lab.equipment_pickup import total_inventory_count
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.action_lab.probe_entrypoint import run_and_save_standard_probe_session
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.types_codecs import encode_teleport_startup_timing
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state.types import ContainerStateDict

log = get_logger(__name__)

_SETTLE_MS = 2000
_PICKUP_SETTLE_MS = 4000
_CARDINALS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

LarderAttemptStatus = Literal["own_tile_pickup", "adjacent_pickup", "no_pickup"]


class LarderAttemptDict(TypedDict):
    """One own-tile pickup attempt against a single equipment container."""

    container_x: int
    container_y: int
    landed_x: int
    landed_y: int
    landed_on_container: bool
    walked_onto_container: bool
    stood_on_container: bool
    own_tile_sent: bool
    own_tile_picked: bool
    stepped_off: bool
    adjacent_sent: bool
    adjacent_picked: bool
    inventory_before: int
    inventory_after: int
    status: LarderAttemptStatus


class LarderProbeSessionDict(TypedDict):
    """One live larder-gate probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    max_attempts: int
    max_extras: int
    search_scans: int
    search_hops: int
    attempts: list[LarderAttemptDict]
    own_tile_successes: int
    own_tile_failures: int
    adjacent_successes: int
    extras_before: int
    extras_enabled_at_start: bool
    toggles_sent: int
    extras_after: int
    fuel_before: int
    fuel_after: int


def encode_larder_attempt(attempt: LarderAttemptDict) -> JSONObject:
    """Encode one larder pickup attempt to a JSON object.

    Args:
        attempt: Attempt to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "container_x": attempt["container_x"],
        "container_y": attempt["container_y"],
        "landed_x": attempt["landed_x"],
        "landed_y": attempt["landed_y"],
        "landed_on_container": attempt["landed_on_container"],
        "walked_onto_container": attempt["walked_onto_container"],
        "stood_on_container": attempt["stood_on_container"],
        "own_tile_sent": attempt["own_tile_sent"],
        "own_tile_picked": attempt["own_tile_picked"],
        "stepped_off": attempt["stepped_off"],
        "adjacent_sent": attempt["adjacent_sent"],
        "adjacent_picked": attempt["adjacent_picked"],
        "inventory_before": attempt["inventory_before"],
        "inventory_after": attempt["inventory_after"],
        "status": attempt["status"],
    }


def encode_larder_probe_session(session: LarderProbeSessionDict) -> JSONObject:
    """Encode a larder-probe session to a JSON object.

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
        "max_attempts": session["max_attempts"],
        "max_extras": session["max_extras"],
        "search_scans": session["search_scans"],
        "search_hops": session["search_hops"],
        "attempts": [encode_larder_attempt(attempt) for attempt in session["attempts"]],
        "own_tile_successes": session["own_tile_successes"],
        "own_tile_failures": session["own_tile_failures"],
        "adjacent_successes": session["adjacent_successes"],
        "extras_before": session["extras_before"],
        "extras_enabled_at_start": session["extras_enabled_at_start"],
        "toggles_sent": session["toggles_sent"],
        "extras_after": session["extras_after"],
        "fuel_before": session["fuel_before"],
        "fuel_after": session["fuel_after"],
    }


def format_larder_probe_summary(session: LarderProbeSessionDict) -> str:
    """Format a compact human-readable summary for a larder session.

    Args:
        session: Session to summarize.

    Returns:
        One-line summary string.
    """
    own_tile_sent = session["own_tile_successes"] + session["own_tile_failures"]
    return (
        f"Larder probe complete: attempts={len(session['attempts'])}/"
        f"{session['max_attempts']} "
        f"own-tile {session['own_tile_successes']}/{own_tile_sent} "
        f"adjacent={session['adjacent_successes']} "
        f"scans={session['search_scans']} hops={session['search_hops']} "
        f"extras {session['extras_before']}->{session['extras_after']} "
        f"fuel {session['fuel_before']}->{session['fuel_after']}"
    )


class LarderProbe(DensityProbe):
    """Own-tile equipment pickup prober — the larder plan's build gate."""

    def _inventory_total(self) -> int:
        """Query inventory and return the summed slot count.

        Returns:
            Total items across all five equipment slots.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        self.request_inventory()
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self, self.world)
        return total_inventory_count(get_inventory_state(self.world))

    def _nearest_equipment(self, tried: set[tuple[int, int]]) -> ContainerStateDict | None:
        """Return the nearest untried believed equipment container on land.

        Water-sitting shore containers are excluded: the first live run
        (2026-07-27 22:49) aimed at three of them, every walk-on was
        rejected (0x52 err=1), and no trial could execute — and the
        larder's own scope already excludes inaccessible containers.

        Args:
            tried: Container tiles already attempted this session.

        Returns:
            Nearest candidate by Chebyshev distance, or ``None``.

        Raises:
            ProbeError: If the terrain map is unavailable.
        """
        terrain = self.world.get_terrain_map()
        if terrain is None:
            raise ProbeError("terrain map is unavailable")
        _, x, y = self._current_fuel()
        best: ContainerStateDict | None = None
        best_distance = 0
        for container in self.get_world_state()["containers"].values():
            if container["is_fuel"] or container["failed_pickups"] > 0:
                continue
            if (container["x"], container["y"]) in tried:
                continue
            if not terrain.is_passable(container["x"], container["y"]):
                continue
            distance = max(abs(container["x"] - x), abs(container["y"] - y))
            if best is None or distance < best_distance:
                best, best_distance = container, distance
        return best

    def _search_equipment(
        self,
        tried: set[tuple[int, int]],
        scans_left: int,
    ) -> tuple[ContainerStateDict | None, int, int]:
        """Hop funded sites, one extra-radar reveal each, until equipment shows.

        Args:
            tried: Container tiles already attempted this session.
            scans_left: Remaining extra-radar budget for the search.

        Returns:
            Tuple of (found container or None, scans used, site hops used).
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        found = self._nearest_equipment(tried)
        if found is not None:
            return found, 0, 0
        scans = 0
        hops = 0
        _, x, y = self._current_fuel()
        sites = sorted(
            (teleport_cost(x, y, site_x, site_y), site_x, site_y)
            for site_x, site_y in DENSITY_SITES
        )
        for _, site_x, site_y in sites:
            if scans >= scans_left:
                break
            landed, _, _ = self._reach_site(site_x, site_y)
            hops += 1
            if not landed:
                continue
            self.use_radar()
            scans += 1
            page.wait_for_timeout(float(_SETTLE_MS))
            action_hooks.drain_buffered_messages(self, self.world)
            found = self._nearest_equipment(tried)
            if found is not None:
                return found, scans, hops
        return None, scans, hops

    def _step_off(self, container_x: int, container_y: int) -> bool:
        """Walk to a cardinal neighbor of the container, verified by position.

        Args:
            container_x: Container X tile.
            container_y: Container Y tile.

        Returns:
            True once the tank verifiably stands cardinally adjacent.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        _, x, y = self._current_fuel()
        if abs(x - container_x) + abs(y - container_y) == 1:
            return True
        for dx, dy in _CARDINALS:
            neighbor_x, neighbor_y = container_x + dx, container_y + dy
            self.move_to(neighbor_x, neighbor_y)
            page.wait_for_timeout(float(_PICKUP_SETTLE_MS))
            action_hooks.drain_buffered_messages(self, self.world)
            _, x, y = self._current_fuel()
            if (x, y) == (neighbor_x, neighbor_y):
                return True
        return False

    def _attempt_container(self, container: ContainerStateDict) -> LarderAttemptDict:
        """Teleport onto one equipment container and run both pickup trials.

        Args:
            container: Believed equipment container to attempt.

        Returns:
            Fully-populated attempt record.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        container_x, container_y = container["x"], container["y"]
        # Open one slot of pickup headroom BEFORE reading the baseline:
        # a fully-capped tank rejects every pickup with the 0x52 code-7
        # receipt (run 2026-07-27 22:56 — all-capped inventory failed
        # both trials while the one radar-spent slot let the identical
        # adjacent pickup credit), so each attempt burns one extra
        # radar to make the trials interpretable.
        self.use_radar()
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self, self.world)
        inventory_before = self._inventory_total()
        self.open_map()
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self, self.world)
        self.teleport_to(container_x, container_y)
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self, self.world)
        _, landed_x, landed_y = self._current_fuel()
        landed_on = (landed_x, landed_y) == (container_x, container_y)
        walked_onto = False
        if not landed_on:
            self.move_to(container_x, container_y)
            page.wait_for_timeout(float(_PICKUP_SETTLE_MS))
            action_hooks.drain_buffered_messages(self, self.world)
            _, walk_x, walk_y = self._current_fuel()
            walked_onto = (walk_x, walk_y) == (container_x, container_y)
        stood = landed_on or walked_onto
        own_sent = False
        own_picked = False
        if stood:
            own_sent = True
            self.pickup_equipment(container_x, container_y)
            page.wait_for_timeout(float(_PICKUP_SETTLE_MS))
            action_hooks.drain_buffered_messages(self, self.world)
            own_picked = self._inventory_total() > inventory_before
        stepped_off = False
        adjacent_sent = False
        adjacent_picked = False
        if not own_picked:
            stepped_off = self._step_off(container_x, container_y)
            if stepped_off:
                adjacent_sent = True
                control_before = self._inventory_total()
                self.pickup_equipment(container_x, container_y)
                page.wait_for_timeout(float(_PICKUP_SETTLE_MS))
                action_hooks.drain_buffered_messages(self, self.world)
                adjacent_picked = self._inventory_total() > control_before
        inventory_after = self._inventory_total()
        if own_picked:
            status: LarderAttemptStatus = "own_tile_pickup"
        elif adjacent_picked:
            status = "adjacent_pickup"
        else:
            status = "no_pickup"
        log.info(
            "Larder probe: (%d,%d) stood=%s own_tile=%s adjacent=%s",
            container_x,
            container_y,
            stood,
            own_picked,
            adjacent_picked,
        )
        return LarderAttemptDict(
            container_x=container_x,
            container_y=container_y,
            landed_x=landed_x,
            landed_y=landed_y,
            landed_on_container=landed_on,
            walked_onto_container=walked_onto,
            stood_on_container=stood,
            own_tile_sent=own_sent,
            own_tile_picked=own_picked,
            stepped_off=stepped_off,
            adjacent_sent=adjacent_sent,
            adjacent_picked=adjacent_picked,
            inventory_before=inventory_before,
            inventory_after=inventory_after,
            status=status,
        )

    def execute_larder_probe(
        self,
        *,
        max_attempts: int,
        max_extras: int,
        initial_sync_timeout_ms: int,
    ) -> LarderProbeSessionDict:
        """Run the live larder-gate probe session."""
        if max_attempts <= 0:
            raise ProbeError("max_attempts must be positive")
        if max_extras <= 0:
            raise ProbeError("max_extras must be positive")

        def _run_ready_session(context: ProbeCommandReadyContextDict) -> LarderProbeSessionDict:
            from tankpit_bot.action_lab import _test_hooks as action_hooks

            fuel_before, _, _ = self._current_fuel()
            extras_before, was_enabled, toggles = self._ensure_extras_enabled()
            first_attempt_started_ms = action_hooks.get_current_time_ms()
            attempts: list[LarderAttemptDict] = []
            tried: set[tuple[int, int]] = set()
            scans = 0
            hops = 0
            while len(attempts) < max_attempts:
                container, used_scans, used_hops = self._search_equipment(
                    tried,
                    max_extras - scans,
                )
                scans += used_scans
                hops += used_hops
                if container is None:
                    log.info("Larder probe: no equipment container found (scans=%d)", scans)
                    break
                tried.add((container["x"], container["y"]))
                attempts.append(self._attempt_container(container))
            toggles += self._restore_extras_state(was_enabled)
            extras_after, _ = self._read_extras()
            fuel_after, _, _ = self._current_fuel()
            self._quit_to_lobby()
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_attempt_started_ms,
            )
            own_successes = sum(1 for attempt in attempts if attempt["own_tile_picked"])
            own_failures = sum(
                1
                for attempt in attempts
                if attempt["own_tile_sent"] and not attempt["own_tile_picked"]
            )
            adjacent_successes = sum(1 for attempt in attempts if attempt["adjacent_picked"])
            return LarderProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=envelope.startup_timing,
                max_attempts=max_attempts,
                max_extras=max_extras,
                search_scans=scans,
                search_hops=hops,
                attempts=attempts,
                own_tile_successes=own_successes,
                own_tile_failures=own_failures,
                adjacent_successes=adjacent_successes,
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


def _create_larder_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> LarderProbe:
    """Factory for LarderProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        LarderProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, LarderProbe)
    return probe


def run_larder_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = True,
    max_attempts: int = 3,
    max_extras: int = 6,
    initial_sync_timeout_ms: int = 10000,
) -> LarderProbeSessionDict:
    """Run a live larder-gate probe and save the session JSON."""

    def _run_session(probe: LarderProbe) -> LarderProbeSessionDict:
        return probe.execute_larder_probe(
            max_attempts=max_attempts,
            max_extras=max_extras,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_larder_probe,
        run_session=_run_session,
        encoder=encode_larder_probe_session,
        summary_formatter=format_larder_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "LarderAttemptDict",
    "LarderAttemptStatus",
    "LarderProbe",
    "LarderProbeSessionDict",
    "encode_larder_attempt",
    "encode_larder_probe_session",
    "format_larder_probe_summary",
    "run_larder_probe",
]
