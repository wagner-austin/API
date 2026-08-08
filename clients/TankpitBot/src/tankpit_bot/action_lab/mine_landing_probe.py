"""Live mine-landing probe: does a teleport LANDING on a mine detonate it?

Walk-over is law (user, 2026-07-28): only the mine on the stepped tile
fires, movement stops, one hit max per movement. What a teleport
LANDING on a mined tile does has never been measured, and it gates the
ring-2 miner stand-off doctrine ([[bot-behavior-contract]] §6): a
fresh 3x3 placement mines all 8 tiles around the placer, so aiming an
approach teleport at a miner hands the server 8 mined displacement
landings. Three possible answers, each a different doctrine:
detonate-on-landing (ring-2 aiming mandatory), coexist-until-step
(rings safe to land in, teleport out), or displaced-off-mines (no aim
change needed at all).

Per attempt: find an enemy mine (radar reveals mines into the
registry with their team), teleport aiming AT its tile, then read the
wire -- actual landing, fuel drop vs the pure teleport cost (an extra
~45 is the detonation bill), and whether the mine survived in the
registry (0x45 removes it). Reuses the density probe's funded hops
and extras etiquette for the search.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger

from tankpit_bot.action_lab.density_probe import DENSITY_SITES, DensityProbe
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
from tankpit_bot.state.types import MineStateDict

log = get_logger(__name__)

_SETTLE_MS = 2000

MINE_HIT_COST = 45
"""Measured walk-into-enemy-mine bill ([[game-economy]]); the landing
verdict compares the observed extra loss against this."""


class MineLandingAttemptDict(TypedDict):
    """One teleport-at-a-mine attempt."""

    mine_x: int
    mine_y: int
    mine_team: int
    own_team: int
    start_x: int
    start_y: int
    landed_x: int
    landed_y: int
    landed_on_mine: bool
    fuel_before: int
    fuel_after: int
    landing_teleport_cost: int
    extra_loss: int
    mine_survived: bool


class MineLandingProbeSessionDict(TypedDict):
    """One live mine-landing probe session."""

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
    attempts: list[MineLandingAttemptDict]
    detonations: int
    coexists: int
    displaced_off: int
    extras_before: int
    extras_enabled_at_start: bool
    toggles_sent: int
    extras_after: int
    fuel_before: int
    fuel_after: int


def encode_mine_landing_attempt(attempt: MineLandingAttemptDict) -> JSONObject:
    """Encode one mine-landing attempt to a JSON object.

    Args:
        attempt: Attempt to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "mine_x": attempt["mine_x"],
        "mine_y": attempt["mine_y"],
        "mine_team": attempt["mine_team"],
        "own_team": attempt["own_team"],
        "start_x": attempt["start_x"],
        "start_y": attempt["start_y"],
        "landed_x": attempt["landed_x"],
        "landed_y": attempt["landed_y"],
        "landed_on_mine": attempt["landed_on_mine"],
        "fuel_before": attempt["fuel_before"],
        "fuel_after": attempt["fuel_after"],
        "landing_teleport_cost": attempt["landing_teleport_cost"],
        "extra_loss": attempt["extra_loss"],
        "mine_survived": attempt["mine_survived"],
    }


def encode_mine_landing_probe_session(session: MineLandingProbeSessionDict) -> JSONObject:
    """Encode a mine-landing probe session to a JSON object.

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
        "attempts": [encode_mine_landing_attempt(attempt) for attempt in session["attempts"]],
        "detonations": session["detonations"],
        "coexists": session["coexists"],
        "displaced_off": session["displaced_off"],
        "extras_before": session["extras_before"],
        "extras_enabled_at_start": session["extras_enabled_at_start"],
        "toggles_sent": session["toggles_sent"],
        "extras_after": session["extras_after"],
        "fuel_before": session["fuel_before"],
        "fuel_after": session["fuel_after"],
    }


def format_mine_landing_probe_summary(session: MineLandingProbeSessionDict) -> str:
    """Format a compact human-readable summary for a mine-landing session.

    Args:
        session: Session to summarize.

    Returns:
        One-line summary string.
    """
    return (
        f"Mine-landing probe complete: attempts={len(session['attempts'])}/"
        f"{session['max_attempts']} "
        f"detonations={session['detonations']} coexists={session['coexists']} "
        f"displaced_off={session['displaced_off']} "
        f"scans={session['search_scans']} hops={session['search_hops']} "
        f"extras {session['extras_before']}->{session['extras_after']} "
        f"fuel {session['fuel_before']}->{session['fuel_after']}"
    )


class MineLandingProbe(DensityProbe):
    """Teleport-onto-enemy-mine prober -- the ring-2 doctrine's gate."""

    def _own_team(self) -> int:
        """Return the tank's own team.

        Returns:
            Team id from self state.

        Raises:
            ProbeError: If self state is unavailable.
        """
        state = self.get_self_state()
        if state is None:
            raise ProbeError("self state unavailable mid-probe")
        return state["team"]

    def _nearest_enemy_mine(self, tried: set[tuple[int, int]]) -> MineStateDict | None:
        """Return the nearest untried believed enemy mine.

        Args:
            tried: Mine tiles already attempted this session.

        Returns:
            Nearest candidate by Chebyshev distance, or ``None``.
        """
        own_team = self._own_team()
        _, x, y = self._current_fuel()
        best: MineStateDict | None = None
        best_distance = 0
        for mine in self.get_world_state()["mines"].values():
            if mine["team"] == own_team:
                continue
            if (mine["x"], mine["y"]) in tried:
                continue
            distance = max(abs(mine["x"] - x), abs(mine["y"] - y))
            if best is None or distance < best_distance:
                best, best_distance = mine, distance
        return best

    def _search_enemy_mine(
        self,
        tried: set[tuple[int, int]],
        scans_left: int,
    ) -> tuple[MineStateDict | None, int, int]:
        """Hop funded sites, one extra-radar reveal each, until a mine shows.

        Args:
            tried: Mine tiles already attempted this session.
            scans_left: Remaining extra-radar budget for the search.

        Returns:
            Tuple of (found mine or None, scans used, site hops used).
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        found = self._nearest_enemy_mine(tried)
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
            found = self._nearest_enemy_mine(tried)
            if found is not None:
                return found, scans, hops
        return None, scans, hops

    def _mine_present(self, mine_x: int, mine_y: int) -> bool:
        """Return whether the registry still holds a mine at the tile.

        Args:
            mine_x: Mine X tile.
            mine_y: Mine Y tile.

        Returns:
            True when the mine belief survives.
        """
        return f"{mine_x},{mine_y}" in self.get_world_state()["mines"]

    def _attempt_mine_landing(self, mine: MineStateDict) -> MineLandingAttemptDict:
        """Teleport aiming AT one enemy mine and read what the server did.

        Args:
            mine: Believed enemy mine to land on.

        Returns:
            Fully-populated attempt record.
        """
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        page = self._require_page()
        mine_x, mine_y = mine["x"], mine["y"]
        fuel_before, start_x, start_y = self._current_fuel()
        self.open_map()
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self, self.world)
        self.teleport_to(mine_x, mine_y)
        page.wait_for_timeout(float(_SETTLE_MS))
        action_hooks.drain_buffered_messages(self, self.world)
        fuel_after, landed_x, landed_y = self._current_fuel()
        cost = teleport_cost(start_x, start_y, landed_x, landed_y)
        extra_loss = (fuel_before - fuel_after) - cost
        survived = self._mine_present(mine_x, mine_y)
        record = MineLandingAttemptDict(
            mine_x=mine_x,
            mine_y=mine_y,
            mine_team=mine["team"],
            own_team=self._own_team(),
            start_x=start_x,
            start_y=start_y,
            landed_x=landed_x,
            landed_y=landed_y,
            landed_on_mine=(landed_x, landed_y) == (mine_x, mine_y),
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            landing_teleport_cost=cost,
            extra_loss=extra_loss,
            mine_survived=survived,
        )
        log.info(
            "Mine-landing probe: aim (%d,%d) landed (%d,%d) cost=%d extra=%d survived=%s",
            mine_x,
            mine_y,
            landed_x,
            landed_y,
            cost,
            extra_loss,
            survived,
        )
        return record

    def execute_mine_landing_probe(
        self,
        *,
        max_attempts: int,
        max_extras: int,
        initial_sync_timeout_ms: int,
    ) -> MineLandingProbeSessionDict:
        """Run the live mine-landing probe session."""
        if max_attempts <= 0:
            raise ProbeError("max_attempts must be positive")
        if max_extras <= 0:
            raise ProbeError("max_extras must be positive")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> MineLandingProbeSessionDict:
            from tankpit_bot.action_lab import _test_hooks as action_hooks

            fuel_before, _, _ = self._current_fuel()
            extras_before, was_enabled, toggles = self._ensure_extras_enabled()
            first_attempt_started_ms = action_hooks.get_current_time_ms()
            attempts: list[MineLandingAttemptDict] = []
            tried: set[tuple[int, int]] = set()
            scans = 0
            hops = 0
            while len(attempts) < max_attempts:
                mine, used_scans, used_hops = self._search_enemy_mine(
                    tried,
                    max_extras - scans,
                )
                scans += used_scans
                hops += used_hops
                if mine is None:
                    log.info("Mine-landing probe: no enemy mine found (scans=%d)", scans)
                    break
                tried.add((mine["x"], mine["y"]))
                attempts.append(self._attempt_mine_landing(mine))
            toggles += self._restore_extras_state(was_enabled)
            extras_after, _ = self._read_extras()
            fuel_after, _, _ = self._current_fuel()
            self._quit_to_lobby()
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_attempt_started_ms,
            )
            detonations = 0
            coexists = 0
            displaced_off = 0
            for attempt in attempts:
                billed = attempt["extra_loss"] >= MINE_HIT_COST
                if billed or not attempt["mine_survived"]:
                    detonations += 1
                elif attempt["landed_on_mine"]:
                    coexists += 1
                else:
                    displaced_off += 1
            return MineLandingProbeSessionDict(
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
                detonations=detonations,
                coexists=coexists,
                displaced_off=displaced_off,
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


def _create_mine_landing_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> MineLandingProbe:
    """Factory for MineLandingProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        MineLandingProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, MineLandingProbe)
    return probe


def run_mine_landing_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = True,
    max_attempts: int = 3,
    max_extras: int = 6,
    initial_sync_timeout_ms: int = 10000,
) -> MineLandingProbeSessionDict:
    """Run a live mine-landing probe and save the session JSON."""

    def _run_session(probe: MineLandingProbe) -> MineLandingProbeSessionDict:
        return probe.execute_mine_landing_probe(
            max_attempts=max_attempts,
            max_extras=max_extras,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_mine_landing_probe,
        run_session=_run_session,
        encoder=encode_mine_landing_probe_session,
        summary_formatter=format_mine_landing_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "MINE_HIT_COST",
    "MineLandingAttemptDict",
    "MineLandingProbe",
    "MineLandingProbeSessionDict",
    "encode_mine_landing_attempt",
    "encode_mine_landing_probe_session",
    "format_mine_landing_probe_summary",
    "run_mine_landing_probe",
]
