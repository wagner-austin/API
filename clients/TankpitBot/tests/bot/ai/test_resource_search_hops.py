"""Tests for resource-search hops and fallbacks."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.resource_search import (
    is_recently_attempted,
    make_resource_search_hop,
    record_attempt_mark,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


class TestAttemptMarks:
    """Tests for the failed-pickup attempt-tracking helpers."""

    def test_is_recently_attempted_returns_true_within_ttl(self) -> None:
        """A coordinate marked inside the TTL is reported as attempted."""
        marks = {"100,100": 95000}

        assert is_recently_attempted(marks, 100, 100, 100000, ttl_ms=10000) is True

    def test_is_recently_attempted_returns_false_outside_ttl(self) -> None:
        """A coordinate marked outside the TTL is reported as not attempted."""
        marks = {"100,100": 80000}

        assert is_recently_attempted(marks, 100, 100, 100000, ttl_ms=10000) is False

    def test_is_recently_attempted_returns_false_for_unknown_coord(self) -> None:
        """An unmarked coordinate is reported as not attempted."""
        assert is_recently_attempted({}, 100, 100, 100000, ttl_ms=10000) is False

    def test_record_attempt_mark_adds_new_coordinate(self) -> None:
        """A fresh attempt is recorded with the dispatch timestamp."""
        result = record_attempt_mark({}, 100, 100, 100000, ttl_ms=10000)

        assert result == {"100,100": 100000}

    def test_record_attempt_mark_prunes_expired_entries(self) -> None:
        """Expired marks are dropped while the new mark is added."""
        marks = {"50,50": 80000, "60,60": 95000}

        result = record_attempt_mark(marks, 100, 100, 100000, ttl_ms=10000)

        assert "50,50" not in result
        assert result["60,60"] == 95000
        assert result["100,100"] == 100000


class TestHarvestMemoryVeto:
    """The known-empty veto ([[flag-triage-20260729]] F2)."""

    def _ctx_with_beliefs(
        self,
        *,
        container_volume: int,
        container_ts: int,
        now_ms: int,
    ) -> DecideCtx:
        """Build a ctx whose single dot lands on a viewport with one belief."""
        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=1100)
        world["containers"]["152,101"] = make_container_state(
            x=152,
            y=101,
            is_fuel=False,
            volume=container_volume,
            timestamp_ms=container_ts,
            failed_pickups=0,
        )
        # A drained belief far OUTSIDE the landing viewport must never
        # influence the veto — only in-bounds beliefs count.
        world["containers"]["30,30"] = make_container_state(
            x=30,
            y=30,
            is_fuel=False,
            volume=0,
            timestamp_ms=container_ts,
            failed_pickups=0,
        )
        return DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            now_ms,
            None,
            "",
            ((150, 100),),
            ws=ws,
        )

    def test_known_empty_viewport_is_vetoed(self) -> None:
        """A drained fresh belief in the landing viewport kills the hop."""
        ctx = self._ctx_with_beliefs(container_volume=0, container_ts=100000, now_ms=100000)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        assert decision is None

    def test_positive_volume_belief_keeps_the_hop(self) -> None:
        """A single positive-volume belief means the ground has value."""
        ctx = self._ctx_with_beliefs(container_volume=40, container_ts=100000, now_ms=100000)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("positive-volume belief must keep the hop alive")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100

    def test_stale_empty_beliefs_reopen_the_ground(self) -> None:
        """Beliefs older than the harvest window may have respawned."""
        ctx = self._ctx_with_beliefs(container_volume=0, container_ts=100000, now_ms=800000)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("stale empty beliefs must reopen the ground")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100


class TestPreHuntTopOffBias:
    """The pre-hunt top-off direction bias ([[flag-triage-20260729]] F1)."""

    def _ctx_with_enemy(self, *, hunt_ready: bool, fuel: int = 900) -> DecideCtx:
        """Two-dot world: a far pair vs a lone dot next to the enemy."""
        from tankpit_bot.state.types import make_tank_state

        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=fuel, block_scanned=False)
        world["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=100,
            y=124,
            team=2,
            rank=1,
            damage_state=3,
            name="red-5",
            is_bot=True,
            is_self=False,
            timestamp_ms=100000,
        )
        return DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=30 if hunt_ready else 10),
            100000,
            None,
            "",
            ((120, 100), (121, 100), (100, 120)),
            ws=ws,
        )

    def test_hunt_ready_stocks_bias_toward_the_prey(self) -> None:
        """With stocks hunt-ready, the lone dot near the enemy wins."""
        ctx = self._ctx_with_enemy(hunt_ready=True)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("hunt-ready stocks must still produce a hop")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 100
        assert decision["command"]["target_y"] == 120

    def test_understocked_forage_ignores_enemy_direction(self) -> None:
        """Below combat-ready with fuel headroom the denser far cluster wins."""
        ctx = self._ctx_with_enemy(hunt_ready=False)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("understocked forage must still produce a hop")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 120
        assert decision["command"]["target_y"] == 100

    def test_capped_fuel_understocked_runs_the_loot_bias(self) -> None:
        """Equipment-hungry at fuel cap drifts toward the fights.

        Session 7 of run 20260730: radar-broke at fuel 1100 (cap), the
        hop ranking kept touring dense fuel-dot viewports whose fuel
        the tank could not absorb. At cap the dots' only value is
        location, and equipment comes from kills -- the lone dot next
        to the enemy must outrank the far dense pair.
        """
        ctx = self._ctx_with_enemy(hunt_ready=False, fuel=1200)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("capped understocked forage must still produce a hop")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 100
        assert decision["command"]["target_y"] == 120


class TestNearestAliveEnemy:
    """Enemy selection for the top-off bias."""

    def test_skips_allies_corpses_and_unsynced(self) -> None:
        """Only alive, position-synced enemies qualify; nearest wins."""
        from tankpit_bot.bot.ai.resource_search import _nearest_alive_enemy
        from tankpit_bot.state.types import TankStateDict, make_tank_state
        from tankpit_bot.types.constants import TankLiveness

        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=900)

        def _tank(tank_id: int, x: int, y: int, team: int, liveness: TankLiveness) -> TankStateDict:
            return make_tank_state(
                tank_id=tank_id,
                x=x,
                y=y,
                team=team,
                rank=1,
                damage_state=3,
                name=f"t{tank_id}",
                is_bot=True,
                is_self=False,
                timestamp_ms=100000,
                liveness=liveness,
            )

        world["tanks"]["10"] = _tank(10, 101, 101, 1, "alive")  # ally
        world["tanks"]["11"] = _tank(11, 102, 102, 2, "deactivated")  # corpse
        world["tanks"]["12"] = _tank(12, 0, 0, 2, "alive")  # unsynced
        world["tanks"]["13"] = _tank(13, 100, 130, 2, "alive")  # far enemy
        world["tanks"]["14"] = _tank(14, 100, 110, 2, "alive")  # near enemy
        world["tanks"]["15"] = _tank(15, 100, 140, 2, "alive")  # farther after near
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        assert _nearest_alive_enemy(ctx) == (100, 110)

    def test_returns_none_with_no_enemies(self) -> None:
        """An empty registry produces no bias target."""
        from tankpit_bot.bot.ai.resource_search import _nearest_alive_enemy

        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=900)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        assert _nearest_alive_enemy(ctx) is None


class TestBarrenScanVeto:
    """The barren-memory veto ([[flag-triage-20260729]] F2, second half).

    Ground the radar fully swept that revealed NOTHING leaves no
    container beliefs, so the known-empty veto cannot see it; once the
    180 s forage coverage expired it read fully clean and got re-hopped
    for a guaranteed zero-delta scan. The barren veto remembers the
    sweep itself for ``HARVEST_MEMORY_TTL_MS``.
    """

    def _ctx_with_swept_landing(
        self,
        *,
        scan_age_ms: int,
        positive_belief: bool = False,
        hole: bool = False,
    ) -> DecideCtx:
        """One dot at (150,100); its 16x16 landing viewport swept scan_age_ms ago."""
        from tankpit_bot.state.scan_coverage import FORAGE_COVERAGE_TTL_MS

        ws = WorldService()
        now_ms = 1000000
        world, self_state = make_world(self_x=100, self_y=100, fuel=1100)
        # Sweep age must exceed the forage TTL in every test here, so the
        # zero-overlap gate ("already_scanned") never masks the barren gate.
        assert scan_age_ms > FORAGE_COVERAGE_TTL_MS
        sweep_ts = now_ms - scan_age_ms
        for y in range(92, 108):
            for x in range(142, 158):
                world["scanned_tiles"][f"{x},{y}"] = sweep_ts
        if hole:
            del world["scanned_tiles"]["150,100"]
        if positive_belief:
            world["containers"]["151,101"] = make_container_state(
                x=151,
                y=101,
                is_fuel=False,
                volume=60,
                timestamp_ms=sweep_ts,
                failed_pickups=0,
            )
        return DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            now_ms,
            None,
            "",
            ((150, 100),),
            ws=ws,
        )

    def test_barren_swept_viewport_is_vetoed(self) -> None:
        """Fully swept + nothing believed there = guaranteed zero delta."""
        ctx = self._ctx_with_swept_landing(scan_age_ms=200000)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        assert decision is None

    def test_positive_belief_overrides_barren_sweep(self) -> None:
        """A known unharvested container gives the swept ground real value."""
        ctx = self._ctx_with_swept_landing(scan_age_ms=200000, positive_belief=True)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("a positive-volume belief must keep the swept hop")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100

    def test_incomplete_sweep_is_not_barren(self) -> None:
        """One unswept tile means the ground still holds unknowns."""
        ctx = self._ctx_with_swept_landing(scan_age_ms=200000, hole=True)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("an incompletely swept viewport must stay hoppable")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100

    def test_expired_sweep_reopens_the_ground(self) -> None:
        """Past harvest memory the sweep is forgotten and the hop returns."""
        from tankpit_bot.state.scan_coverage import HARVEST_MEMORY_TTL_MS

        ctx = self._ctx_with_swept_landing(scan_age_ms=HARVEST_MEMORY_TTL_MS + 1)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("an expired sweep must reopen the ground")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100
