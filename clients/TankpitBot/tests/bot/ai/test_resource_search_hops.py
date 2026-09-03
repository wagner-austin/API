"""Tests for resource-search hops and fallbacks."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.intent import set_resource_target
from tankpit_bot.bot.ai.resource_search import (
    make_resource_search_hop,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict, make_container_state, make_tank_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


def _foreign_human(tank_id: int, timestamp_ms: int) -> TankStateDict:
    """Build a foreign human-named tank observation.

    The settled-knowledge law's clock trigger: a human-style name that
    is neither this bot nor a fleet sibling advances the foreign-human
    watermark, restoring TTL aging for scans older than the sighting.

    Args:
        tank_id: Registry id.
        timestamp_ms: Observation stamp — becomes the watermark.

    Returns:
        An alive enemy human tank observation.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=50,
        y=50,
        team=2,
        rank=3,
        damage_state=3,
        name="Sigma",
        is_bot=False,
        is_self=False,
        timestamp_ms=timestamp_ms,
        liveness="alive",
    )


class TestHarvestMemoryVeto:
    """The known-empty veto ([[flag-triage-20260729]] F2)."""

    def _ctx_with_beliefs(
        self,
        *,
        container_volume: int,
        container_ts: int,
        now_ms: int,
        human_seen_ms: int | None = None,
    ) -> DecideCtx:
        """Build a ctx whose single dot lands on a viewport with one belief.

        Args:
            container_volume: Belief volume at the landing viewport.
            container_ts: Belief timestamp.
            now_ms: Decision time.
            human_seen_ms: When set, a foreign human tank observed at
                this instant — the settled-knowledge law's clock arm
                ([[flag-triage-20260902]] rows 3-5). ``None`` leaves
                the room settled: knowledge never ages.
        """
        ws = WorldService()
        ws.map_fuel_dots = ((150, 100),)
        world, self_state = make_world(self_x=100, self_y=100, fuel=1100)
        if human_seen_ms is not None:
            world["tanks"]["900"] = _foreign_human(900, human_seen_ms)
        # Production invariant: the ctx world IS the service's world
        # (tick_body passes bot.world.world_state), and the settled-
        # knowledge sweep reads the service's copy.
        ws.world_state = world
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

    def test_stale_empty_beliefs_reopen_the_ground_under_human_presence(self) -> None:
        """With a human about, beliefs older than the window may be wrong.

        The settled-knowledge law's clock arm: only a foreign human
        can refill ground unobserved, so the reopening needs one seen.
        """
        ctx = self._ctx_with_beliefs(
            container_volume=0, container_ts=100000, now_ms=800000, human_seen_ms=800000
        )

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("stale empty beliefs must reopen the ground")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100


class TestSettledKnowledge:
    """The fact arm: a room with no foreign humans never forgets.

    [[flag-triage-20260902]] rows 3-5, measured live: 49% of a static
    Practice session's radars re-scanned known ground and 139 frontier
    teleports never left the viewport, because clock expiry invented
    change no agent could have made.
    """

    def test_a_settled_rooms_barren_sweep_never_reopens(self) -> None:
        """No human, any age: swept-and-empty ground stays dead."""
        from tankpit_bot.state.knowledge_floors import HARVEST_MEMORY_TTL_MS

        # Age well past harvest memory while keeping the stamp inside
        # the fixture's epoch (now_ms = 1_000_000).
        ctx = TestBarrenScanVeto()._ctx_with_swept_landing(
            scan_age_ms=HARVEST_MEMORY_TTL_MS + 100000
        )

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        assert decision is None

    def test_a_settled_rooms_stale_empty_beliefs_stay_empty(self) -> None:
        """No human, any age: a drained container cannot have refilled."""
        ctx = TestHarvestMemoryVeto()._ctx_with_beliefs(
            container_volume=0, container_ts=100000, now_ms=100000000
        )

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        assert decision is None

    def test_a_human_sighting_older_than_the_scan_changes_nothing(self) -> None:
        """A sweep made AFTER the human left is permanent knowledge."""
        from tankpit_bot.state.knowledge_floors import HARVEST_MEMORY_TTL_MS

        # Sweep at t=300_000, human last seen at t=200_000: the scan
        # postdates the last possible unobserved change.
        ctx = TestBarrenScanVeto()._ctx_with_swept_landing(
            scan_age_ms=HARVEST_MEMORY_TTL_MS + 100000,
            human_seen_ms=200000,
        )

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        assert decision is None


class TestPreHuntTopOffBias:
    """The pre-hunt top-off direction bias ([[flag-triage-20260729]] F1)."""

    def _ctx_with_enemy(self, *, hunt_ready: bool, fuel: int = 900) -> DecideCtx:
        """Two-dot world: a far pair vs a lone dot next to the enemy."""
        from tankpit_bot.state.types import make_tank_state

        ws = WorldService()
        ws.map_fuel_dots = ((120, 100), (121, 100), (100, 120))
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
        from tankpit_bot.state.types import make_tank_state
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
        human_seen_ms: int | None = None,
    ) -> DecideCtx:
        """One dot at (150,100); its 16x16 landing viewport swept scan_age_ms ago.

        Args:
            scan_age_ms: Age of the landing-viewport sweep.
            positive_belief: Whether a live container belief sits there.
            hole: Whether one sweep tile is missing.
            human_seen_ms: When set, a foreign human tank observed at
                this instant restores clock aging; ``None`` leaves the
                room settled ([[flag-triage-20260902]] rows 3-5).
        """
        from tankpit_bot.state.knowledge_floors import FORAGE_COVERAGE_TTL_MS

        ws = WorldService()
        ws.map_fuel_dots = ((150, 100),)
        now_ms = 1000000
        world, self_state = make_world(self_x=100, self_y=100, fuel=1100)
        if human_seen_ms is not None:
            world["tanks"]["900"] = _foreign_human(900, human_seen_ms)
        ws.world_state = world
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
            ws=ws,
        )

    def test_barren_swept_viewport_is_vetoed(self) -> None:
        """Fully swept + nothing believed there = guaranteed zero delta.

        Human presence ages the sweep past the forage gate, so the
        BARREN verdict — not "already_scanned" — is what vetoes; the
        settled arm is pinned in ``TestSettledKnowledge``.
        """
        ctx = self._ctx_with_swept_landing(scan_age_ms=200000, human_seen_ms=1000000)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        assert decision is None

    def test_positive_belief_overrides_barren_sweep(self) -> None:
        """A known unharvested container gives the swept ground real value.

        Under human presence the aged sweep no longer blocks the hop
        gate, and the positive belief defeats the barren veto.
        """
        ctx = self._ctx_with_swept_landing(
            scan_age_ms=200000, positive_belief=True, human_seen_ms=1000000
        )

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("a positive-volume belief must keep the swept hop")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100

    def test_incomplete_sweep_is_not_barren(self) -> None:
        """One unswept tile means the ground still holds unknowns.

        Human presence ages the partial sweep past the forage gate so
        the barren question is the one being asked.
        """
        ctx = self._ctx_with_swept_landing(scan_age_ms=200000, hole=True, human_seen_ms=1000000)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("an incompletely swept viewport must stay hoppable")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100

    def test_expired_sweep_reopens_the_ground_under_human_presence(self) -> None:
        """Past harvest memory WITH a human about, the sweep is forgotten.

        The clock arm: unobserved refills need a refiller.
        """
        from tankpit_bot.state.knowledge_floors import HARVEST_MEMORY_TTL_MS

        ctx = self._ctx_with_swept_landing(
            scan_age_ms=HARVEST_MEMORY_TTL_MS + 1, human_seen_ms=1000000
        )

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("an expired sweep must reopen the ground")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100


class TestSearchHopReleasesHeldLocks:
    """The dot hop relocates, so a held plan is RELEASED, not erased.

    Until 2026-09-02 this site cleared the lock with no diagnostic —
    invisible to churn analysis ([[flag-triage-20260902]]). The drop
    now flows through ``release_collect_plan`` with the enumerated
    ``relocated`` reason; the observable contract is that the hop's
    decision leaves no lock behind.
    """

    def test_the_hop_decision_carries_no_lock(self) -> None:
        ws = WorldService()
        ws.map_fuel_dots = ((150, 100),)
        world, self_state = make_world(self_x=100, self_y=100, fuel=1100)
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
        locked = set_resource_target(ctx.ai_state, "fuel", 104, 100)

        decision = make_resource_search_hop(
            ctx, mode="COLLECT", score=500, reason="search_collect_local", ai_state=locked
        )

        if decision is None:
            raise AssertionError("expected a search hop decision")
        assert decision["command"]["cmd_type"] == "teleport"
        updated = decision["updated_ai_state"]
        assert updated["resource_target_kind"] == ""
        assert updated["resource_target_held_ticks"] == 0
