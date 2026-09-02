"""Tests for :mod:`tankpit_bot.bot.ai.mode_gates`.

Every entry and exit predicate: fuel floors, weapon reserves, radar
minimums, the human-combat lock, and the resume permission.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.mode_controller import (
    apply_dispatch_counters,
    derive_hunt_mode_state,
    make_hold_decision,
    set_ai_mode,
)
from tankpit_bot.bot.ai.mode_gates import (
    hunt_fuel_floor,
    should_enter_collect,
    should_enter_hunt,
    should_exit_collect,
    should_exit_hunt,
)
from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_initial_ai_state,
)
from tankpit_bot.bot.tick_loop_types import (
    make_tick_decision,
)
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_move_command,
    make_radar_command,
    make_teleport_command,
)
from tankpit_bot.fleetshare.types import EngagementDoctrine
from tests.bot.ai._mode_fixtures import (
    _make_ctx,
    _make_decision,
    _make_hold_inventory,
)


def test_set_ai_mode_preserves_started_timestamp_when_mode_continues() -> None:
    """Rewriting substate within the same mode keeps the original entry time."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 2000,
        }
    )

    updated = set_ai_mode(state, "HUNT", "ENGAGE", 5000)

    assert updated["mode_state"] == "ENGAGE"
    assert updated["mode_started_ms"] == 2000


def test_should_enter_collect_fires_below_full_between_kills() -> None:
    """With no combat lock, anything short of a full tank collects.

    User contract 2026-07-25: hunting is a privilege of a full tank,
    so between kills the entry bar is the rank capacity (1200 at
    rank 2) -- the old ``fuel_low + engagement_budget`` floor (650)
    is subsumed. The low threshold itself still fires regardless.
    """
    assert should_enter_collect(_make_ctx(fuel=150)) is True
    assert should_enter_collect(_make_ctx(fuel=200)) is True
    assert should_enter_collect(_make_ctx(fuel=650)) is True
    assert should_enter_collect(_make_ctx(fuel=1199)) is True
    assert should_enter_collect(_make_ctx(fuel=1200)) is False


def test_should_exit_collect_requires_rank_capacity_fuel() -> None:
    """Fuel recovery exit demands the rank's actual full tank.

    User ruling 2026-07-25: "just determine max fuel based on the
    tank rank" -- at rank 2 the capacity is 1200, so 1100 no longer
    releases the mode.
    """
    assert should_exit_collect(_make_ctx(fuel=1200)) is True
    assert should_exit_collect(_make_ctx(fuel=1100)) is False
    assert should_exit_collect(_make_ctx(fuel=800)) is False


def test_should_enter_collect_uses_break_threshold() -> None:
    """Equipment recovery entry uses the configured break threshold."""
    assert should_enter_collect(_make_ctx(dual_count=5, radar_count=5)) is True
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=30)) is False


def test_should_exit_collect_requires_a_full_stock() -> None:
    """COLLECT releases only at a genuinely full stock.

    User contract (2026-07-25): "never hunt if it is not full on
    everything except -5 max radar." At rank 2 the cap is 30, so
    duals below 30 hold the mode even though the old resume
    threshold (25) is satisfied.
    """
    assert should_exit_collect(_make_ctx(dual_count=30, radar_count=30)) is True
    assert should_exit_collect(_make_ctx(dual_count=25, radar_count=25)) is False
    assert should_exit_collect(_make_ctx(dual_count=5, radar_count=5)) is False


def test_radar_at_break_enters_recover_equipment_to_restock() -> None:
    """Radars at the break threshold enter restock even with full weapons.

    Radars find enemies and equipment, so the bot rebuilds the kit
    before hunting blind. The grid-sweep forager makes this safe at
    zero extras (it spends none), reversing the conservative exclusion
    that left the bot looping 0->3->2->1 (live run 20260613-011044).
    """
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=5)) is True


def test_radars_below_the_cap_floor_trigger_restock_between_kills() -> None:
    """Radar counts below cap-5 re-enter recovery between kills.

    User contract 2026-07-25: the between-kills bar is the rank cap
    (30 at rank 2, radar floor 25) -- the old fixed resume threshold
    (20) under-restocked high ranks. The bot rebuilds a genuinely
    full kit before every engagement cycle.
    """
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=6)) is True
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=24)) is True
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=25)) is False


def test_exit_recover_equipment_requires_radars_within_five_of_cap() -> None:
    """Full weapons do NOT release recovery while radars stay low.

    The mode holds until extra radars are within 5 of the rank cap
    (cap 30 at rank 2, so the floor is 25), so the bot reaches a
    genuinely full kit before returning to the hunt instead of
    leaving at the first radar it scrapes together.
    """
    assert should_exit_collect(_make_ctx(dual_count=30, radar_count=5)) is False
    assert should_exit_collect(_make_ctx(dual_count=30, radar_count=24)) is False
    assert should_exit_collect(_make_ctx(dual_count=30, radar_count=25)) is True


def test_hunt_fuel_floor_is_the_rank_fuel_capacity() -> None:
    """The full-fuel floor is exactly what the rank's tank holds.

    User ruling 2026-07-25: "just determine max fuel based on the
    tank rank". A recruit is hunt-ready at their genuine full tank
    of 1000; rank 2 needs its full 1200. An unreachable fixed floor
    would trap low ranks in COLLECT forever.
    """
    recruit_ctx = _make_ctx(fuel=1000)
    recruit_ctx.self_state["rank"] = 0
    assert hunt_fuel_floor(recruit_ctx) == 1000
    assert should_enter_hunt(recruit_ctx) is True
    assert hunt_fuel_floor(_make_ctx(fuel=1200)) == 1200


def test_should_enter_hunt_requires_full_fuel_and_full_stock() -> None:
    """HUNT entry is a privilege of a full tank (contract 2026-07-25).

    Fuel below the rank's capacity (1200 at rank 2) refuses entry
    even with a perfect inventory; a full tank with weapons below
    cap refuses too.
    """
    assert should_enter_hunt(_make_ctx(fuel=1200, dual_count=30, radar_count=30)) is True
    assert should_enter_hunt(_make_ctx(fuel=700, dual_count=30, radar_count=30)) is False
    assert should_enter_hunt(_make_ctx(fuel=1200, dual_count=25, radar_count=30)) is False
    assert should_enter_hunt(_make_ctx(fuel=150, dual_count=30, radar_count=30)) is False


def test_should_exit_hunt_when_recovery_takes_priority() -> None:
    """HUNT exits when a COLLECT trigger fires.

    Between kills (no lock) a non-full tank releases the hunt for a
    restock; a full tank holds it.
    """
    assert should_exit_hunt(_make_ctx(fuel=1200, dual_count=30, radar_count=30)) is False
    assert should_exit_hunt(_make_ctx(fuel=700, dual_count=30, radar_count=30)) is True
    assert should_exit_hunt(_make_ctx(fuel=150, dual_count=30, radar_count=30)) is True


def test_derive_hunt_mode_state_map_open_without_lock_acquires() -> None:
    """A non-find_enemies map open with no locked target derives ACQUIRE."""
    acquiring = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 800, 0, 0, "dot_relay"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    assert derive_hunt_mode_state(acquiring) == "ACQUIRE"


def test_derive_hunt_mode_state_maps_unlocked_map_open_to_acquire() -> None:
    """A map_open without a locked target and a non-search reason derives ACQUIRE.

    Defensive: production map_opens during HUNT carry either the
    ``find_enemies`` reason (acquire search) or a locked target
    (REFRESH). A map_open with neither must still land in ACQUIRE.
    """
    decision = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 800, 0, 0, "find_enemies"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "ACQUIRE"


def test_derive_hunt_mode_state_maps_locked_walk_to_close() -> None:
    """A combat walk toward a locked target is a CLOSE transition."""
    decision = make_tick_decision(
        command=make_move_command(103, 100),
        behavior=make_behavior_score("HUNT", 800, 103, 100, "find_target"),
        updated_ai_state=AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 42,
            }
        ),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "CLOSE"


def test_make_hold_decision_preserves_started_ms_when_already_unset() -> None:
    """A UNSET → UNSET transition keeps the earlier ``mode_started_ms``."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "manual_mode": "UNSET",
            "mode": "UNSET",
            "mode_state": "",
            "mode_started_ms": 8000,
        }
    )

    decision = make_hold_decision(
        state, timestamp_ms=15000, fuel=900, inventory=_make_hold_inventory()
    )

    assert decision["updated_ai_state"]["mode_started_ms"] == 8000


def test_apply_dispatch_counters_preserves_untouched_fields() -> None:
    """Applying counters does not clobber unrelated decision fields."""
    decision = _make_decision(
        make_teleport_command(100, 100),
        secondary_command=make_radar_command(),
    )

    updated = apply_dispatch_counters(decision)

    assert updated["command"] == decision["command"]
    assert updated["behavior"] == decision["behavior"]
    assert updated["desired_equipment"] == decision["desired_equipment"]
    assert updated["secondary_command"] == decision["secondary_command"]


def test_weapon_resume_slack_relaxes_the_entry_bar() -> None:
    """A configured slack admits weapons at cap minus the slack.

    Default 0 keeps the verbatim 2026-07-25 full-stock contract (the
    cap-25 fixture above refuses at dual_count=25 against cap 30);
    slack 5 admits that same 25 -- mirroring the radar cap-5 shape --
    because equipment has no map atlas and an exact-cap bar forces a
    hop-scan discovery loop after every kill (nine HUD flags,
    2026-07-29 session).
    """
    from tankpit_bot import _test_hooks
    from tests.conftest import FakeEnv

    original = _test_hooks.get_env
    _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_WEAPON_RESUME_SLACK": "5"})
    try:
        assert should_enter_hunt(_make_ctx(fuel=1200, dual_count=25, radar_count=30)) is True
        assert should_enter_hunt(_make_ctx(fuel=1200, dual_count=24, radar_count=30)) is False
    finally:
        _test_hooks.get_env = original


class TestGathererRoleGate:
    """A gatherer's ticks never permit hunting ([[fleet-coordination]])."""

    def test_gatherer_inventory_never_permits_hunt(self) -> None:
        """Full stock notwithstanding, the gatherer role bars HUNT entry.

        Fleet ruling 2026-08-14: the gatherer roams and publishes for
        the fighters of its color; every yield-to-hunt gesture funnels
        through this predicate, so this single bar disables them all.
        """
        from tankpit_bot.bot.ai.mode_gates import hunt_entry_permitted
        from tankpit_bot.bot.ai.types import AIConfigDict, AIStateDict

        ctx = _make_ctx()
        assert hunt_entry_permitted(ctx) is True

        gatherer_state = AIStateDict(
            **{
                **ctx.ai_state,
                "config": AIConfigDict(**{**ctx.config, "role": "gatherer"}),
            }
        )
        gatherer_ctx = DecideCtx(
            ctx.world,
            ctx.self_state,
            gatherer_state,
            ctx.inventory,
            ctx.timestamp_ms,
            None,
            "",
            ws=ctx.ws,
        )
        assert hunt_entry_permitted(gatherer_ctx) is False


class TestWartimeReadinessFloor:
    """The 80%/50% wartime bar (operator ruling 2026-09-01).

    The yuppler/TESLA case ("he shouldve engaged the person but he
    kept just farming radars"): while a consented human is alive on
    the map, HUNT entry needs 80% of the weapon caps and half the
    radar cap instead of the full peacetime bar. Fuel stays governed
    by ``should_enter_hunt`` unchanged.
    """

    def _war_ctx(
        self,
        *,
        dual_count: int,
        radar_count: int,
        consented: bool = True,
        human_alive: bool = True,
        human_fresh: bool = True,
        doctrine: EngagementDoctrine = "skirmish",
    ) -> DecideCtx:
        """A rank-2 ctx (caps 30) with one enemy human plus arm-coverage tanks."""
        from tankpit_bot.sniffer.world_service import WorldService
        from tankpit_bot.state.types import TankStateDict, make_tank_state
        from tankpit_bot.types.constants import TankLiveness
        from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world

        def _resident(
            tank_id: int,
            team: int,
            name: str,
            *,
            is_self: bool = False,
            liveness: TankLiveness = "alive",
            stamp: int = 100000,
        ) -> TankStateDict:
            return make_tank_state(
                tank_id=tank_id,
                x=100 + tank_id,
                y=100,
                team=team,
                rank=1,
                name=name,
                is_self=is_self,
                is_bot=False,
                damage_state=3,
                liveness=liveness,
                timestamp_ms=stamp,
            )

        ws = WorldService()
        tanks = {
            # Every skip arm of human_war_is_live gets a resident:
            "1": _resident(1, 0, "SelfTank", is_self=True),
            "2": _resident(2, 0, "AllyHuman"),
            "3": _resident(3, 2, "Corpse", liveness="deactivated"),
            "4": _resident(4, 2, "red-4"),
            "5": _resident(5, 2, "Silent"),
            "7": _resident(7, 2, "Departed", stamp=0),
            "60": _resident(
                60,
                2,
                "TESLA",
                liveness="alive" if human_alive else "deactivated",
                stamp=100000 if human_fresh else 0,
            ),
        }
        world, self_state = make_world(fuel=1200, tanks=tanks)
        if consented:
            ws.fleet_consented_tank_ids = {60}
        inventory = make_inventory(default_count=30, dual_count=dual_count)
        inventory["homing_shots"]["count"] = dual_count
        inventory["extra_radars"]["count"] = radar_count
        state = make_scanned_ai_state()
        state["config"]["doctrine"] = doctrine
        return DecideCtx(world, self_state, state, inventory, 100000, None, "", ws=ws)

    def test_war_floor_admits_the_eighty_fifty_bot(self) -> None:
        """24/30 weapons and 15/30 radars clear the wartime bar."""
        from tankpit_bot.bot.ai.mode_gates import human_war_is_live, hunt_entry_permitted

        ctx = self._war_ctx(dual_count=24, radar_count=15)

        assert human_war_is_live(ctx) is True
        assert hunt_entry_permitted(ctx) is True

    def test_war_floor_still_bars_below_eighty_percent(self) -> None:
        """23/30 weapons is under the 80% floor even at wartime."""
        from tankpit_bot.bot.ai.mode_gates import hunt_entry_permitted

        assert hunt_entry_permitted(self._war_ctx(dual_count=23, radar_count=15)) is False
        assert hunt_entry_permitted(self._war_ctx(dual_count=24, radar_count=14)) is False

    def test_unconsented_human_keeps_the_peacetime_bar(self) -> None:
        """No consent, no war: 24/30 fails the full peacetime bar."""
        from tankpit_bot.bot.ai.mode_gates import human_war_is_live, hunt_entry_permitted

        ctx = self._war_ctx(dual_count=24, radar_count=15, consented=False)

        assert human_war_is_live(ctx) is False
        assert hunt_entry_permitted(ctx) is False

    def test_dead_consented_human_ends_the_war(self) -> None:
        """A deactivated human is not a live war: peacetime bar returns."""
        from tankpit_bot.bot.ai.mode_gates import human_war_is_live, hunt_entry_permitted

        ctx = self._war_ctx(dual_count=24, radar_count=15, human_alive=False)

        assert human_war_is_live(ctx) is False
        assert hunt_entry_permitted(ctx) is False

    def test_logged_out_consented_human_ends_the_war(self) -> None:
        """A stale registry ghost (the logout case) is not a live war.

        A logout emits no deactivation — the entry lingers ``alive``
        (the Yuppler ghost precedent) — so presence staleness is what
        releases the fleet back to the peacetime bar.
        """
        from tankpit_bot.bot.ai.mode_gates import human_war_is_live, hunt_entry_permitted

        ctx = self._war_ctx(dual_count=24, radar_count=15, human_fresh=False)

        assert human_war_is_live(ctx) is False
        assert hunt_entry_permitted(ctx) is False

    def test_war_joining_doctrines_only_get_the_wartime_floor(self) -> None:
        """Duelist, passive, AND a pre-muster swarm keep the peacetime bar.

        Doctrine scope (2026-09-01; the pre-muster hole closed by the
        2026-09-02 double-check): the relaxed bar exists to join a
        war. A duelist that cannot claim the duel, a passive bot, and
        a swarm bot whose muster does not stand would all hunt
        practice bots understocked on it.
        """
        from tankpit_bot.bot.ai.mode_gates import hunt_entry_permitted

        for doctrine in ("duelist", "passive", "swarm"):
            ctx = self._war_ctx(dual_count=24, radar_count=15, doctrine=doctrine)
            assert hunt_entry_permitted(ctx) is False, doctrine

    def test_swarm_gets_the_floor_once_the_muster_stands(self) -> None:
        """A war-ready sibling arms the swarm bot's wartime bar."""
        from tankpit_bot.bot.ai.mode_gates import hunt_entry_permitted

        ctx = self._war_ctx(dual_count=24, radar_count=15, doctrine="swarm")
        ctx.ws.fleet_war_ready_count = 1

        assert hunt_entry_permitted(ctx) is True

    def test_swarm_gets_the_floor_when_a_sibling_holds_the_war_human(self) -> None:
        """Reinforcement needs no quorum: an engaged war human arms the bar."""
        from tankpit_bot.bot.ai.mode_gates import hunt_entry_permitted

        ctx = self._war_ctx(dual_count=24, radar_count=15, doctrine="swarm")
        ctx.ws.fleet_engaged_target_ids = {60: 99000}

        assert hunt_entry_permitted(ctx) is True
