"""The break projection's arithmetic: rate, hits-to-kill, floors, bands.

Unit pins over ``assess_engagement_break`` and
``estimate_hits_to_kill`` alone — the hunt-mode flow that consumes
the verdict (latch, escape, close-phase entry) lives in
``test_hunt_break_flow.py`` (split 2026-09-03 at the 600-line
ceiling)."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_break import (
    assess_engagement_break,
    estimate_hits_to_kill,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.world_types import (
    EnemyThreatDict,
    make_enemy_threat,
)
from tankpit_bot.inventory import InventoryState
from tankpit_bot.sniffer.world_service import WorldService
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


def _threat(*, damage_state: int, rank: int = 1, name: str = "red-7") -> EnemyThreatDict:
    # A practice-bot name: the human-fight break band (2026-07-31)
    # keys on the name-shape classifier, so the plain projection pins
    # must use a bot-shaped target.
    return make_enemy_threat(
        tank_id=50,
        x=120,
        y=100,
        distance=20,
        damage_state=damage_state,
        rank=rank,
        team=2,
        name=name,
        is_bot=True,
    )


def _ctx(*, fuel: int, rank: int = 2, inventory: InventoryState | None = None) -> DecideCtx:
    ws = WorldService()
    world, self_state = make_world(fuel=fuel, rank=rank)
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory if inventory is not None else make_inventory(),
        100000,
        None,
        "",
        ws=ws,
    )


def test_a_private_at_full_tank_holds_the_flag_one_fight() -> None:
    """The measured 2026-09-01 full-tank break no longer fires.

    Run bot-20260901-210631 21:43:44 ([[flag-triage-20260902]] row 6):
    fuel 1100/1100 at PRIVATE, incoming 270 fuel over 6 hits in the
    window (rate 54), hits_to_kill 10 — projected 360 against the flat
    lieutenant-tuned floor 408 broke a winnable fight at literally
    full fuel. The rank-scaled floor is 157 + 78 + 108 = 343, and the
    projection clears it: the fight is finished, not fled.
    """
    assessment = assess_engagement_break(
        _ctx(fuel=1100, rank=1),
        _threat(damage_state=2, rank=1),
        6,
        270,
    )

    assert assessment["incoming_rate_per_tick"] == 54
    assert assessment["hits_to_kill"] == 10
    assert assessment["projected_fuel_at_kill"] == 1100 - 10 * 74
    assert assessment["escape_floor"] == 157 + 78 + 108
    assert assessment["break_engagement"] is False


def test_the_same_fight_at_the_reference_rank_still_breaks() -> None:
    """The scaling is the difference, not a loosened law.

    Identical inputs at rank 4 keep the exact pre-2026-09-03 floor
    (200 + 100 + 108 = 408 > 360): a lieutenant judging this fight by
    its own capacity still walks away, exactly as tuned.
    """
    assessment = assess_engagement_break(
        _ctx(fuel=1100, rank=4),
        _threat(damage_state=2, rank=1),
        6,
        270,
    )

    assert assessment["escape_floor"] == 200 + 100 + 108
    assert assessment["break_engagement"] is True


def test_hits_to_kill_scales_with_the_quartile() -> None:
    """Near-death rounds to 4 dual hits at private cap; full health to 13."""
    ctx = _ctx(fuel=1100)
    assert estimate_hits_to_kill(ctx, _threat(damage_state=0)) == 4
    assert estimate_hits_to_kill(ctx, _threat(damage_state=3)) == 13


def test_hits_to_kill_falls_back_to_homing_damage_without_duals() -> None:
    """At zero duals the per-hit damage is homing's 45."""
    ctx = _ctx(fuel=1100, inventory=make_inventory(dual_count=0))
    assert estimate_hits_to_kill(ctx, _threat(damage_state=0)) == 7


def test_full_health_target_under_sustained_fire_breaks() -> None:
    """13 hits to kill at 100/tick incoming projects a strand -> break."""
    assessment = assess_engagement_break(_ctx(fuel=1100), _threat(damage_state=3), 5, 500)
    assert assessment["incoming_rate_per_tick"] == 100
    assert assessment["hits_to_kill"] == 13
    assert assessment["projected_fuel_at_kill"] == 1100 - 13 * 120
    # Rank-scaled reserves (row 6): the rank-2 fixture reads the
    # rank-4 reference tuning at capacity 1200/1400 — 171 + 85 —
    # plus the 2-tick escape-latency exposure at rate 100.
    assert assessment["escape_floor"] == 171 + 85 + 200
    assert assessment["break_engagement"] is True


def test_near_death_target_under_the_same_fire_is_finished() -> None:
    """4 hits to kill projects clear of the floor -> keep fighting."""
    assessment = assess_engagement_break(_ctx(fuel=1100), _threat(damage_state=0), 5, 500)
    assert assessment["hits_to_kill"] == 4
    assert assessment["projected_fuel_at_kill"] == 1100 - 4 * 120
    assert assessment["break_engagement"] is False


def test_quiet_fight_never_breaks_even_at_low_fuel() -> None:
    """Zero measured incoming leaves the break inert."""
    assessment = assess_engagement_break(_ctx(fuel=400), _threat(damage_state=3), 0, 0)
    assert assessment["incoming_rate_per_tick"] == 0
    assert assessment["break_engagement"] is False


def test_spaced_hits_count_toward_the_measured_rate() -> None:
    """Two spaced hits in the window feed the projection.

    The retired 3-hit floor made SPACED sustained fire invisible:
    arterial's second main-map death (2026-08-26 18:45, Blue Killer)
    traded 952 fuel to 132 across seventeen shots without one break
    because -90 every 4-12 s kept the window at 2 hits and the rate
    counted zero. Every confirmed hit now counts; a single light hit
    still breaks nothing healthy (the projection math is the guard).
    """
    assessment = assess_engagement_break(_ctx(fuel=500), _threat(damage_state=3), 2, 180)
    assert assessment["incoming_rate_per_tick"] == 36
    assert assessment["projected_fuel_at_kill"] == 500 - 13 * (20 + 36)
    assert assessment["break_engagement"] is True


def test_blue_killer_regression_breaks_below_the_human_band() -> None:
    """The second-death shape breaks once below half capacity.

    At 952 the human attrition band correctly holds (fuel above half
    the rank-2 fixture's 1200 capacity); at 560 — below the band —
    the now-counted spaced rate breaks the fight with escape still
    cheap. Live, the rate blindness held the fight to 132 and death.
    """
    holding = assess_engagement_break(
        _ctx(fuel=952), _threat(damage_state=3, name="Blue Killer"), 2, 180
    )
    assert holding["incoming_rate_per_tick"] == 36
    assert holding["break_engagement"] is False

    breaking = assess_engagement_break(
        _ctx(fuel=560), _threat(damage_state=3, name="Blue Killer"), 2, 180
    )
    assert breaking["break_engagement"] is True


def test_human_fight_holds_above_half_capacity() -> None:
    """The human break band (user ruling 2026-07-31) suppresses the break.

    The same fire that breaks a practice-bot fight
    (:func:`test_full_health_target_under_sustained_fire_breaks`)
    holds against a human while fuel is at or above half the rank
    capacity (rank-2 self: 600 of 1200) -- human fights are attrition
    and "does damage then leaves" was the complaint.
    """
    assessment = assess_engagement_break(
        _ctx(fuel=600),
        _threat(damage_state=3, name="Yuppler"),
        5,
        500,
    )
    assert assessment["projected_fuel_at_kill"] == 600 - 13 * 120
    assert assessment["break_engagement"] is False


def test_human_fight_breaks_below_half_capacity() -> None:
    """Below half capacity the normal projection governs humans too."""
    assessment = assess_engagement_break(
        _ctx(fuel=599),
        _threat(damage_state=3, name="Yuppler"),
        5,
        500,
    )
    assert assessment["break_engagement"] is True
