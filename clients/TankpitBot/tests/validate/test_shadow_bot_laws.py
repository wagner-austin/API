"""The bot-return-fire shadow law: obeying and violating timelines."""

from __future__ import annotations

from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.sim.bot_policy import BOT_RETURN_WINDOW_MS
from tankpit_bot.sim.server import CORPSE_WINDOW_TICKS
from tankpit_bot.validate.shadow_bot_laws import (
    shadow_bot_reactivation,
    shadow_bot_return_fire,
)
from tankpit_bot.validate.shadow_timeline import (
    KillEventDict,
    PositionEventDict,
    ShadowTimelineDict,
    ShotEventDict,
    TankSyncEventDict,
)

CORPSE_MS = CORPSE_WINDOW_TICKS * TICK_RATE_MS

HUMAN_ID = 7
BOT_ID = 21
OTHER_BOT_ID = 22


def _timeline(
    names: dict[int, str],
    shots: list[ShotEventDict] | None = None,
    positions: list[PositionEventDict] | None = None,
    kills: list[KillEventDict] | None = None,
    syncs: list[TankSyncEventDict] | None = None,
) -> ShadowTimelineDict:
    return ShadowTimelineDict(
        session_id="bot-law-test",
        self_id=HUMAN_ID,
        names=names,
        syncs=syncs if syncs is not None else [],
        kills=kills if kills is not None else [],
        gains=[],
        removals=[],
        exits=[],
        inventories=[],
        shots=shots if shots is not None else [],
        positions=positions if positions is not None else [],
    )


def _kill(timestamp_ms: int, victim_id: int) -> KillEventDict:
    return KillEventDict(
        timestamp_ms=timestamp_ms,
        victim_id=victim_id,
        killer_id=HUMAN_ID,
        is_mine_kill=False,
    )


def _sync(timestamp_ms: int, tank_id: int, damage_state: int, rank: int = 0) -> TankSyncEventDict:
    return TankSyncEventDict(
        timestamp_ms=timestamp_ms,
        tank_id=tank_id,
        damage_state=damage_state,
        rank=rank,
        fuel=None,
    )


def _names() -> dict[int, str]:
    return {HUMAN_ID: "Artax", BOT_ID: "orange-3", OTHER_BOT_ID: "orange-4"}


def _pos(timestamp_ms: int, tank_id: int, x: int, y: int) -> PositionEventDict:
    return PositionEventDict(timestamp_ms=timestamp_ms, tank_id=tank_id, x=x, y=y)


def _shot(
    timestamp_ms: int,
    shooter_id: int,
    source: tuple[int, int],
    target: tuple[int, int],
    weapon: int = 0,
) -> ShotEventDict:
    return ShotEventDict(
        timestamp_ms=timestamp_ms,
        shooter_id=shooter_id,
        source_x=source[0],
        source_y=source[1],
        target_x=target[0],
        target_y=target[1],
        weapon=weapon,
    )


def _provoked_return(return_weapon: int = 0, delay_ms: int = 2000) -> ShadowTimelineDict:
    """Human at (10,10) hits the bot at (11,10); the bot returns."""
    return _timeline(
        _names(),
        shots=[
            _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
            _shot(1000 + delay_ms, BOT_ID, (11, 10), (10, 10), weapon=return_weapon),
        ],
        positions=[_pos(0, HUMAN_ID, 10, 10), _pos(0, BOT_ID, 11, 10)],
    )


class TestBotReturnFire:
    def test_provoked_single_at_the_attacker_is_exact(self) -> None:
        evidence = shadow_bot_return_fire([_provoked_return()])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_wrong_weapon_is_a_mismatch(self) -> None:
        evidence = shadow_bot_return_fire([_provoked_return(return_weapon=1)])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_late_return_outside_the_window_is_a_mismatch(self) -> None:
        evidence = shadow_bot_return_fire([_provoked_return(delay_ms=BOT_RETURN_WINDOW_MS + 1000)])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_unprovoked_bot_shot_is_a_mismatch(self) -> None:
        timeline = _timeline(
            _names(),
            shots=[_shot(1000, BOT_ID, (11, 10), (10, 10))],
            positions=[_pos(0, BOT_ID, 11, 10)],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_wrong_aim_tile_is_a_mismatch(self) -> None:
        timeline = _timeline(
            _names(),
            shots=[
                _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
                _shot(3000, BOT_ID, (11, 10), (12, 12)),
            ],
            positions=[_pos(0, HUMAN_ID, 10, 10), _pos(0, BOT_ID, 11, 10)],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_unknown_attacker_position_passes_the_aim_clause(self) -> None:
        """The attacker never stated a position: weapon + window judge."""
        timeline = _timeline(
            _names(),
            shots=[
                _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
                _shot(3000, BOT_ID, (11, 10), (10, 10)),
            ],
            positions=[_pos(0, BOT_ID, 11, 10)],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_human_shots_are_not_samples(self) -> None:
        timeline = _timeline(
            _names(),
            shots=[_shot(1000, HUMAN_ID, (10, 10), (11, 10))],
            positions=[_pos(0, BOT_ID, 11, 10)],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert evidence["samples"] == 0

    def test_sessions_without_bots_are_skipped(self) -> None:
        timeline = _timeline(
            {HUMAN_ID: "Artax"},
            shots=[_shot(1000, HUMAN_ID, (10, 10), (11, 10))],
            positions=[],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert evidence["samples"] == 0

    def test_bot_on_bot_hit_attribution_excludes_the_shooter(self) -> None:
        """A bot shot at its own tile is not a hit on itself; an unhit,
        unpositioned other bot never records a hit."""
        timeline = _timeline(
            _names(),
            shots=[
                _shot(1000, BOT_ID, (11, 10), (11, 10)),
                _shot(2000, BOT_ID, (11, 10), (10, 10)),
            ],
            positions=[_pos(0, BOT_ID, 11, 10)],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (2, 0)

    def test_position_updates_move_the_aim_target(self) -> None:
        """The aim clause judges the attacker's LATEST tile."""
        timeline = _timeline(
            _names(),
            shots=[
                _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
                _shot(3000, BOT_ID, (11, 10), (10, 11)),
            ],
            positions=[
                _pos(0, HUMAN_ID, 10, 10),
                _pos(0, BOT_ID, 11, 10),
                _pos(2000, HUMAN_ID, 10, 11),
            ],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)


class TestBotReactivation:
    def test_full_tier_sync_after_the_corpse_window_is_exact(self) -> None:
        timeline = _timeline(
            _names(),
            kills=[_kill(1000, BOT_ID)],
            syncs=[_sync(1000 + CORPSE_MS, BOT_ID, 3)],
        )
        evidence = shadow_bot_reactivation([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_early_sync_is_a_mismatch(self) -> None:
        timeline = _timeline(
            _names(),
            kills=[_kill(1000, BOT_ID)],
            syncs=[_sync(5000, BOT_ID, 3)],
        )
        evidence = shadow_bot_reactivation([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_partial_fuel_return_is_a_mismatch(self) -> None:
        timeline = _timeline(
            _names(),
            kills=[_kill(1000, BOT_ID)],
            syncs=[_sync(1000 + CORPSE_MS, BOT_ID, 1)],
        )
        evidence = shadow_bot_reactivation([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_late_damaged_resight_is_unobserved_not_a_mismatch(self) -> None:
        """A first re-sight far past the corpse window judges nothing.

        The bot reactivated at full OFF-viewport and was damaged by
        someone else before drifting back into view — 34/35 of the
        2026-08-03 sweep's "failures" were exactly this shape (gaps to
        1,047 s, all non-full). The reactivation moment was unobserved.
        """
        timeline = _timeline(
            _names(),
            kills=[_kill(1000, BOT_ID)],
            syncs=[_sync(1000 + CORPSE_MS + 300_000, BOT_ID, 1)],
        )
        evidence = shadow_bot_reactivation([timeline])
        assert evidence["samples"] == 0

    def test_death_with_no_later_sync_is_skipped(self) -> None:
        timeline = _timeline(
            _names(),
            kills=[_kill(1000, BOT_ID)],
            syncs=[_sync(500, BOT_ID, 2)],
        )
        evidence = shadow_bot_reactivation([timeline])
        assert evidence["samples"] == 0

    def test_non_bot_victims_are_not_samples(self) -> None:
        timeline = _timeline(
            _names(),
            kills=[_kill(1000, HUMAN_ID)],
            syncs=[_sync(1000 + CORPSE_MS, HUMAN_ID, 3)],
        )
        evidence = shadow_bot_reactivation([timeline])
        assert evidence["samples"] == 0

    def test_sessions_without_bots_are_skipped(self) -> None:
        timeline = _timeline(
            {HUMAN_ID: "Artax"},
            kills=[_kill(1000, HUMAN_ID)],
        )
        evidence = shadow_bot_reactivation([timeline])
        assert evidence["samples"] == 0


ALLY_BOT_ID = 30


class TestTeamAggro:
    """The gang-up and assist reflexes (sim/bot_policy team aggro)."""

    def test_gang_up_at_the_sighted_attacker_is_exact(self) -> None:
        """A teammate of the hit bot, within sight, avenges at the
        attacker's tile — one of the 48 archive gang-up shots."""
        timeline = _timeline(
            _names(),
            shots=[
                _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
                _shot(3000, OTHER_BOT_ID, (15, 10), (10, 10)),
            ],
            positions=[
                _pos(0, HUMAN_ID, 10, 10),
                _pos(0, BOT_ID, 11, 10),
                _pos(0, OTHER_BOT_ID, 15, 10),
            ],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_assist_at_the_engaged_enemy_bot_is_exact(self) -> None:
        """A bot on the attacker's SIDE, within sight of the victim,
        joins against it — the live blue-7 shape."""
        names = _names()
        names[ALLY_BOT_ID] = "blue-1"
        timeline = _timeline(
            names,
            shots=[
                _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
                _shot(3000, ALLY_BOT_ID, (14, 10), (11, 10)),
            ],
            positions=[
                _pos(0, HUMAN_ID, 10, 10),
                _pos(0, BOT_ID, 11, 10),
                _pos(0, ALLY_BOT_ID, 14, 10),
            ],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_out_of_sight_avenger_is_a_mismatch(self) -> None:
        """The reflex is sight-gated at AGGRO_SIGHT_RADIUS (129/129
        archive shots within 8 tiles): a far teammate never joins."""
        timeline = _timeline(
            _names(),
            shots=[
                _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
                _shot(3000, OTHER_BOT_ID, (30, 10), (10, 10)),
            ],
            positions=[
                _pos(0, HUMAN_ID, 10, 10),
                _pos(0, BOT_ID, 11, 10),
                _pos(0, OTHER_BOT_ID, 30, 10),
            ],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_unpositioned_shooter_cannot_claim_team_aggro(self) -> None:
        """Without the shooter's tile the sight gate cannot pass."""
        timeline = _timeline(
            _names(),
            shots=[
                _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
                _shot(3000, OTHER_BOT_ID, (15, 10), (10, 10)),
            ],
            positions=[_pos(0, HUMAN_ID, 10, 10), _pos(0, BOT_ID, 11, 10)],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_gang_up_wrong_aim_tile_is_a_mismatch(self) -> None:
        """A teammate was hit, but the shot ignores the attacker's
        known tile — not the mined reflex."""
        timeline = _timeline(
            _names(),
            shots=[
                _shot(1000, HUMAN_ID, (10, 10), (11, 10)),
                _shot(3000, OTHER_BOT_ID, (15, 10), (16, 10)),
            ],
            positions=[
                _pos(0, HUMAN_ID, 10, 10),
                _pos(0, BOT_ID, 11, 10),
                _pos(0, OTHER_BOT_ID, 15, 10),
            ],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)

    def test_shot_at_an_unengaged_enemy_bot_is_a_mismatch(self) -> None:
        """Assist requires the target bot to have been hit recently —
        bots never open fire on a calm enemy bot."""
        names = _names()
        names[ALLY_BOT_ID] = "blue-1"
        timeline = _timeline(
            names,
            shots=[_shot(3000, ALLY_BOT_ID, (14, 10), (11, 10))],
            positions=[_pos(0, BOT_ID, 11, 10), _pos(0, ALLY_BOT_ID, 14, 10)],
        )
        evidence = shadow_bot_return_fire([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 0)
