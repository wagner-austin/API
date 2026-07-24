"""The bot-return-fire shadow law: obeying and violating timelines."""

from __future__ import annotations

from tankpit_bot.sim.bot_policy import BOT_RETURN_WINDOW_MS
from tankpit_bot.validate.shadow_bot_laws import shadow_bot_return_fire
from tankpit_bot.validate.shadow_timeline import (
    PositionEventDict,
    ShadowTimelineDict,
    ShotEventDict,
)

HUMAN_ID = 7
BOT_ID = 21
OTHER_BOT_ID = 22


def _timeline(
    names: dict[int, str],
    shots: list[ShotEventDict],
    positions: list[PositionEventDict],
) -> ShadowTimelineDict:
    return ShadowTimelineDict(
        session_id="bot-law-test",
        self_id=HUMAN_ID,
        names=names,
        syncs=[],
        kills=[],
        gains=[],
        removals=[],
        exits=[],
        inventories=[],
        shots=shots,
        positions=positions,
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
