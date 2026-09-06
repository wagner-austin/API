"""Law 10 — the client's rank, and the 0x2B frames that announce it."""

from __future__ import annotations

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.progression import (
    DEMOTED_RANK,
    PROMOTED_RANK,
    PROMOTION_RECOVERY_TICKS,
    RECOVERING_PROMO_STATE,
    STEADY_PROMO_STATE,
    RankProgression,
)
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _world() -> SimWorldDict:
    """A world whose client tank is a rank-1 private at capacity."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, PROMOTED_RANK, 10, 10, fuel_capacity(PROMOTED_RANK))
    return world


def test_the_bar_rests_at_the_wires_steady_value() -> None:
    """10, not 0: 62,209 of 62,528 own-tank frames carry it.

    The sim emitted 0, which the real server carries on 189
    ([[session-state-deglobalisation]]).
    """
    assert RankProgression(9).promo_state == STEADY_PROMO_STATE


def test_a_deactivation_demotes_the_client_silently() -> None:
    """Rank drops to recruit with no banner, three archived times out
    of three, at a zero-second gap from the 0x41."""
    world = _world()
    progression = RankProgression(9)
    messages: list[BinaryMessage] = []

    progression.note_deactivation(world, messages)

    assert messages == [{"msg_type": 0x2B, "new_rank": DEMOTED_RANK, "was_promoted": False}]
    assert world["tanks"][9]["rank"] == DEMOTED_RANK
    assert progression.promo_state == RECOVERING_PROMO_STATE


def test_the_demotion_clamps_fuel_to_the_lower_capacity() -> None:
    """A recruit cannot hold a private's tank."""
    world = _world()
    progression = RankProgression(9)

    progression.note_deactivation(world, [])

    assert world["tanks"][9]["fuel"] == fuel_capacity(DEMOTED_RANK)


def test_a_second_deactivation_while_demoted_says_nothing() -> None:
    """You cannot be demoted from recruit twice."""
    world = _world()
    progression = RankProgression(9)
    progression.note_deactivation(world, [])
    messages: list[BinaryMessage] = []

    progression.note_deactivation(world, messages)

    assert messages == []


def test_recovery_promotes_the_client_back_with_the_banner() -> None:
    """``new_rank=1, was_promoted=True`` — the archived banner."""
    world = _world()
    progression = RankProgression(9)
    progression.note_deactivation(world, [])

    world["tick"] = PROMOTION_RECOVERY_TICKS - 1
    early: list[BinaryMessage] = []
    progression.advance(world, early)
    assert early == []

    world["tick"] = PROMOTION_RECOVERY_TICKS
    late: list[BinaryMessage] = []
    progression.advance(world, late)

    assert late == [{"msg_type": 0x2B, "new_rank": PROMOTED_RANK, "was_promoted": True}]
    assert world["tanks"][9]["rank"] == PROMOTED_RANK
    assert progression.promo_state == STEADY_PROMO_STATE


def test_an_undemoted_client_never_promotes() -> None:
    """No demotion, no recovery — the bar just rests."""
    world = _world()
    progression = RankProgression(9)
    world["tick"] = PROMOTION_RECOVERY_TICKS * 4
    messages: list[BinaryMessage] = []

    progression.advance(world, messages)

    assert messages == []
    assert world["tanks"][9]["rank"] == PROMOTED_RANK


def test_the_server_demotes_on_the_clients_own_deactivation() -> None:
    """The tick processor reads its own 0x41 and acts on it."""
    from tankpit_bot.sim.combat import SLOT_DUAL
    from tankpit_bot.sim.commands import ClientCommandDict
    from tankpit_bot.sim.server import SimServer

    world = _world()
    world["tanks"][9]["fuel"] = 40
    world["tanks"][11] = make_sim_tank(11, 1, 8, 12, 10, 1800)
    world["tanks"][11]["counts"][SLOT_DUAL] = 3
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    server.queue_command(
        11,
        ClientCommandDict(
            kind="shoot",
            command=115,
            x=10,
            y=10,
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
            amount=0,
        ),
    )

    batch = server.advance_tick()

    kinds = [message["msg_type"] for message in batch]
    assert 0x41 in kinds
    assert kinds.index(0x41) < kinds.index(0x2B)
    assert world["tanks"][9]["rank"] == DEMOTED_RANK


def test_another_tanks_deactivation_leaves_the_clients_rank_alone() -> None:
    """Only the CLIENT's own death demotes the client."""
    from tankpit_bot.sim.combat import SLOT_DUAL
    from tankpit_bot.sim.commands import ClientCommandDict
    from tankpit_bot.sim.server import SimServer

    world = _world()
    world["tanks"][9]["counts"][SLOT_DUAL] = 3
    world["tanks"][11] = make_sim_tank(11, 1, 1, 12, 10, 40)
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    server.queue_command(
        9,
        ClientCommandDict(
            kind="shoot",
            command=115,
            x=12,
            y=10,
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
            amount=0,
        ),
    )

    batch = server.advance_tick()

    assert 0x41 in [message["msg_type"] for message in batch]
    assert 0x2B not in [message["msg_type"] for message in batch]
    assert world["tanks"][9]["rank"] == PROMOTED_RANK


def test_the_sync_cadence_carries_the_resting_bar() -> None:
    """Every status sync reports the bar, and it rests at 10."""
    from tankpit_bot.sim.server import SimServer

    world = _world()
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)

    syncs = [message for message in server.advance_tick() if message["msg_type"] == 0x2E]

    assert syncs != []
    assert {sync["promo_state"] for sync in syncs} == {STEADY_PROMO_STATE}
