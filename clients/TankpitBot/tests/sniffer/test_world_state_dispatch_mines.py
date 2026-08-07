"""Tests for tunneled mine dispatch into world state.

Placement (including team resolution), the two real-capture cascade
regressions, and detonation removal.
"""

from __future__ import annotations

from tankpit_bot.sniffer.world_state import (
    get_world_service,
)
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update


class TestDispatchTunneledMines:
    """Mine placement, cascade, and detonation dispatch."""

    def test_dispatch_tunneled_mine_placement_adds_mines(self) -> None:
        """Test tunneled 0x4B mine placement updates world mine state."""
        from tankpit_bot.protocol import MovementResponseDict

        dispatch_world_state_update(
            get_world_service(),
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=131,
                y=126,
                direction=8,
                damage_state=0,
                rank=1,
                lb_score=1313,
                carrying=0,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [
                    (131, 126),
                    (131, 125),
                    (132, 125),
                    (132, 126),
                    (132, 127),
                ],
            },
        )

        state = get_world_service().world_state
        assert state["mines"]["131,126"]["team"] == 2
        assert state["mines"]["131,126"]["tank_id"] == 1301
        assert state["mines"]["131,126"]["mine_type"] == 2
        assert state["mines"]["132,127"]["x"] == 132
        assert state["mines"]["132,127"]["y"] == 127

    def test_dispatch_tunneled_mine_placement_uses_known_tank_team(self) -> None:
        """Test tunneled 0x4B uses tracked tank team when placer is not self."""
        from tankpit_bot.protocol import TankEntryDict, TankInfoDict

        dispatch_world_state_update(
            get_world_service(),
            TankInfoDict(
                msg_type=0x21,
                tank_id=777,
                name="placer",
                team=3,
                decoration_state=b"",
                persistent_tank_id=0,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            TankEntryDict(
                msg_type=0x28,
                team=3,
                tank_id=777,
                rank=0,
                damage_state=0,
                score=0,
                x=40,
                y=41,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 1,
                "tank_id": 777,
                "positions": [(40, 41), (40, 42)],
            },
        )

        state = get_world_service().world_state
        assert state["mines"]["40,41"]["team"] == 3
        assert state["mines"]["40,42"]["team"] == 3
        assert state["mines"]["40,41"]["tank_id"] == 777

    def test_dispatch_tunneled_mine_placement_skips_unknown_team(self) -> None:
        """Test tunneled 0x4B does nothing when placer team is unknown."""
        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 9999,
                "positions": [(10, 11), (11, 11)],
            },
        )

        state = get_world_service().world_state
        assert state["mines"] == {}

    def test_mine_on_mine_destruction_real_capture(self) -> None:
        """Regression for the 3x3 placement that destroys adjacent enemy mines.

        Captured 2026-06-20 (practice-vs-real-20260620-150138, t+56.15s):
        Artax (team 2, blue) on the tile center placed 7 blue mines and
        the server simultaneously fired a 0x45 MineDetonation listing 2
        adjacent enemy mines (purple) destroyed by the same placement.
        Total = 9 = full 3x3 attempted around the placer per the game
        mechanic (server filters water / terrain / tanks / enemy mines;
        clear tiles get the mine, enemy-mine tiles get the detonation,
        impossible tiles get nothing).

        Post-state: 7 of 9 tiles have our blue mines; 2 tiles are empty
        (the detonated enemy mines did not become our mines -- per
        user, those gaps require re-placement to fill).
        """
        from tankpit_bot.protocol import MovementResponseDict
        from tankpit_bot.state import add_mine

        ws = get_world_service()
        # Establish self at the placement center.
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=133,
                y=124,
                direction=8,
                damage_state=0,
                rank=1,
                lb_score=1313,
                carrying=0,
            ),
        )
        # Seed the two enemy (purple, team=1) mines that the placement
        # is about to detonate.
        ws.world_state = add_mine(ws.world_state, 132, 123, 2, 1229, 1, 1)
        ws.world_state = add_mine(ws.world_state, 134, 125, 2, 1229, 1, 1)
        assert ws.world_state["mines"]["132,123"]["team"] == 1
        assert ws.world_state["mines"]["134,125"]["team"] == 1

        # Wire packet 1: 7 blue mines placed at the 7 clear tiles in the 3x3.
        dispatch_world_state_update(
            ws,
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [
                    (133, 124),
                    (132, 124),
                    (133, 123),
                    (134, 123),
                    (134, 124),
                    (133, 125),
                    (132, 125),
                ],
            },
        )
        # Wire packet 2 (same wire tick): the 2 enemy-mine tiles get
        # 0x45 MineDetonation -- enemy mines destroyed.
        dispatch_world_state_update(
            ws,
            {"msg_type": 0x45, "positions": [(132, 123), (134, 125)]},
        )

        mines = ws.world_state["mines"]
        # 7 own mines placed.
        own_mine_positions = [
            (133, 124),
            (132, 124),
            (133, 123),
            (134, 123),
            (134, 124),
            (133, 125),
            (132, 125),
        ]
        for x, y in own_mine_positions:
            assert mines[f"{x},{y}"]["team"] == 2
            assert mines[f"{x},{y}"]["tank_id"] == 1301
        # 2 detonated tiles are empty -- no own mine, no enemy mine.
        assert "132,123" not in mines
        assert "134,125" not in mines

    def test_mine_cascade_two_packet_chain_real_capture(self) -> None:
        """Regression for one shot detonating a mine + chain detonation.

        Captured 2026-06-20 (practice-vs-real-20260620-150138, t+62.15s):
        Artax shot tile (134, 126) -- the server emitted two 0x45
        MineDetonate packets in the same wire tick. First packet listed
        the directly hit mine [(134, 126)]; second packet listed the
        6 adjacent chain mines destroyed in the cascade
        [(135, 126), (134, 127), (133, 126), (135, 127), (135, 125),
        (133, 127)]. Total 7 tiles cleared.

        World-state must apply both packets atomically (each removes its
        listed positions) and end with all 7 tiles empty regardless of
        the mines' original team.
        """
        from tankpit_bot.state import add_mine

        ws = get_world_service()
        # Seed the 7 mines that the cascade is about to destroy --
        # mix of own (blue, team=2) and enemy (purple, team=1).
        seed = [
            (134, 126, 2, 1301),  # own, the directly-hit one
            (135, 126, 1, 1229),  # enemy
            (134, 127, 2, 1301),  # own
            (133, 126, 1, 1229),  # enemy
            (135, 127, 1, 1229),  # enemy
            (135, 125, 2, 1301),  # own
            (133, 127, 1, 1229),  # enemy
        ]
        for x, y, team, tid in seed:
            ws.world_state = add_mine(ws.world_state, x, y, 2, tid, team, 1)
        assert len(ws.world_state["mines"]) == 7

        # Wire packet 1: directly hit mine.
        dispatch_world_state_update(
            ws,
            {"msg_type": 0x45, "positions": [(134, 126)]},
        )
        # Wire packet 2 (same wire tick): chain detonation cascade.
        dispatch_world_state_update(
            ws,
            {
                "msg_type": 0x45,
                "positions": [
                    (135, 126),
                    (134, 127),
                    (133, 126),
                    (135, 127),
                    (135, 125),
                    (133, 127),
                ],
            },
        )

        # All 7 tiles empty after the cascade -- own and enemy alike.
        mines = ws.world_state["mines"]
        for x, y, _team, _tid in seed:
            assert f"{x},{y}" not in mines

    def test_dispatch_tunneled_mine_detonation_removes_mines(self) -> None:
        """Test tunneled 0x45 removes mines at decoded coordinates."""
        from tankpit_bot.protocol import MovementResponseDict

        dispatch_world_state_update(
            get_world_service(),
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=38,
                y=53,
                direction=8,
                damage_state=0,
                rank=1,
                lb_score=1313,
                carrying=0,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [(38, 52), (39, 53), (38, 54)],
            },
        )

        dispatch_world_state_update(
            get_world_service(), {"msg_type": 0x45, "positions": [(39, 53), (38, 54)]}
        )

        state = get_world_service().world_state
        assert "38,52" in state["mines"]
        assert "39,53" not in state["mines"]
        assert "38,54" not in state["mines"]
