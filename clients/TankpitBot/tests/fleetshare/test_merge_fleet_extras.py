"""Fleet merge: the mine map and the war-ready count.

Split from ``test_merge.py`` at the 600-line ceiling (2026-09-02).
Both halves are 2026-09-01 operator orders: the mine-aware layer
between the bots, and the swarm muster's quorum signal.
"""

from __future__ import annotations

from tankpit_bot.fleetshare.merge import merge_fleet_reports
from tankpit_bot.fleetshare.types import FleetMineSightingDict
from tankpit_bot.sniffer.world_service import WorldService
from tests.fleetshare.test_merge import _NOW, _report


class TestMergeMineKnowledge:
    """The fleet mine map's receiving half (operator order 2026-09-01)."""

    def _world_service(self) -> WorldService:
        ws = WorldService()
        ws.update_world_state_from_position(90, 90)
        return ws

    def _mine(self, *, observed_ms: int = _NOW - 500) -> FleetMineSightingDict:
        return FleetMineSightingDict(
            x=101, y=100, mine_type=1, tank_id=709, team=1, observed_ms=observed_ms
        )

    def test_sibling_mine_lands_in_the_registry_as_an_import(self) -> None:
        """A shared hostile mine arrives with the fleet import stamp."""
        ws = self._world_service()
        report = _report("artax", mines=[self._mine()])

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["mines"] == 1
        landed = ws.world_state["mines"]["101,100"]
        assert landed["team"] == 1
        assert landed["mine_type"] == 1
        assert landed["source"] == "world_state"
        assert landed["timestamp_ms"] == _NOW - 500

    def test_fresher_local_belief_is_never_outranked(self) -> None:
        """Own wire is the higher trust tier: an older sighting is a no-op."""
        from tankpit_bot.state.mine_mutations import add_mine

        ws = self._world_service()
        ws.world_state = add_mine(
            ws.world_state, 101, 100, mine_type=0, tank_id=-1, team=1, timestamp_ms=_NOW - 100
        )
        report = _report("artax", mines=[self._mine(observed_ms=_NOW - 500)])

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["mines"] == 0
        assert ws.world_state["mines"]["101,100"]["source"] == "viewport"

    def test_fresher_sighting_refreshes_an_older_local_belief(self) -> None:
        """Newer remote knowledge advances the tile's stamp."""
        from tankpit_bot.state.mine_mutations import add_mine

        ws = self._world_service()
        ws.world_state = add_mine(
            ws.world_state, 101, 100, mine_type=1, tank_id=709, team=1, timestamp_ms=_NOW - 900
        )
        report = _report("artax", mines=[self._mine(observed_ms=_NOW - 500)])

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["mines"] == 1
        assert ws.world_state["mines"]["101,100"]["timestamp_ms"] == _NOW - 500


class TestWarReadyCount:
    """The swarm muster's quorum input is a wholesale-replaced count."""

    def _world_service(self) -> WorldService:
        ws = WorldService()
        ws.update_world_state_from_position(90, 90)
        return ws

    def test_ready_siblings_are_counted_and_replaced(self) -> None:
        """Two ready of three reports counts 2; the next pass replaces it."""
        ws = self._world_service()
        reports = [
            _report("artax", war_ready=True),
            _report("yuppler", war_ready=True),
            _report("despair", war_ready=False),
        ]

        merge_fleet_reports(ws, reports, own_tank_id=2731, own_team=2)
        assert ws.fleet_war_ready_count == 2

        merge_fleet_reports(ws, [_report("artax")], own_tank_id=2731, own_team=2)
        assert ws.fleet_war_ready_count == 0
