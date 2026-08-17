"""Tests for building and writing the fleet knowledge report."""

from __future__ import annotations

from platform_core.json_utils import load_json_str

from tankpit_bot.fleetshare.codecs import decode_fleet_report
from tankpit_bot.fleetshare.report import (
    ENEMY_SIGHTING_TTL_MS,
    FLEET_REPORT_FILENAME,
    build_fleet_report,
    write_fleet_report,
)
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    ContainerStateDict,
    TankStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
    make_tank_state,
)
from tests.conftest import FakeFileSystem

_NOW = 100000


def _world_service(
    tanks: dict[str, TankStateDict] | None = None,
    containers: dict[str, ContainerStateDict] | None = None,
) -> WorldService:
    """Build a session with an established self at (100,100), team 2."""
    ws = WorldService()
    ws.update_world_state_from_position(100, 100)
    ws.world_state = WorldStateDict(
        **{
            **ws.world_state,
            "self_state": make_self_state(
                tank_id=2731,
                x=100,
                y=100,
                team=2,
                rank=1,
                fuel=900,
                leaderboard_position=0,
            ),
            "tanks": dict(tanks) if tanks else {},
            "containers": dict(containers) if containers else {},
        }
    )
    return ws


def _enemy(tank_id: int, *, team: int = 0, observed_ms: int = _NOW - 1000) -> TankStateDict:
    """Build a live, position-fresh tank record."""
    return make_tank_state(
        tank_id=tank_id,
        x=50,
        y=60,
        team=team,
        rank=1,
        name=f"red-{tank_id % 10}",
        is_self=False,
        is_bot=False,
        damage_state=2,
        timestamp_ms=observed_ms,
        last_wire_seen_ms=observed_ms,
        last_position_update_ms=observed_ms,
        last_viewport_observation_ms=observed_ms,
    )


class TestBuildFleetReport:
    """What the report offers, and what it withholds."""

    def test_no_established_self_builds_nothing(self) -> None:
        """Before entry there is nothing attributable to offer."""
        ws = WorldService()

        assert (
            build_fleet_report(
                ws, instance="artax", role="fighter", engaged_target_id=-1, now_ms=_NOW
            )
            is None
        )

    def test_report_carries_identity_lock_and_fresh_enemies(self) -> None:
        """A fresh enemy sighting rides the report with its own age."""
        ws = _world_service(tanks={"506": _enemy(506)})

        report = build_fleet_report(
            ws, instance="arterial", role="fighter", engaged_target_id=506, now_ms=_NOW
        )

        if report is None:
            raise AssertionError("expected a report")
        assert report["instance"] == "arterial"
        assert report["team"] == 2
        assert report["tank_id"] == 2731
        assert report["engaged_target_id"] == 506
        assert report["written_ms"] == _NOW
        assert len(report["enemies"]) == 1
        sighting = report["enemies"][0]
        assert sighting["tank_id"] == 506
        assert sighting["observed_ms"] == _NOW - 1000

    def test_report_withholds_allies_corpses_and_stale_positions(self) -> None:
        """Same-team, dead, unplaced, and stale tanks are not knowledge."""
        ally = _enemy(600, team=2)
        corpse = make_tank_state(
            tank_id=601,
            x=51,
            y=61,
            team=0,
            rank=1,
            name="red-1",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=_NOW,
            last_wire_seen_ms=_NOW,
            last_position_update_ms=_NOW,
            liveness="deactivated",
        )
        unplaced = make_tank_state(
            tank_id=602,
            x=0,
            y=0,
            team=0,
            rank=1,
            name="red-2",
            is_self=False,
            is_bot=False,
            damage_state=3,
            timestamp_ms=_NOW,
        )
        stale = _enemy(603, observed_ms=_NOW - ENEMY_SIGHTING_TTL_MS - 1)
        ws = _world_service(tanks={"600": ally, "601": corpse, "602": unplaced, "603": stale})

        report = build_fleet_report(
            ws, instance="arterial", role="fighter", engaged_target_id=-1, now_ms=_NOW
        )

        if report is None:
            raise AssertionError("expected a report")
        assert report["enemies"] == []

    def test_report_withholds_locally_failed_containers(self) -> None:
        """A failed-pickup mark is this bot's verdict, not knowledge."""
        good = make_container_state(
            x=50, y=60, is_fuel=True, volume=700, timestamp_ms=_NOW - 500, failed_pickups=0
        )
        failed = make_container_state(
            x=51, y=61, is_fuel=False, volume=0, timestamp_ms=_NOW - 500, failed_pickups=1
        )
        ws = _world_service(containers={"50,60": good, "51,61": failed})

        report = build_fleet_report(
            ws, instance="arterial", role="gatherer", engaged_target_id=-1, now_ms=_NOW
        )

        if report is None:
            raise AssertionError("expected a report")
        assert len(report["containers"]) == 1
        assert report["containers"][0]["x"] == 50
        assert report["containers"][0]["observed_ms"] == _NOW - 500


class TestWriteFleetReport:
    """The report lands atomically in the reporter's run directory."""

    def test_write_places_the_encoded_report(self, fake_fs: FakeFileSystem) -> None:
        ws = _world_service()
        report = build_fleet_report(
            ws, instance="arterial", role="fighter", engaged_target_id=-1, now_ms=_NOW
        )
        if report is None:
            raise AssertionError("expected a report")

        write_fleet_report(report)

        path = bot_run_dir("arterial") / FLEET_REPORT_FILENAME
        written = fake_fs.get_written_files()[str(path)]
        assert decode_fleet_report(load_json_str(written)) == report


class TestReportFreshnessBounds:
    """What the report publishes and withholds by age."""

    def test_only_live_coverage_is_published(self) -> None:
        """Expired tiles answer no sibling's question and stay home."""
        from tankpit_bot.state.scan_coverage import FORAGE_COVERAGE_TTL_MS
        from tankpit_bot.state.types import WorldStateDict

        ws = _world_service()
        ws.world_state = WorldStateDict(
            **{
                **ws.world_state,
                "scanned_tiles": {
                    "10,10": _NOW - 1000,
                    "11,10": _NOW - FORAGE_COVERAGE_TTL_MS - 1,
                },
            }
        )

        report = build_fleet_report(
            ws, instance="arterial", role="fighter", engaged_target_id=-1, now_ms=_NOW
        )

        if report is None:
            raise AssertionError("expected a report")
        assert [(tile["x"], tile["y"]) for tile in report["scanned"]] == [(10, 10)]
        assert report["scanned"][0]["observed_ms"] == _NOW - 1000

    def test_aged_container_beliefs_stay_home(self) -> None:
        """The share bound is stricter than local larder memory: a
        receiver must TRAVEL to act on a sighting, so anything older
        than the co-farm consumption horizon is noise (2026-08-14
        240 s run: 18 code-4 disproofs on aged imports)."""
        from tankpit_bot.fleetshare.report import CONTAINER_SIGHTING_TTL_MS

        fresh = make_container_state(
            x=50, y=60, is_fuel=True, volume=700, timestamp_ms=_NOW - 1000, failed_pickups=0
        )
        aged = make_container_state(
            x=51,
            y=61,
            is_fuel=False,
            volume=0,
            timestamp_ms=_NOW - CONTAINER_SIGHTING_TTL_MS - 1,
            failed_pickups=0,
        )
        ws = _world_service(containers={"50,60": fresh, "51,61": aged})

        report = build_fleet_report(
            ws, instance="arterial", role="fighter", engaged_target_id=-1, now_ms=_NOW
        )

        if report is None:
            raise AssertionError("expected a report")
        assert [(c["x"], c["y"]) for c in report["containers"]] == [(50, 60)]

    def test_recent_removals_are_published_and_old_ones_stay_home(self) -> None:
        """The tombstone map is the removal ledger; only removals
        inside the container share horizon still refute anything."""
        from tankpit_bot.fleetshare.report import CONTAINER_SIGHTING_TTL_MS

        ws = _world_service()
        ws.container_disproofs["70,80"] = _NOW - 1000
        ws.container_disproofs["71,81"] = _NOW - CONTAINER_SIGHTING_TTL_MS - 1

        report = build_fleet_report(
            ws, instance="arterial", role="fighter", engaged_target_id=-1, now_ms=_NOW
        )

        if report is None:
            raise AssertionError("expected a report")
        assert [(r["x"], r["y"], r["removed_ms"]) for r in report["removed"]] == [
            (70, 80, _NOW - 1000)
        ]
