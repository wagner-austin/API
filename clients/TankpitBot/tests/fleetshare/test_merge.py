"""Tests for reading and merging teammates' fleet reports."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot import _test_hooks
from tankpit_bot.fleetshare.codecs import encode_fleet_report
from tankpit_bot.fleetshare.merge import (
    FLEET_REPORT_TTL_MS,
    merge_fleet_reports,
    read_team_reports,
)
from tankpit_bot.fleetshare.report import FLEET_REPORT_FILENAME
from tankpit_bot.fleetshare.types import (
    FleetContainerRemovalDict,
    FleetContainerSightingDict,
    FleetEnemySightingDict,
    FleetMineSightingDict,
    FleetReportDict,
    FleetScannedTileDict,
)
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    WorldStateDict,
    make_container_state,
    make_self_state,
    make_tank_state,
)
from tests.conftest import FakeFileSystem

_NOW = 100000


def _report(
    instance: str,
    *,
    team: int = 2,
    room: str = "6",
    tank_id: int = 1301,
    written_ms: int = _NOW - 1000,
    engaged_target_id: int = -1,
    war_ready: bool = False,
    forage_goal_x: int = -1,
    forage_goal_y: int = -1,
    collect_claim_x: int = -1,
    collect_claim_y: int = -1,
    enemies: list[FleetEnemySightingDict] | None = None,
    containers: list[FleetContainerSightingDict] | None = None,
    removed: list[FleetContainerRemovalDict] | None = None,
    mines: list[FleetMineSightingDict] | None = None,
    scanned: list[FleetScannedTileDict] | None = None,
) -> FleetReportDict:
    """Build a sibling report."""
    return FleetReportDict(
        instance=instance,
        team=team,
        room=room,
        tank_id=tank_id,
        role="fighter",
        x=90,
        y=90,
        engaged_target_id=engaged_target_id,
        war_ready=war_ready,
        forage_goal_x=forage_goal_x,
        forage_goal_y=forage_goal_y,
        collect_claim_x=collect_claim_x,
        collect_claim_y=collect_claim_y,
        combat_consent_ids=[],
        written_ms=written_ms,
        enemies=list(enemies) if enemies else [],
        containers=list(containers) if containers else [],
        removed=list(removed) if removed else [],
        mines=list(mines) if mines else [],
        scanned=list(scanned) if scanned else [],
    )


def _sighting(
    tank_id: int, *, team: int = 0, observed_ms: int = _NOW - 500
) -> FleetEnemySightingDict:
    """Build an enemy sighting row."""
    return FleetEnemySightingDict(
        tank_id=tank_id,
        name=f"red-{tank_id % 10}",
        team=team,
        rank=1,
        x=170,
        y=40,
        damage_state=1,
        observed_ms=observed_ms,
    )


def _write(fake_fs: FakeFileSystem, report: FleetReportDict) -> None:
    """Place an encoded sibling report in the fake run tree."""
    path = bot_run_dir(report["instance"]) / FLEET_REPORT_FILENAME
    fake_fs.write_text(path, dump_json_str(encode_fleet_report(report)))


class TestReadTeamReports:
    """Which sibling files count as fleet knowledge."""

    def test_reads_fresh_same_team_siblings_only(self, fake_fs: FakeFileSystem) -> None:
        """Own file, stale files, and other teams are all excluded."""
        _write(fake_fs, _report("artax"))
        _write(fake_fs, _report("arterial", tank_id=2731))
        _write(fake_fs, _report("old", written_ms=_NOW - FLEET_REPORT_TTL_MS - 1))
        _write(fake_fs, _report("enemyteam", team=0))
        # Same team, different ROOM: a Desert sibling's coordinates are
        # poison in a Practice belief set (2026-08-26) -- excluded.
        _write(fake_fs, _report("desertscout", room="7"))

        reports = read_team_reports("arterial", 2, "6", _NOW)

        assert [report["instance"] for report in reports] == ["artax"]

    def test_reads_the_sole_bot_namespace_file(self, fake_fs: FakeFileSystem) -> None:
        """A sole-namespace bot ('') is a sibling to a named instance."""
        _write(fake_fs, _report(""))

        reports = read_team_reports("arterial", 2, "6", _NOW)

        assert len(reports) == 1
        assert reports[0]["instance"] == ""

    def test_sole_bot_skips_its_own_namespace_file(self, fake_fs: FakeFileSystem) -> None:
        """The sole-namespace bot never reads its own report back."""
        _write(fake_fs, _report(""))

        assert read_team_reports("", 2, "6", _NOW) == []

    def test_read_retries_through_the_windows_replace_window(self, fake_fs: FakeFileSystem) -> None:
        """One PermissionError mid-swap: the immediate retry lands.

        The arterial tick-264 crash (2026-08-26 03:01:06): Windows
        refuses a concurrent open while the writer's ``os.replace``
        swaps the report; the swap completes within the failed open,
        so the very next attempt reads the fresh report.
        """
        _write(fake_fs, _report("artax"))
        real_read = _test_hooks.read_text
        calls = {"n": 0}

        def swap_window_read(path: Path) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError(13, "Permission denied")
            return real_read(path)

        _test_hooks.read_text = swap_window_read
        try:
            reports = read_team_reports("arterial", 2, "6", _NOW)
        finally:
            _test_hooks.read_text = real_read

        assert len(reports) == 1
        assert calls["n"] == 2

    def test_read_skips_a_sibling_denied_through_the_budget(self, fake_fs: FakeFileSystem) -> None:
        """Every attempt denied: the sibling is skipped, never a crash.

        Falsified premise (arterial tick 316, 2026-08-28 21:10, first
        two-bot World fleet under heavy machine load): three immediate
        retries all landed inside one Windows replace swap and the
        raise killed the session over one beat of advisory data that
        rewrites every ~2 s. The next tick reads the fresh rewrite.
        """
        _write(fake_fs, _report("artax"))
        real_read = _test_hooks.read_text

        def denied_read(path: Path) -> str:
            raise PermissionError(13, "Permission denied")

        _test_hooks.read_text = denied_read
        try:
            assert read_team_reports("arterial", 2, "6", _NOW) == []
        finally:
            _test_hooks.read_text = real_read


class TestMergeFleetReports:
    """How remote beliefs land in local state."""

    def _world_service(self) -> WorldService:
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
            }
        )
        return ws

    def test_fresh_enemy_sighting_lands_in_the_registry(self) -> None:
        ws = self._world_service()
        report = _report("artax", enemies=[_sighting(506)], engaged_target_id=506)

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary == {
            "reports": 1,
            "enemies": 1,
            "containers": 0,
            "removed": 0,
            "mines": 0,
            "scanned": 0,
        }
        merged = ws.world_state["tanks"]["506"]
        assert (merged["x"], merged["y"]) == (170, 40)
        assert merged["timestamp_ms"] == _NOW - 500
        # A merged sighting is map-like knowledge: it must never
        # advance the viewport gate that authorizes firing.
        assert merged["last_viewport_observation_ms"] == 0
        assert ws.fleet_engaged_target_ids == {506: _NOW - 1000}

    def test_local_fresher_belief_wins(self) -> None:
        ws = self._world_service()
        local = make_tank_state(
            tank_id=506,
            x=150,
            y=150,
            team=0,
            rank=1,
            name="red-6",
            is_self=False,
            is_bot=False,
            damage_state=3,
            timestamp_ms=_NOW - 100,
        )
        ws.world_state = WorldStateDict(**{**ws.world_state, "tanks": {"506": local}})
        report = _report("artax", enemies=[_sighting(506, observed_ms=_NOW - 500)])

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["enemies"] == 0
        assert ws.world_state["tanks"]["506"]["x"] == 150

    def test_own_and_same_team_sightings_never_merge(self) -> None:
        ws = self._world_service()
        report = _report(
            "artax",
            enemies=[_sighting(2731), _sighting(777, team=2)],
        )

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["enemies"] == 0
        assert ws.world_state["tanks"] == {}

    def test_container_sightings_merge_by_freshness(self) -> None:
        ws = self._world_service()
        fresh_local = make_container_state(
            x=50, y=60, is_fuel=True, volume=700, timestamp_ms=_NOW - 100, failed_pickups=0
        )
        ws.world_state = WorldStateDict(**{**ws.world_state, "containers": {"50,60": fresh_local}})
        report = _report(
            "artax",
            containers=[
                FleetContainerSightingDict(
                    x=50, y=60, is_fuel=True, volume=650, observed_ms=_NOW - 500
                ),
                FleetContainerSightingDict(
                    x=80, y=90, is_fuel=False, volume=0, observed_ms=_NOW - 200
                ),
            ],
        )

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["containers"] == 1
        assert ws.world_state["containers"]["50,60"]["volume"] == 700
        remote = ws.world_state["containers"]["80,90"]
        assert remote["is_fuel"] is False
        assert remote["refresh_kind"] == "fleet_report"

    def test_engaged_ids_replace_wholesale_and_skip_own(self) -> None:
        ws = self._world_service()
        ws.fleet_engaged_target_ids = {999: 1}
        newer = _report("artax", engaged_target_id=506, written_ms=_NOW - 500)
        older = _report("third", tank_id=1400, engaged_target_id=506, written_ms=_NOW - 900)
        own_target = _report("fourth", tank_id=1500, engaged_target_id=2731)

        merge_fleet_reports(ws, [newer, older, own_target], own_tank_id=2731, own_team=2)

        assert ws.fleet_engaged_target_ids == {506: _NOW - 500}

    def test_local_failed_mark_survives_a_fresher_remote_sighting(self) -> None:
        """A teammate's fresher volume refreshes the belief, but the
        local failed-pickup verdict is this bot's own and stays."""
        ws = self._world_service()
        marked = make_container_state(
            x=50, y=60, is_fuel=True, volume=700, timestamp_ms=_NOW - 900, failed_pickups=2
        )
        ws.world_state = WorldStateDict(**{**ws.world_state, "containers": {"50,60": marked}})
        report = _report(
            "artax",
            containers=[
                FleetContainerSightingDict(
                    x=50, y=60, is_fuel=True, volume=650, observed_ms=_NOW - 100
                )
            ],
        )

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["containers"] == 1
        refreshed = ws.world_state["containers"]["50,60"]
        assert refreshed["volume"] == 650
        assert refreshed["failed_pickups"] == 2

    def test_tombstoned_tile_rejects_older_sightings(self) -> None:
        """The Empty-container loop killer (run arterial 2026-08-14
        19:20): a locally disproven tile refuses a teammate's OLDER
        sighting -- without the tombstone, (102,85) was disproved
        three times in five seconds, re-imported between each."""
        ws = self._world_service()
        ws.container_disproofs["102,85"] = _NOW - 300
        report = _report(
            "artax",
            containers=[
                FleetContainerSightingDict(
                    x=102, y=85, is_fuel=True, volume=400, observed_ms=_NOW - 500
                )
            ],
        )

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["containers"] == 0
        assert "102,85" not in ws.world_state["containers"]

    def test_fresher_sighting_readmits_a_tombstoned_tile(self) -> None:
        """A respawned container passes naturally: its observation
        postdates the disproof."""
        ws = self._world_service()
        ws.container_disproofs["102,85"] = _NOW - 500
        report = _report(
            "artax",
            containers=[
                FleetContainerSightingDict(
                    x=102, y=85, is_fuel=True, volume=400, observed_ms=_NOW - 100
                )
            ],
        )

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["containers"] == 1
        assert ws.world_state["containers"]["102,85"]["volume"] == 400


class TestClockDomainGuard:
    """Future-stamped reports are clock-domain garbage."""

    def test_future_stamped_report_is_rejected(self, fake_fs: FakeFileSystem) -> None:
        """Every fleet bot shares one machine clock, so a written_ms
        AFTER now belongs to a different clock regime (a live
        artifact read under the sim seam's stepped clock, or vice
        versa) and must never pass as fresh."""
        _write(fake_fs, _report("artax", written_ms=_NOW + 5000))

        assert read_team_reports("arterial", 2, "6", _NOW) == []


class TestSharedScanCoverage:
    """A sibling's live coverage marks ground as known here too."""

    def test_coverage_advances_and_never_regresses(self) -> None:
        """Newer shared stamps land; fresher local coverage survives."""
        ws = TestMergeFleetReports()._world_service()
        ws.world_state = WorldStateDict(
            **{**ws.world_state, "scanned_tiles": {"10,10": _NOW - 100, "11,10": _NOW - 900}}
        )
        report = _report(
            "artax",
            scanned=[
                FleetScannedTileDict(x=10, y=10, observed_ms=_NOW - 500),
                FleetScannedTileDict(x=11, y=10, observed_ms=_NOW - 200),
                FleetScannedTileDict(x=12, y=10, observed_ms=_NOW - 300),
            ],
        )

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["scanned"] == 2
        tiles = ws.world_state["scanned_tiles"]
        assert tiles["10,10"] == _NOW - 100
        assert tiles["11,10"] == _NOW - 200
        assert tiles["12,10"] == _NOW - 300


class TestStaleSchemaArtifacts:
    """A dead build's stale file is dropped before full validation."""

    def test_stale_old_schema_file_is_skipped(self, fake_fs: FakeFileSystem) -> None:
        """Only the freshness key is validated on a stale file --
        its content is never consumed, so a pre-scanned-era schema
        (or any dead build's leftover) cannot crash the reader."""
        fake_fs.write_text(
            bot_run_dir("oldbuild") / FLEET_REPORT_FILENAME,
            dump_json_str({"written_ms": _NOW - FLEET_REPORT_TTL_MS - 1}),
        )

        assert read_team_reports("arterial", 2, "6", _NOW) == []

    def test_fresh_invalid_report_still_raises(self, fake_fs: FakeFileSystem) -> None:
        """A live mixed-schema fleet is a genuine bug and says so."""
        from platform_core.json_utils import JSONTypeError

        fake_fs.write_text(
            bot_run_dir("mixed") / FLEET_REPORT_FILENAME,
            dump_json_str({"written_ms": _NOW - 100}),
        )

        with pytest.raises(JSONTypeError, match="instance"):
            read_team_reports("arterial", 2, "6", _NOW)

    def test_non_object_file_raises(self, fake_fs: FakeFileSystem) -> None:
        """A non-object payload names the shape in the error."""
        from platform_core.json_utils import JSONTypeError

        fake_fs.write_text(bot_run_dir("weird") / FLEET_REPORT_FILENAME, "[1, 2]")

        with pytest.raises(JSONTypeError, match="fleet report must be an object"):
            read_team_reports("arterial", 2, "6", _NOW)


class TestSharedRemovals:
    """One bot's consumption updates the whole fleet."""

    def test_teammate_removal_drops_the_older_local_belief(self) -> None:
        """Fleet ruling 2026-08-14: when one bot takes discovered
        equipment, everyone learns -- the removal drops beliefs
        observed before it and the tombstone is inherited."""
        ws = TestMergeFleetReports()._world_service()
        stale_local = make_container_state(
            x=70, y=80, is_fuel=False, volume=0, timestamp_ms=_NOW - 800, failed_pickups=0
        )
        ws.world_state = WorldStateDict(**{**ws.world_state, "containers": {"70,80": stale_local}})
        report = _report(
            "artax",
            removed=[FleetContainerRemovalDict(x=70, y=80, removed_ms=_NOW - 300)],
        )

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["removed"] == 1
        assert "70,80" not in ws.world_state["containers"]
        # The tombstone is inherited: a third party's stale sighting
        # (older than the removal) now stays out here too.
        assert ws.container_disproofs["70,80"] == _NOW - 300

    def test_fresher_local_belief_survives_an_old_removal(self) -> None:
        """A tile observed AFTER the teammate's removal may have
        respawned -- the local belief stands."""
        ws = TestMergeFleetReports()._world_service()
        fresh_local = make_container_state(
            x=70, y=80, is_fuel=True, volume=500, timestamp_ms=_NOW - 100, failed_pickups=0
        )
        ws.world_state = WorldStateDict(**{**ws.world_state, "containers": {"70,80": fresh_local}})
        report = _report(
            "artax",
            removed=[FleetContainerRemovalDict(x=70, y=80, removed_ms=_NOW - 400)],
        )

        summary = merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert summary["removed"] == 0
        assert ws.world_state["containers"]["70,80"]["volume"] == 500
        assert ws.container_disproofs["70,80"] == _NOW - 400

    def test_own_fresher_tombstone_is_not_regressed(self) -> None:
        """An older teammate removal never rolls back a fresher local
        disproof stamp."""
        ws = TestMergeFleetReports()._world_service()
        ws.container_disproofs["70,80"] = _NOW - 100
        report = _report(
            "artax",
            removed=[FleetContainerRemovalDict(x=70, y=80, removed_ms=_NOW - 400)],
        )

        merge_fleet_reports(ws, [report], own_tank_id=2731, own_team=2)

        assert ws.container_disproofs["70,80"] == _NOW - 100

    def test_sibling_goal_and_claim_land_in_world_service(self, fake_fs: FakeFileSystem) -> None:
        """Shared collect intents replace wholesale each merge pass."""
        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        report = _report(
            "artax",
            forage_goal_x=120,
            forage_goal_y=104,
            collect_claim_x=44,
            collect_claim_y=24,
        )

        merge_fleet_reports(ws, [report], own_tank_id=1400, own_team=2)

        assert ws.fleet_forage_goals == {"artax": (120, 104)}
        assert ws.fleet_claimed_containers == {"44,24"}

        merge_fleet_reports(ws, [], own_tank_id=1400, own_team=2)

        assert ws.fleet_forage_goals == {}
        assert ws.fleet_claimed_containers == set()
