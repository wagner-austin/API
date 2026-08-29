"""Integration tests for the per-tick fleet knowledge exchange."""

from __future__ import annotations

from platform_core.json_utils import dump_json_str, load_json_str

from tankpit_bot.bot.ai.threat_primitives import human_combat_consented
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.tick_body import _exchange_fleet_knowledge
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.fleetshare.codecs import decode_fleet_report, encode_fleet_report
from tankpit_bot.fleetshare.report import FLEET_REPORT_FILENAME
from tankpit_bot.fleetshare.types import (
    FleetContainerSightingDict,
    FleetEnemySightingDict,
    FleetReportDict,
    FleetScannedTileDict,
)
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import WorldStateDict, make_self_state
from tests._runtime_logging_support import capture_runtime_events, event_fields
from tests.conftest import FakeEnv, FakeFileSystem


def _entered_bot(ws: WorldService) -> Bot:
    """Build a bot whose session has an established self (team 2)."""
    ws.set_selected_room("6")
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
    return Bot("https://test.tankpit.com/", headless=True, world=ws)


def test_exchange_before_entry_writes_nothing(fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
    """No established self: nothing attributable to offer, no file."""
    bot = Bot("https://test.tankpit.com/", headless=True, world=WorldService())

    _exchange_fleet_knowledge(bot)

    assert not any(FLEET_REPORT_FILENAME in path for path in fake_fs.get_written_files())


def test_exchange_publishes_and_merges_a_sibling(
    fake_env: FakeEnv, fake_fs: FakeFileSystem
) -> None:
    """One pass writes our report and absorbs a teammate's knowledge."""
    fake_env.set("TANKPIT_BOT_INSTANCE", "arterial")
    # The sibling's stamps carry the same wall clock the exchange
    # reads -- a just-written report, exactly like a live sibling.
    sibling_ms = get_current_time_ms()
    sibling = FleetReportDict(
        instance="artax",
        team=2,
        room="6",
        tank_id=1301,
        role="fighter",
        x=90,
        y=90,
        engaged_target_id=506,
        forage_goal_x=-1,
        forage_goal_y=-1,
        collect_claim_x=-1,
        collect_claim_y=-1,
        combat_consent_ids=[709],
        written_ms=sibling_ms,
        enemies=[
            FleetEnemySightingDict(
                tank_id=506,
                name="red-6",
                team=0,
                rank=1,
                x=170,
                y=40,
                damage_state=1,
                observed_ms=sibling_ms,
            )
        ],
        containers=[
            FleetContainerSightingDict(x=80, y=90, is_fuel=True, volume=650, observed_ms=sibling_ms)
        ],
        removed=[],
        scanned=[FleetScannedTileDict(x=91, y=90, observed_ms=sibling_ms)],
    )
    fake_fs.write_text(
        bot_run_dir("artax") / FLEET_REPORT_FILENAME,
        dump_json_str(encode_fleet_report(sibling)),
    )
    ws = WorldService()
    bot = _entered_bot(ws)

    with capture_runtime_events() as records:
        _exchange_fleet_knowledge(bot)

    own_path = bot_run_dir("arterial") / FLEET_REPORT_FILENAME
    own = decode_fleet_report(load_json_str(fake_fs.get_written_files()[str(own_path)]))
    assert own["tank_id"] == 2731
    assert own["team"] == 2
    assert own["role"] == "fighter"
    assert "506" in ws.world_state["tanks"]
    assert ws.world_state["containers"]["80,90"]["volume"] == 650
    assert ws.fleet_engaged_target_ids == {506: sibling_ms}
    # The sibling's consent evidence is inherited (operator ruling
    # 2026-08-26: the human engaged our COLOR, not one tank) — the
    # exact gap that kept arterial farming while artax dueled Beerus.
    assert ws.fleet_consented_tank_ids == {709}
    assert human_combat_consented(ws, 709) is True
    merged = [
        event_fields(record)
        for record in records
        if event_fields(record).get("diagnostic_kind") == "fleet_knowledge_merged"
    ]
    assert ws.world_state["scanned_tiles"]["91,90"] == sibling_ms
    assert merged == [
        {
            "diagnostic_kind": "fleet_knowledge_merged",
            "reports": 1,
            "enemies": 1,
            "containers": 1,
            "removed": 0,
            "scanned": 1,
        }
    ]


def test_exchange_with_no_siblings_stays_quiet(fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
    """A single tank exchanges with an empty fleet: write only."""
    fake_env.set("TANKPIT_BOT_INSTANCE", "arterial")
    ws = WorldService()
    bot = _entered_bot(ws)

    with capture_runtime_events() as records:
        _exchange_fleet_knowledge(bot)

    own_path = bot_run_dir("arterial") / FLEET_REPORT_FILENAME
    assert str(own_path) in fake_fs.get_written_files()
    kinds = [event_fields(record).get("diagnostic_kind") for record in records]
    assert "fleet_knowledge_merged" not in kinds
