"""End-to-end tests for the self-alignment sample emitter.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` ->
:meth:`tankpit_bot.diagnostics.self_alignment.SelfAlignmentEmitter.maybe_emit`
-> real ``_HookEventArtifactHandler`` -> JSONL via
:class:`tests.conftest.FakeFileSystem` -> real
:func:`tankpit_bot.diagnostics.event_stream.load_event_records` ->
assertions on the decoded record. Nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes import FakeCDPSession

from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.browser.page_client_snapshot_codecs import encode_client_field_map
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.self_alignment import SelfAlignmentEmitter
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tankpit_bot.runtime_records import RuntimeEventRecordDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import SelfStateDict, make_self_state

_SELF_FIELDS: dict[str, int | float | bool | str | None] = {
    "A": 99,
    "a": 131,
    "b": 110,
    "cy": 1100,
    "flag": True,
    "name": "tank",
    "n": None,
}


def _make_self_state(*, x: int = 131, y: int = 110, fuel: int = 1100) -> SelfStateDict:
    """Return a wire-derived belief state for emitter tests."""
    return make_self_state(
        tank_id=99,
        x=x,
        y=y,
        team=2,
        rank=1,
        fuel=fuel,
        leaderboard_position=1,
    )


def _make_snapshot(
    self_fields: dict[str, int | float | bool | str | None],
) -> PageClientSnapshotDict:
    """Return a healthy live-client snapshot carrying ``self_fields``."""
    return PageClientSnapshotDict(
        timestamp_ms=1000,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=50,
        last_page_client_send_age_ms=100,
        last_bot_send_age_ms=100,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields=self_fields,
        world_fields={},
        map_fields={},
        world_collections={},
    )


def _sample_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``self_alignment_sample`` record from the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["channel"] == "DIAGNOSTIC"
        and record["fields"].get("diagnostic_kind") == "self_alignment_sample"
    ]


def test_emit_writes_sample_through_real_pipeline(fake_fs: FakeFileSystem) -> None:
    """An emitted sample lands in the JSONL with the exact belief + truth payload."""
    artifacts = configure_bot_runtime_logging("20260609-120000")

    emitted = SelfAlignmentEmitter().maybe_emit(_make_self_state(), _make_snapshot(_SELF_FIELDS))

    assert emitted is True
    records = _sample_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"] == {
        "diagnostic_kind": "self_alignment_sample",
        "belief_tank_id": 99,
        "belief_x": 131,
        "belief_y": 110,
        "belief_fuel": 1100,
        "self_fields_json": dump_json_str(encode_client_field_map(_SELF_FIELDS)),
    }


def test_emit_skips_when_self_fields_empty(fake_fs: FakeFileSystem) -> None:
    """An empty truth map produces no sample -- there is nothing to align."""
    artifacts = configure_bot_runtime_logging("20260609-120000")
    emitter = SelfAlignmentEmitter()
    assert emitter.maybe_emit(_make_self_state(), _make_snapshot(_SELF_FIELDS))

    emitted = emitter.maybe_emit(
        _make_self_state(x=200),
        _make_snapshot({}),
    )

    assert emitted is False
    assert len(_sample_records(artifacts["latest_events_path"])) == 1


def test_emit_gates_on_unchanged_belief(fake_fs: FakeFileSystem) -> None:
    """A second tick with the same belief tuple emits nothing new."""
    artifacts = configure_bot_runtime_logging("20260609-120000")
    emitter = SelfAlignmentEmitter()
    assert emitter.maybe_emit(_make_self_state(), _make_snapshot(_SELF_FIELDS))

    emitted = emitter.maybe_emit(_make_self_state(), _make_snapshot(_SELF_FIELDS))

    assert emitted is False
    assert len(_sample_records(artifacts["latest_events_path"])) == 1


def test_emit_again_when_belief_changes(fake_fs: FakeFileSystem) -> None:
    """A belief change re-opens the gate and emits a second sample."""
    artifacts = configure_bot_runtime_logging("20260609-120000")
    emitter = SelfAlignmentEmitter()
    assert emitter.maybe_emit(_make_self_state(), _make_snapshot(_SELF_FIELDS))

    emitted = emitter.maybe_emit(
        _make_self_state(x=147, fuel=980),
        _make_snapshot(_SELF_FIELDS),
    )

    assert emitted is True
    records = _sample_records(artifacts["latest_events_path"])
    assert [r["fields"]["belief_x"] for r in records] == [131, 147]
    assert [r["fields"]["belief_fuel"] for r in records] == [1100, 980]


def test_each_emitter_carries_its_own_gate(fake_fs: FakeFileSystem) -> None:
    """A second emitter emits the same belief -- the gate is instance state.

    This is the property that replaced ``reset_self_alignment_emitter``:
    two sessions in one process must not silence each other's samples
    ([[session-state-deglobalisation]] step 3).
    """
    artifacts = configure_bot_runtime_logging("20260609-120000")
    assert SelfAlignmentEmitter().maybe_emit(_make_self_state(), _make_snapshot(_SELF_FIELDS))

    emitted = SelfAlignmentEmitter().maybe_emit(_make_self_state(), _make_snapshot(_SELF_FIELDS))

    assert emitted is True
    assert len(_sample_records(artifacts["latest_events_path"])) == 2


# Truth map served by the fake page client during the tick test. Tracks
# the wire belief the test seeds (tank_id=0, x=100, y=100, fuel=800) so
# the analyzer report at the end of the pipeline is fully deterministic.
_TICK_SELF_FIELDS: dict[str, int | float | bool | str | None] = {
    "A": 0,
    "a": 100,
    "b": 100,
    "cy": 800,
    "k": 5,
    "flag": True,
}


class _SelfFieldsCDPSession(FakeCDPSession):
    """Fake CDP session whose page-client snapshot carries self fields."""

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Serve a populated snapshot for the page-client capture expression.

        Args:
            method: CDP method name.
            params: CDP call parameters.

        Returns:
            CDP-style result object.
        """
        if method == "Runtime.evaluate" and params is not None:
            expression = str(params.get("expression", ""))
            if "window.__tankpitActiveGame" in expression and "map_visible" in expression:
                return {
                    "result": {
                        "value": {
                            "timestamp_ms": 1000,
                            "client_present": True,
                            "map_visible": False,
                            "client_state": 1,
                            "client_busy": False,
                            "pending_actions": 0,
                            "heartbeat_age_ms": 50,
                            "last_page_client_send_age_ms": 100,
                            "last_bot_send_age_ms": 100,
                            "ws_ready_state": 1,
                            "current_send_label": None,
                            "sent_frame_meta_queue_length": 0,
                            "self_fields": encode_client_field_map(_TICK_SELF_FIELDS),
                            "world_fields": {},
                            "map_fields": {},
                            "world_collections": {},
                        }
                    }
                }
        return super().send(method, params)


def test_tick_once_emits_sample_from_live_world_state(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """A full ``_tick_once`` pairs the wire belief with the page truth in JSONL,
    and the analyzer recovers the tracking keys from that same artifact.

    This is the complete pipeline in one test: fake CDP snapshot value ->
    real ``decode_page_client_snapshot`` -> tick-boundary emission -> real
    JSONL handler -> real ``build_self_map_report``.
    """
    from tankpit_bot.bot.ai.types import AIStateDict
    from tankpit_bot.bot.base import Bot
    from tankpit_bot.bot.tick_body import _tick_once
    from tankpit_bot.diagnostics.self_alignment_types import (
        SelfFieldCandidateDict,
        SelfMapReportDict,
    )
    from tankpit_bot.diagnostics.self_map import build_self_map_report
    from tankpit_bot.sniffer.world_state_containers import (
        update_world_state_from_fuel_total,
    )
    from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol

    ws = WorldService()
    artifacts = configure_bot_runtime_logging("20260609-120000")
    ws.update_world_state_from_position(100, 100)
    update_world_state_from_fuel_total(ws, 800)
    update_inventory_from_protocol(ws, [30, 30, 30, 30, 30], [True] * 5)
    bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
    bot._cdp = _SelfFieldsCDPSession()
    bot._state_data = bot._state_data.copy()
    bot._state_data["state"] = "IDLE"
    bot._ai_state = AIStateDict(**{**bot._ai_state, "last_scan_ms": 1})

    _tick_once(bot)

    records = _sample_records(artifacts["latest_events_path"])
    assert len(records) == 1
    # update_self_position creates the minimal belief (tank_id=0) and the
    # fuel-total update sets fuel=800 -- the sample must mirror exactly that.
    assert records[0]["fields"] == {
        "diagnostic_kind": "self_alignment_sample",
        "belief_tank_id": 0,
        "belief_x": 100,
        "belief_y": 100,
        "belief_fuel": 800,
        "self_fields_json": dump_json_str(encode_client_field_map(_TICK_SELF_FIELDS)),
    }
    # x and y share the value 100 in this single sample, so both "a" and
    # "b" survive for both dimensions -- the ambiguity the analyzer must
    # surface until more varied samples arrive.
    report = build_self_map_report(Path(artifacts["latest_events_path"]))
    assert report == SelfMapReportDict(
        source_path=artifacts["latest_events_path"],
        mode="bot",
        sample_count=1,
        candidates=[
            SelfFieldCandidateDict(
                dimension="tank_id",
                matching_keys=["A"],
                distinct_belief_values=1,
                sample_count=1,
            ),
            SelfFieldCandidateDict(
                dimension="x",
                matching_keys=["a", "b"],
                distinct_belief_values=1,
                sample_count=1,
            ),
            SelfFieldCandidateDict(
                dimension="y",
                matching_keys=["a", "b"],
                distinct_belief_values=1,
                sample_count=1,
            ),
            SelfFieldCandidateDict(
                dimension="fuel",
                matching_keys=["cy"],
                distinct_belief_values=1,
                sample_count=1,
            ),
        ],
    )
