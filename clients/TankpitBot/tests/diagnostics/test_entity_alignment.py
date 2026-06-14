"""End-to-end tests for the entity-alignment sample emitter.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` ->
:func:`tankpit_bot.diagnostics.entity_alignment.maybe_emit_entity_alignment_sample`
-> real ``_HookEventArtifactHandler`` -> JSONL via
:class:`tests.conftest.FakeFileSystem` -> real
:func:`tankpit_bot.diagnostics.event_stream.load_event_records` ->
assertions on the decoded record. Nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str
from tests.conftest import FakeFileSystem

from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    encode_client_collections,
)
from tankpit_bot.diagnostics.entity_alignment import (
    maybe_emit_entity_alignment_sample,
    reset_entity_alignment_emitter,
)
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    configure_bot_runtime_logging,
)
from tankpit_bot.state import make_empty_world_state
from tankpit_bot.state.types import (
    WorldStateDict,
    coord_key,
    encode_container_state,
    make_container_state,
)

_COLLECTIONS: dict[str, list[dict[str, int | float | bool | str | None]]] = {
    "ba": [
        {"u": 146, "v": 44, "w": True},
        {"u": 150, "v": 48, "w": False},
    ],
}


def _make_world(positions: list[tuple[int, int, bool]]) -> WorldStateDict:
    """Return a world state tracking one container per position triple."""
    world = make_empty_world_state()
    for x, y, is_fuel in positions:
        world["containers"][coord_key(x, y)] = make_container_state(
            x, y, is_fuel, 500 if is_fuel else 0, timestamp_ms=1000
        )
    return world


def _make_snapshot(
    collections: dict[str, list[dict[str, int | float | bool | str | None]]],
) -> PageClientSnapshotDict:
    """Return a healthy live-client snapshot carrying ``collections``."""
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
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections=collections,
    )


def _sample_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``entity_alignment_sample`` record from the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["channel"] == "DIAGNOSTIC"
        and record["fields"].get("diagnostic_kind") == "entity_alignment_sample"
    ]


def test_emit_writes_sample_through_real_pipeline(fake_fs: FakeFileSystem) -> None:
    """An emitted sample lands in the JSONL with the exact belief + truth payload."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    world = _make_world([(146, 44, True), (150, 48, False)])

    emitted = maybe_emit_entity_alignment_sample(
        world,
        _make_snapshot(_COLLECTIONS),
        in_combat=False,
    )

    assert emitted is True
    records = _sample_records(artifacts["latest_events_path"])
    assert len(records) == 1
    expected_belief: JSONObject = {
        "containers": [encode_container_state(c) for c in world["containers"].values()],
    }
    assert records[0]["fields"] == {
        "diagnostic_kind": "entity_alignment_sample",
        "belief_container_count": 2,
        "belief_containers_json": dump_json_str(expected_belief),
        "world_collections_json": dump_json_str(encode_client_collections(_COLLECTIONS)),
    }


def test_emit_skips_when_collections_empty(fake_fs: FakeFileSystem) -> None:
    """An empty truth side produces no sample -- there is nothing to align."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    world = _make_world([(146, 44, True)])
    assert maybe_emit_entity_alignment_sample(world, _make_snapshot(_COLLECTIONS), in_combat=False)

    emitted = maybe_emit_entity_alignment_sample(
        _make_world([(1, 2, True)]),
        _make_snapshot({}),
        in_combat=False,
    )

    assert emitted is False
    assert len(_sample_records(artifacts["latest_events_path"])) == 1


def test_emit_gates_on_unchanged_container_signature(fake_fs: FakeFileSystem) -> None:
    """A second tick with the same container set emits nothing new."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    world = _make_world([(146, 44, True), (150, 48, False)])
    assert maybe_emit_entity_alignment_sample(world, _make_snapshot(_COLLECTIONS), in_combat=False)

    emitted = maybe_emit_entity_alignment_sample(
        world,
        _make_snapshot(_COLLECTIONS),
        in_combat=False,
    )

    assert emitted is False
    assert len(_sample_records(artifacts["latest_events_path"])) == 1


def test_emit_again_when_container_set_changes(fake_fs: FakeFileSystem) -> None:
    """A pickup (container removal) re-opens the gate and emits a second sample."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    assert maybe_emit_entity_alignment_sample(
        _make_world([(146, 44, True), (150, 48, False)]),
        _make_snapshot(_COLLECTIONS),
        in_combat=False,
    )

    emitted = maybe_emit_entity_alignment_sample(
        _make_world([(150, 48, False)]),
        _make_snapshot(_COLLECTIONS),
        in_combat=False,
    )

    assert emitted is True
    records = _sample_records(artifacts["latest_events_path"])
    assert [r["fields"]["belief_container_count"] for r in records] == [2, 1]


def test_reset_clears_change_gate(fake_fs: FakeFileSystem) -> None:
    """``reset_entity_alignment_emitter`` lets an identical belief emit again."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    world = _make_world([(146, 44, True)])
    assert maybe_emit_entity_alignment_sample(world, _make_snapshot(_COLLECTIONS), in_combat=False)

    reset_entity_alignment_emitter()
    emitted = maybe_emit_entity_alignment_sample(
        world,
        _make_snapshot(_COLLECTIONS),
        in_combat=False,
    )

    assert emitted is True
    assert len(_sample_records(artifacts["latest_events_path"])) == 2


def test_emit_with_empty_belief_captures_blindness_case(fake_fs: FakeFileSystem) -> None:
    """A client list with zero bot containers still emits -- that IS the divergence.

    This is the join-time blindness scenario: the client renders
    containers discovered before the bot joined while the bot's world
    state tracks none.
    """
    artifacts = configure_bot_runtime_logging("20260610-120000")

    emitted = maybe_emit_entity_alignment_sample(
        _make_world([]),
        _make_snapshot(_COLLECTIONS),
        in_combat=False,
    )

    assert emitted is True
    records = _sample_records(artifacts["latest_events_path"])
    assert records[0]["fields"]["belief_container_count"] == 0
