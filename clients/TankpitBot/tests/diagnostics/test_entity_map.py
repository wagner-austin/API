"""End-to-end tests for the entity-collection discovery analyzer.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` ->
:meth:`tankpit_bot.diagnostics.entity_alignment.EntityAlignmentEmitter.maybe_emit`
-> real ``_HookEventArtifactHandler`` -> JSONL via
:class:`tests.conftest.FakeFileSystem` -> real
:func:`tankpit_bot.diagnostics.entity_map.build_entity_map_report` ->
assertions on the structured :class:`EntityMapReportDict`. Nothing is
mocked; the JSONL is byte-identical to what a live bot run writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.diagnostics.entity_alignment import EntityAlignmentEmitter
from tankpit_bot.diagnostics.entity_alignment_types import (
    EntityCollectionCandidateDict,
    EntityMapReportDict,
)
from tankpit_bot.diagnostics.entity_map import (
    build_entity_map_report,
    main,
    render_entity_map_report,
)
from tankpit_bot.runtime_logging import (
    configure_bot_runtime_logging,
    emit_diagnostic,
)
from tankpit_bot.state import make_empty_world_state
from tankpit_bot.state.types import WorldStateDict, coord_key, make_container_state


def _make_world(positions: list[tuple[int, int, bool]]) -> WorldStateDict:
    """Return a world state tracking one container per position triple."""
    world = make_empty_world_state()
    for x, y, is_fuel in positions:
        world["containers"][coord_key(x, y)] = make_container_state(
            x, y, is_fuel, 500 if is_fuel else 0, timestamp_ms=1000
        )
    return world


def _emit_sample(
    positions: list[tuple[int, int, bool]],
    collections: dict[str, list[dict[str, int | float | bool | str | None]]],
) -> None:
    """Emit one entity alignment sample through the real emitter pipeline."""
    snapshot = PageClientSnapshotDict(
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
    assert EntityAlignmentEmitter().maybe_emit(_make_world(positions), snapshot, in_combat=False)


def _emit_discovery_samples() -> str:
    """Emit two samples where collection ``ba`` carries containers as (u, v).

    Returns:
        Path string of the latest events artifact.
    """
    artifacts = configure_bot_runtime_logging("20260610-120000")
    _emit_sample(
        [(146, 44, True), (150, 48, False), (160, 60, True)],
        {
            "ba": [
                {"u": 146, "v": 44, "w": True, "n": "f1", "z": None},
                {"u": 150, "v": 48, "w": False},
                {"u": 99, "v": 99, "w": True},
                {"u": 5},
            ],
            "cc": [{"n": "Artax", "s": 346}],
            "ok": [{"p": 146, "q": 44}],
        },
    )
    _emit_sample(
        [(150, 48, False)],
        {
            "ba": [
                {"u": 150.0, "v": 48.0, "w": False},
                {"u": 99, "v": 99, "w": True},
            ],
        },
    )
    return artifacts["latest_events_path"]


def test_build_report_discovers_container_collection(fake_fs: FakeFileSystem) -> None:
    """The collection whose (x, y) pair tracks belief containers wins discovery."""
    latest = _emit_discovery_samples()

    report = build_entity_map_report(Path(latest))

    assert report == EntityMapReportDict(
        source_path=latest,
        mode="bot",
        sample_count=2,
        candidates=[
            EntityCollectionCandidateDict(
                collection_key="ba",
                x_key="u",
                y_key="v",
                matched_items=3,
                total_items=6,
                belief_matched=3,
                belief_total=4,
            ),
            EntityCollectionCandidateDict(
                collection_key="ok",
                x_key="p",
                y_key="q",
                matched_items=1,
                total_items=1,
                belief_matched=1,
                belief_total=4,
            ),
            EntityCollectionCandidateDict(
                collection_key="cc",
                x_key="",
                y_key="",
                matched_items=0,
                total_items=1,
                belief_matched=0,
                belief_total=4,
            ),
        ],
    )


def test_report_quantifies_bot_blind_containers(fake_fs: FakeFileSystem) -> None:
    """Client items with zero matching beliefs surface as divergence counts.

    Join-time blindness: the client renders containers discovered before
    the bot joined; the bot's belief list is empty, so every item in the
    client's container collection is unmatched.
    """
    artifacts = configure_bot_runtime_logging("20260610-120000")
    _emit_sample(
        [],
        {"ba": [{"u": 146, "v": 44}, {"u": 150, "v": 48}]},
    )
    _emit_sample(
        [(146, 44, True)],
        {"ba": [{"u": 146, "v": 44}, {"u": 150, "v": 48}]},
    )

    report = build_entity_map_report(Path(artifacts["latest_events_path"]))
    rendered = render_entity_map_report(report)

    assert report["candidates"][0] == EntityCollectionCandidateDict(
        collection_key="ba",
        x_key="u",
        y_key="v",
        matched_items=1,
        total_items=4,
        belief_matched=1,
        belief_total=1,
    )
    assert "DIVERGENCE -- 3 client item(s) the bot does not track" in rendered


def test_build_report_with_no_samples(fake_fs: FakeFileSystem) -> None:
    """An artifact without alignment samples yields an empty candidate list."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    emit_diagnostic(diagnostic_kind="session_room_joined", room_id="1", field_image="field01.gif")

    report = build_entity_map_report(Path(artifacts["latest_events_path"]))

    assert report["sample_count"] == 0
    assert report["candidates"] == []
    assert "(no entity_alignment_sample events" in render_entity_map_report(report)


def test_render_flags_unmatchable_collection_and_clean_collection(
    fake_fs: FakeFileSystem,
) -> None:
    """Collections without a coordinate pair and fully matched ones both render."""
    latest = _emit_discovery_samples()

    rendered = render_entity_map_report(build_entity_map_report(Path(latest)))

    assert "cc       -> no coordinate pair matches belief containers (items=1)" in rendered
    assert "ba       -> x=u y=v" in rendered
    assert "items matching a belief container: 3/6" in rendered
    assert "belief containers found in client list: 3/4" in rendered
    assert "ok       -> x=p y=q" in rendered
    assert "DIVERGENCE -- 3 client item(s)" in rendered


def test_build_report_raises_on_sample_missing_collections_field(
    fake_fs: FakeFileSystem,
) -> None:
    """A sample record missing ``world_collections_json`` fails loudly."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    emit_diagnostic(
        diagnostic_kind="entity_alignment_sample",
        belief_container_count=0,
        belief_containers_json='{"containers": []}',
    )

    with pytest.raises(KeyError, match="world_collections_json"):
        build_entity_map_report(Path(artifacts["latest_events_path"]))


def test_build_report_raises_on_belief_payload_without_containers_key(
    fake_fs: FakeFileSystem,
) -> None:
    """A belief payload missing its ``containers`` list fails strict decode."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    emit_diagnostic(
        diagnostic_kind="entity_alignment_sample",
        belief_container_count=0,
        belief_containers_json="{}",
        world_collections_json="{}",
    )

    with pytest.raises(JSONTypeError, match="belief_containers"):
        build_entity_map_report(Path(artifacts["latest_events_path"]))


def test_main_renders_report_for_explicit_artifact(fake_fs: FakeFileSystem) -> None:
    """``main()`` resolves the artifact path from argv and exits 0."""
    latest = _emit_discovery_samples()
    argv_value = ["tankpit-entity-map", latest]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: argv_value
    try:
        exit_code = main()
    finally:
        _test_hooks.get_argv = original_get_argv

    assert exit_code == 0
