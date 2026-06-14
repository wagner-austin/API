"""End-to-end tests for the self-field mapping-discovery analyzer.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` ->
:func:`tankpit_bot.diagnostics.self_alignment.maybe_emit_self_alignment_sample`
-> real ``_HookEventArtifactHandler`` -> JSONL via
:class:`tests.conftest.FakeFileSystem` -> real
:func:`tankpit_bot.diagnostics.self_map.build_self_map_report` ->
assertions on the structured :class:`SelfMapReportDict`. Nothing is
mocked; the JSONL is byte-identical to what a live bot run writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.diagnostics.self_alignment import maybe_emit_self_alignment_sample
from tankpit_bot.diagnostics.self_alignment_types import (
    SelfAlignmentSampleDict,
    SelfFieldCandidateDict,
    SelfMapReportDict,
)
from tankpit_bot.diagnostics.self_map import (
    _belief_value,
    build_self_map_report,
    main,
    render_self_map_report,
)
from tankpit_bot.runtime_logging import configure_bot_runtime_logging, emit_diagnostic
from tankpit_bot.state.types import SelfStateDict


def _emit_sample(
    *,
    tank_id: int,
    x: int,
    y: int,
    fuel: int,
    self_fields: dict[str, int | float | bool | str | None],
) -> None:
    """Emit one alignment sample through the real emitter pipeline."""
    self_state = SelfStateDict(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        fuel=fuel,
        leaderboard_position=1,
    )
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
        self_fields=self_fields,
        world_fields={},
        map_fields={},
        world_collections={},
    )
    assert maybe_emit_self_alignment_sample(self_state, snapshot)


def _emit_two_movement_samples() -> str:
    """Emit two samples where ``a``/``fx`` track x and ``cy`` tracks fuel.

    Returns:
        Path string of the latest events artifact.
    """
    artifacts = configure_bot_runtime_logging("20260609-120000")
    _emit_sample(
        tank_id=99,
        x=131,
        y=110,
        fuel=1100,
        self_fields={
            "A": 99,
            "a": 131,
            "fx": 131.0,
            "b": 110,
            "cy": 1100,
            "k": 5,
            "flag": True,
            "name": "tank",
            "n": None,
        },
    )
    _emit_sample(
        tank_id=99,
        x=147,
        y=110,
        fuel=980,
        self_fields={
            "A": 99,
            "a": 147,
            "fx": 147.0,
            "b": 110,
            "cy": 980,
            "k": 5,
            "flag": True,
            "name": "tank",
            "n": None,
        },
    )
    return artifacts["latest_events_path"]


def test_build_report_discovers_tracking_keys(fake_fs: FakeFileSystem) -> None:
    """Keys that numerically track each belief dimension survive intersection."""
    latest = _emit_two_movement_samples()

    report = build_self_map_report(Path(latest))

    assert report == SelfMapReportDict(
        source_path=latest,
        mode="bot",
        sample_count=2,
        candidates=[
            SelfFieldCandidateDict(
                dimension="tank_id",
                matching_keys=["A"],
                distinct_belief_values=1,
                sample_count=2,
            ),
            SelfFieldCandidateDict(
                dimension="x",
                matching_keys=["a", "fx"],
                distinct_belief_values=2,
                sample_count=2,
            ),
            SelfFieldCandidateDict(
                dimension="y",
                matching_keys=["b"],
                distinct_belief_values=1,
                sample_count=2,
            ),
            SelfFieldCandidateDict(
                dimension="fuel",
                matching_keys=["cy"],
                distinct_belief_values=2,
                sample_count=2,
            ),
        ],
    )


def test_build_report_with_no_samples(fake_fs: FakeFileSystem) -> None:
    """An artifact without alignment samples yields empty candidates.

    The artifact also carries a non-DIAGNOSTIC record with an empty
    mode string -- both must be skipped without disturbing the mode
    already learned from real records.
    """
    artifacts = configure_bot_runtime_logging("20260609-120000")
    emit_diagnostic(diagnostic_kind="session_room_joined", room_id="1", field_image="field01.gif")
    latest = Path(artifacts["latest_events_path"])
    fake_fs.append_text(
        latest,
        '{"timestamp": "2026-06-09T12:00:01", "level": "INFO", "logger": "x", '
        '"mode": "", "channel": "WIRE", "message": "map_open"}\n',
    )

    report = build_self_map_report(latest)

    assert report["sample_count"] == 0
    assert report["mode"] == "bot"  # empty-mode record must not clobber the learned mode
    assert report["candidates"] == [
        SelfFieldCandidateDict(
            dimension=dimension,
            matching_keys=[],
            distinct_belief_values=0,
            sample_count=0,
        )
        for dimension in ("tank_id", "x", "y", "fuel")
    ]
    assert "(no self_alignment_sample events" in render_self_map_report(report)


def test_bool_values_never_match_int_beliefs(fake_fs: FakeFileSystem) -> None:
    """``True`` never survives as a candidate for belief value ``1``."""
    artifacts = configure_bot_runtime_logging("20260609-120000")
    _emit_sample(
        tank_id=1,
        x=10,
        y=20,
        fuel=30,
        self_fields={"flag": True, "id": 1, "px": 10, "py": 20, "f": 30},
    )

    report = build_self_map_report(Path(artifacts["latest_events_path"]))

    assert report["candidates"][0] == SelfFieldCandidateDict(
        dimension="tank_id",
        matching_keys=["id"],
        distinct_belief_values=1,
        sample_count=1,
    )


def test_render_flags_no_tracking_key(fake_fs: FakeFileSystem) -> None:
    """A dimension no key tracks renders the missing-truth warning."""
    artifacts = configure_bot_runtime_logging("20260609-120000")
    _emit_sample(
        tank_id=99,
        x=131,
        y=110,
        fuel=555,
        self_fields={"A": 99, "a": 131, "b": 110},
    )

    report = build_self_map_report(Path(artifacts["latest_events_path"]))
    rendered = render_self_map_report(report)

    assert report["candidates"][3]["matching_keys"] == []
    assert "NO KEY tracks this dimension" in rendered


def test_render_flags_low_confidence_and_ambiguity(fake_fs: FakeFileSystem) -> None:
    """Single-value matches and multi-key survivors carry explicit warnings."""
    latest = _emit_two_movement_samples()

    rendered = render_self_map_report(build_self_map_report(Path(latest)))

    assert "tank_id  -> A" in rendered
    assert "LOW CONFIDENCE" in rendered
    assert "x        -> a, fx" in rendered
    assert "AMBIGUOUS" in rendered
    assert "fuel     -> cy" in rendered
    assert "Samples: 2" in rendered


def test_build_report_raises_on_sample_missing_belief_field(
    fake_fs: FakeFileSystem,
) -> None:
    """A sample record missing a belief field fails loudly at classify time."""
    artifacts = configure_bot_runtime_logging("20260609-120000")
    emit_diagnostic(
        diagnostic_kind="self_alignment_sample",
        belief_tank_id=1,
        belief_y=2,
        belief_fuel=3,
        self_fields_json="{}",
    )

    with pytest.raises(JSONTypeError, match="belief_x"):
        build_self_map_report(Path(artifacts["latest_events_path"]))


def test_belief_value_rejects_unknown_dimension_field() -> None:
    """``_belief_value`` raises ``ValueError`` for an unknown field name."""
    sample = SelfAlignmentSampleDict(
        timestamp="2026-06-09T12:00:00",
        belief_tank_id=1,
        belief_x=2,
        belief_y=3,
        belief_fuel=4,
        self_fields={},
    )

    with pytest.raises(ValueError, match="unknown belief dimension"):
        _belief_value(sample, "belief_rank")


def test_main_renders_report_for_explicit_artifact(fake_fs: FakeFileSystem) -> None:
    """``main()`` resolves the artifact path from argv and exits 0."""
    latest = _emit_two_movement_samples()
    argv_value = ["tankpit-self-map", latest]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: argv_value
    try:
        exit_code = main()
    finally:
        _test_hooks.get_argv = original_get_argv

    assert exit_code == 0


def test_main_raises_when_default_artifact_missing(fake_fs: FakeFileSystem) -> None:
    """``main()`` with no args fails fast when the default artifact is absent."""
    empty: list[str] = []
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: empty
    try:
        with pytest.raises(FileNotFoundError, match="events artifact not found"):
            main()
    finally:
        _test_hooks.get_argv = original_get_argv
