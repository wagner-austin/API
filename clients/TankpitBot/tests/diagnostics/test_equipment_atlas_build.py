"""Tests for the equipment atlas builder CLI."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str, load_json_str

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.equipment_atlas_build import (
    ATLAS_MAX_TILES_PER_FIELD,
    SNAPSHOT_SAMPLE_STRIDE,
    build_equipment_atlas,
    main,
)


def _run_lines(field: str, tiles: list[tuple[int, int]], *, source: str = "viewport") -> str:
    """Build one synthetic events artifact sighting the given tiles.

    Args:
        field: Field image for the room-join line.
        tiles: Equipment tiles the belief snapshot carries.
        source: Container belief source (``world_state`` = fleet import).

    Returns:
        JSONL text.
    """
    containers = [{"x": x, "y": y, "is_fuel": False, "source": source} for x, y in tiles]
    lines = [
        dump_json_str(
            {
                "timestamp": "2026-08-05T00:00:00",
                "channel": "DIAGNOSTIC",
                "message": "diagnostic_kind=session_room_joined",
                "diagnostic_kind": "session_room_joined",
                "room_id": "7",
                "field_image": field,
            }
        )
    ]
    # One sampled snapshot (the stride keeps only every Nth; emit
    # exactly one so it is the sampled one).
    lines.append(
        dump_json_str(
            {
                "timestamp": "2026-08-05T00:00:02",
                "channel": "DIAGNOSTIC",
                "message": "diagnostic_kind=entity_alignment_sample",
                "diagnostic_kind": "entity_alignment_sample",
                "belief_containers_json": dump_json_str({"containers": containers}),
            }
        )
    )
    return "\n".join(lines) + "\n"


def _write_run(root: Path, name: str, text: str) -> None:
    """Write one events artifact under the corpus root.

    Args:
        root: Corpus root.
        name: Artifact filename.
        text: JSONL text.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text, encoding="utf-8")


def test_two_run_tiles_survive_and_one_offs_drop(tmp_path: Path) -> None:
    """The atlas keeps persistence, drops scatter and fleet imports."""
    _write_run(tmp_path, "bot-1.events.jsonl", _run_lines("field01.gif", [(5, 5), (9, 9)]))
    _write_run(tmp_path, "bot-2.events.jsonl", _run_lines("field01.gif", [(5, 5)]))
    # A fleet-imported sighting counts for nobody.
    _write_run(
        tmp_path,
        "bot-3.events.jsonl",
        _run_lines("field01.gif", [(9, 9)], source="world_state"),
    )
    # latest mirrors are duplicates of their stamped twins.
    _write_run(tmp_path, "latest.events.jsonl", _run_lines("field01.gif", [(7, 7), (5, 5)]))

    atlas = build_equipment_atlas(tmp_path)

    assert atlas == {"field01.gif": [[5, 5, 2]]}


def test_snapshot_stride_samples_the_first_of_each_block(tmp_path: Path) -> None:
    """Dense snapshots are sampled; the stride keeps 1 in N."""
    lines = _run_lines("field01.gif", [(5, 5)]).splitlines()
    # Duplicate the snapshot line so only samples 1 and N+1 are read;
    # the unsampled copies carry a DIFFERENT tile that must not land.
    other = (
        lines[1].replace('"x":5,"y":5', '"x":8,"y":8').replace('"x": 5, "y": 5', '"x": 8, "y": 8')
    )
    padded = [lines[0], lines[1], *([other] * (SNAPSHOT_SAMPLE_STRIDE - 1))]
    _write_run(tmp_path, "bot-1.events.jsonl", "\n".join(padded) + "\n")
    _write_run(tmp_path, "bot-2.events.jsonl", _run_lines("field01.gif", [(5, 5)]))

    atlas = build_equipment_atlas(tmp_path)

    assert atlas == {"field01.gif": [[5, 5, 2]]}


def test_field_cap_keeps_the_strongest_tiles(tmp_path: Path) -> None:
    """Above the per-field cap, the weakest qualified tiles are cut."""
    many = [(x, y) for x in range(30) for y in range(20)]  # 600 tiles
    _write_run(tmp_path, "bot-1.events.jsonl", _run_lines("field01.gif", many))
    _write_run(tmp_path, "bot-2.events.jsonl", _run_lines("field01.gif", many))
    # One tile seen in a third run outranks the rest.
    _write_run(tmp_path, "bot-3.events.jsonl", _run_lines("field01.gif", [(0, 0)]))

    atlas = build_equipment_atlas(tmp_path)

    rows = atlas["field01.gif"]
    assert len(rows) == ATLAS_MAX_TILES_PER_FIELD
    assert rows[0] == [0, 0, 3]


def test_unreadable_and_roomless_artifacts_are_skipped(tmp_path: Path) -> None:
    """A run that never joined a room contributes nothing; unreadable
    artifacts are skipped by the reader seam without killing the build."""
    _write_run(tmp_path, "bot-1.events.jsonl", _run_lines("field01.gif", [(5, 5)]))
    _write_run(tmp_path, "bot-2.events.jsonl", _run_lines("field01.gif", [(5, 5)]))
    no_room = _run_lines("field01.gif", [(6, 6)]).splitlines()[1]
    _write_run(tmp_path, "bot-3.events.jsonl", no_room + "\n")
    _write_run(tmp_path, "bot-4.events.jsonl", "unreadable")

    original = _test_hooks.read_text

    def read(path: Path) -> str:
        if path.name == "bot-4.events.jsonl":
            raise OSError("simulated unreadable artifact")
        return original(path)

    _test_hooks.read_text = read
    try:
        atlas = build_equipment_atlas(tmp_path)
    finally:
        _test_hooks.read_text = original

    assert atlas == {"field01.gif": [[5, 5, 2]]}


def test_cli_writes_the_atlas_to_the_given_output(tmp_path: Path) -> None:
    """The CLI mines the given root and writes the given output path."""
    _write_run(tmp_path / "corpus", "bot-1.events.jsonl", _run_lines("field05.gif", [(58, 170)]))
    _write_run(tmp_path / "corpus", "bot-2.events.jsonl", _run_lines("field05.gif", [(58, 170)]))
    out = tmp_path / "out" / "atlas.json"
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: ["tankpit-equipment-atlas", str(tmp_path / "corpus"), str(out)]
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original_get_argv

    assert load_json_str(out.read_text(encoding="utf-8")) == {"field05.gif": [[58, 170, 2]]}


def test_malformed_snapshot_shapes_contribute_nothing(tmp_path: Path) -> None:
    """Every malformed belief shape is skipped, never mis-mined."""
    room = dump_json_str(
        {
            "diagnostic_kind": "session_room_joined",
            "field_image": "field01.gif",
        }
    )
    snapshots = [
        '["entity_alignment_sample"]',
        dump_json_str({"diagnostic_kind": "entity_alignment_sample"}),
        dump_json_str(
            {"diagnostic_kind": "entity_alignment_sample", "belief_containers_json": "[1]"}
        ),
        dump_json_str(
            {
                "diagnostic_kind": "entity_alignment_sample",
                "belief_containers_json": dump_json_str({"containers": 7}),
            }
        ),
        dump_json_str(
            {
                "diagnostic_kind": "entity_alignment_sample",
                "belief_containers_json": dump_json_str(
                    {
                        "containers": [
                            "not-a-dict",
                            {"x": 1, "y": 2, "is_fuel": True},
                            {"x": 3, "is_fuel": False},
                            {"x": 4, "y": 5, "is_fuel": False, "source": "viewport"},
                        ]
                    }
                ),
            }
        ),
    ]
    # Each snapshot must be the SAMPLED one, so give each its own run
    # (plus a twin so the one good tile clears the min-runs floor).
    for i, snap in enumerate(snapshots):
        _write_run(tmp_path, f"bot-{i}.events.jsonl", room + "\n" + snap + "\n")
    good = dump_json_str(
        {
            "diagnostic_kind": "entity_alignment_sample",
            "belief_containers_json": dump_json_str(
                {"containers": [{"x": 4, "y": 5, "is_fuel": False, "source": "viewport"}]}
            ),
        }
    )
    _write_run(tmp_path, "bot-9.events.jsonl", room + "\n" + good + "\n")
    # Malformed room-join shapes: a non-object line and a non-string
    # field image both leave the run field-less (and thus unmined).
    _write_run(
        tmp_path,
        "bot-10.events.jsonl",
        '["session_room_joined"]\n'
        + dump_json_str({"diagnostic_kind": "session_room_joined", "field_image": 7})
        + "\n"
        + good
        + "\n",
    )

    atlas = build_equipment_atlas(tmp_path)

    assert atlas == {"field01.gif": [[4, 5, 2]]}
