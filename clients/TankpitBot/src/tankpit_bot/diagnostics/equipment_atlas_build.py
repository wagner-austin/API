"""Build the per-field equipment atlas from the archived run corpus.

Full-corpus mining (2026-08-28, [[equipment-system]]) proved equipment
spawns cluster at persistent per-field hotspots. This builder turns
the whole ``runs/bot`` archive into the committed atlas the collect
planner navigates (``data/equipment_atlas.json``): for every field
image, every tile where an equipment container was sighted in the
bot's OWN viewport, weighted by how many independent sessions saw it.

CLI: ``tankpit-equipment-atlas [root]`` (default ``runs/bot``) writes
the atlas and prints the per-field summary. Rerun after new farm
sessions to fold fresh sightings in; commit the regenerated file.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.logging import get_logger
from platform_core.rich_logging import setup_rich_logging

from tankpit_bot import _test_hooks

log = get_logger(__name__)

ATLAS_OUTPUT_PATH = Path("data") / "equipment_atlas.json"
"""Where the committed atlas lives; the collect planner reads it."""

SNAPSHOT_SAMPLE_STRIDE = 20
"""Belief snapshots are dense (one per alignment tick); every Nth
carries the same long-lived container set at 5% of the parse cost."""

ATLAS_MIN_RUNS = 2
"""Tiles sighted in fewer independent runs are scatter, not atlas."""

ATLAS_MAX_TILES_PER_FIELD = 500
"""Strongest-first cap so the committed file stays reviewable; the
cut is logged per field, never silent."""


def _snapshot_tiles(line: str) -> set[tuple[int, int]]:
    """Extract own-viewport equipment tiles from one snapshot line.

    Args:
        line: One ``entity_alignment_sample`` JSONL line.

    Returns:
        The equipment tiles the snapshot's beliefs carry; fleet
        imports (``source == "world_state"``) are excluded so the
        atlas counts independent own-eyes runs.
    """
    tiles: set[tuple[int, int]] = set()
    decoded = load_json_str(line)
    if not isinstance(decoded, dict):
        return tiles
    blob = decoded.get("belief_containers_json")
    if not isinstance(blob, str):
        return tiles
    beliefs = load_json_str(blob)
    if not isinstance(beliefs, dict):
        return tiles
    containers = beliefs.get("containers")
    if not isinstance(containers, list):
        return tiles
    for container in containers:
        if not isinstance(container, dict):
            continue
        if container.get("is_fuel") is not False:
            continue
        if container.get("source") == "world_state":
            continue
        x, y = container.get("x"), container.get("y")
        if isinstance(x, int) and isinstance(y, int):
            tiles.add((x, y))
    return tiles


def _mine_events_text(text: str) -> tuple[str | None, set[tuple[int, int]]]:
    """Mine one events artifact's text for equipment-tile sightings.

    Args:
        text: Full JSONL text of one run's events artifact.

    Returns:
        The run's field image (None when the run never joined a room)
        and the set of own-viewport equipment tiles it sighted.
    """
    field: str | None = None
    tiles: set[tuple[int, int]] = set()
    samples = 0
    for line in text.splitlines():
        if field is None and '"session_room_joined"' in line:
            decoded = load_json_str(line)
            if isinstance(decoded, dict):
                image = decoded.get("field_image")
                if isinstance(image, str):
                    field = image
        if '"entity_alignment_sample"' not in line:
            continue
        samples += 1
        if samples % SNAPSHOT_SAMPLE_STRIDE != 1:
            continue
        tiles |= _snapshot_tiles(line)
    return field, tiles


def build_equipment_atlas(root: Path) -> dict[str, list[list[int]]]:
    """Mine every archived run under ``root`` into the atlas mapping.

    Args:
        root: Directory holding ``*.events.jsonl`` artifacts.

    Returns:
        ``{field_image: [[x, y, run_count], ...]}``, strongest first,
        tiles below :data:`ATLAS_MIN_RUNS` dropped and each field
        capped at :data:`ATLAS_MAX_TILES_PER_FIELD`.
    """
    run_counts: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    for source in sorted(root.rglob("*.events.jsonl")):
        if source.name == "latest.events.jsonl":
            continue
        try:
            text = _test_hooks.read_text(source)
        except OSError as error:
            log.warning("equipment atlas: skipping unreadable %s: %s", source, error)
            continue
        field, tiles = _mine_events_text(text)
        if field is None or not tiles:
            continue
        for tile in tiles:
            run_counts[field][tile] += 1
    atlas: dict[str, list[list[int]]] = {}
    for field, counts in sorted(run_counts.items()):
        rows = [[x, y, runs] for (x, y), runs in counts.most_common() if runs >= ATLAS_MIN_RUNS]
        if len(rows) > ATLAS_MAX_TILES_PER_FIELD:
            log.info(
                "equipment atlas: %s capped at %d tiles (%d qualified)",
                field,
                ATLAS_MAX_TILES_PER_FIELD,
                len(rows),
            )
            rows = rows[:ATLAS_MAX_TILES_PER_FIELD]
        atlas[field] = rows
    return atlas


def main() -> int:
    """Run the ``tankpit-equipment-atlas`` CLI entrypoint.

    Returns:
        0 on success (an empty corpus still writes an empty atlas).
    """
    setup_rich_logging(level="INFO")
    argv = _test_hooks.get_argv()
    root = Path(argv[1]) if len(argv) > 1 else Path("runs") / "bot"
    output = Path(argv[2]) if len(argv) > 2 else ATLAS_OUTPUT_PATH
    atlas = build_equipment_atlas(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump_json_str(dict(atlas), indent=0), encoding="utf-8")
    for field, rows in atlas.items():
        top = rows[0] if rows else None
        log.info("equipment atlas: %s -> %d tiles (top %s)", field, len(rows), top)
    log.info("equipment atlas written: %s", output)
    return 0


__all__ = [
    "ATLAS_MAX_TILES_PER_FIELD",
    "ATLAS_MIN_RUNS",
    "ATLAS_OUTPUT_PATH",
    "SNAPSHOT_SAMPLE_STRIDE",
    "build_equipment_atlas",
    "main",
]
