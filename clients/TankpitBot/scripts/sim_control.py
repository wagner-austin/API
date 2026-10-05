"""Record and compare the sim's N=1 control: is a change byte-identical at one client?

A change to the sim's plumbing -- the multiplayer track's registry and
fan-out (board task b008ab91) is the worked example -- has to leave a
one-client session exactly as it was. The sim is byte-deterministic
([[capture-differ]]), so that is checkable rather than arguable: play a
fixed set of sessions on the old tree and on the new one and compare
every artifact they write.

Two things make the comparison honest.

* **The world is NAMED.** Every session plays one layout and one
  population seed, never the stamp's derivation, so the two trees play
  the same room ([[sim-world-parameterization]]).
* **Only the clock is stripped.** Wall-clock time leaks into three
  places: the capture's ``start_timestamp_ms``/``end_timestamp_ms`` and
  each frame's ``timestamp_ms``, every event record's ``timestamp``, and
  the ``timestamp_ms`` inside an event's ``*_json`` text field (the
  client-structure survey). Two runs of one tree minutes apart differ in
  those and in nothing else, measured 2026-10-05 on the ghost
  self-replay of ``bot-20260802-205105`` (314 frames, 1,865 events).
  Everything else -- every payload byte, every frame's direction and
  order, every event field -- is digested.

The set is :data:`scripts.build_sim_baseline.SCENARIOS` plus a ghost
self-replay of one archived capture, the control the 2026-08-03
refusal-law lift used. Each session writes a capture, a world and an
event stream, and each of those becomes one digest in the manifest.

To control a change, record the tree before it and the tree after it,
then compare. The older tree is played by putting its ``src`` first on
the path, e.g. ``PYTHONPATH=<old>/src poetry run python -m
scripts.sim_control record runs/sim-control/before.json``.

Usage:
    poetry run python -m scripts.sim_control record MANIFEST [--rounds R] [--ghost CAPTURE]
    poetry run python -m scripts.sim_control compare BEFORE AFTER
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from scripts import _test_hooks as script_hooks
from scripts.build_sim_baseline import SCENARIOS
from tankpit_bot import _test_hooks
from tankpit_bot.sim.run import SimRunResultDict, run_sim_session

#: The practice layout every control session plays, named so both trees
#: play the same room.
CONTROL_LAYOUT = "bot-20260706-223721"

#: The container field's determinism seed for every control session.
CONTROL_POPULATION_SEED = 7

#: Server ticks per session: the ghost self-replay's standing length.
DEFAULT_ROUNDS = 150

#: The archived capture the ghost self-replay plays, the one the
#: 2026-08-03 refusal-law lift was controlled against.
DEFAULT_GHOST_CAPTURE = Path("runs") / "bot" / "bot-20260802-205105.capture_session.json"

#: Where each recording's sessions are archived, one directory per
#: manifest so the two trees never share an artifact.
CONTROL_ROOT = Path("runs") / "sim-control"

#: The ghost session's label in the manifest.
GHOST_LABEL = "ghost"

#: Keys whose values are wall-clock time, removed at every depth.
CLOCK_KEYS: frozenset[str] = frozenset(
    {"timestamp", "timestamp_ms", "start_timestamp_ms", "end_timestamp_ms"}
)

#: Suffix of an event field that carries a JSON document as text.
_JSON_TEXT_SUFFIX = "_json"


class ControlArtifactDict(TypedDict):
    """One digested artifact.

    Attributes:
        label: The session that wrote it: a scenario label or ``ghost``.
        kind: Which of the session's three artifacts it is.
        sha256: Hex digest of its clock-stripped canonical form.
    """

    label: str
    kind: Literal["capture", "world", "events"]
    sha256: str


class ControlManifestDict(TypedDict):
    """Everything one recording played and what it produced.

    Attributes:
        rounds: Server ticks per session.
        layout: The practice layout every session played.
        population_seed: The container seed every session played.
        ghost_capture: The archived capture the ghost session replayed.
        artifacts: One digest per artifact, in play order.
    """

    rounds: int
    layout: str
    population_seed: int
    ghost_capture: str
    artifacts: list[ControlArtifactDict]


def _require_kind(obj: dict[str, JSONValue]) -> Literal["capture", "world", "events"]:
    """Read an artifact's ``kind`` and refuse anything but the three.

    Args:
        obj: The artifact object.

    Returns:
        The kind.

    Raises:
        JSONTypeError: If ``kind`` is missing or names no artifact.
    """
    kind = require_str(obj, "kind")
    if kind == "capture":
        return "capture"
    if kind == "world":
        return "world"
    if kind == "events":
        return "events"
    raise JSONTypeError(f"SIM_CONTROL_KIND: artifact kind {kind!r} is not capture, world or events")


def encode_manifest(manifest: ControlManifestDict) -> str:
    """Serialize a manifest, one artifact per line for a readable diff.

    Args:
        manifest: The manifest.

    Returns:
        Its JSON text.
    """
    artifacts: list[JSONValue] = [
        {"label": a["label"], "kind": a["kind"], "sha256": a["sha256"]}
        for a in manifest["artifacts"]
    ]
    document: dict[str, JSONValue] = {
        "rounds": manifest["rounds"],
        "layout": manifest["layout"],
        "population_seed": manifest["population_seed"],
        "ghost_capture": manifest["ghost_capture"],
        "artifacts": artifacts,
    }
    return dump_json_str(document, indent=2) + "\n"


def decode_manifest(text: str) -> ControlManifestDict:
    """Parse and validate a manifest.

    Args:
        text: JSON text written by :func:`encode_manifest`.

    Returns:
        The manifest.

    Raises:
        JSONTypeError: If a field is missing or of the wrong type.
    """
    obj = narrow_json_to_dict(load_json_str(text))
    artifacts: list[ControlArtifactDict] = []
    for raw in require_list(obj, "artifacts"):
        entry = narrow_json_to_dict(raw)
        artifacts.append(
            ControlArtifactDict(
                label=require_str(entry, "label"),
                kind=_require_kind(entry),
                sha256=require_str(entry, "sha256"),
            )
        )
    return ControlManifestDict(
        rounds=require_int(obj, "rounds"),
        layout=require_str(obj, "layout"),
        population_seed=require_int(obj, "population_seed"),
        ghost_capture=require_str(obj, "ghost_capture"),
        artifacts=artifacts,
    )


def strip_clock(value: JSONValue) -> JSONValue:
    """Remove every wall-clock field from a decoded artifact.

    A string under a key ending ``_json`` is decoded and stripped too,
    because the client-structure survey carries its own ``timestamp_ms``
    inside one. A value there that is not JSON raises: the field's name
    promises a document, and an artifact that breaks that promise is not
    one to digest quietly.

    Args:
        value: A decoded JSON value.

    Returns:
        The same value with every :data:`CLOCK_KEYS` entry removed.

    Raises:
        InvalidJsonError: If a ``*_json`` string is not JSON.
    """
    if isinstance(value, dict):
        stripped: dict[str, JSONValue] = {}
        for key, item in value.items():
            if key in CLOCK_KEYS:
                continue
            if key.endswith(_JSON_TEXT_SUFFIX) and isinstance(item, str):
                stripped[key] = strip_clock(load_json_str(item))
            else:
                stripped[key] = strip_clock(item)
        return stripped
    if isinstance(value, list):
        return [strip_clock(item) for item in value]
    return value


def canonical_digest(text: str, kind: Literal["capture", "world", "events"]) -> str:
    """Digest one artifact's clock-stripped canonical form.

    Args:
        text: The artifact's text as written.
        kind: ``events`` is JSON lines; the other two are one document.

    Returns:
        The hex SHA-256 of the stripped artifact, re-serialized compactly.
    """
    if kind == "events":
        lines: list[JSONValue] = [
            strip_clock(load_json_str(line)) for line in text.splitlines() if line
        ]
        canonical = dump_json_str(lines)
    else:
        canonical = dump_json_str(strip_clock(load_json_str(text)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _digest_session(label: str, result: SimRunResultDict) -> list[ControlArtifactDict]:
    """Digest the three artifacts one session wrote.

    Args:
        label: The session's label.
        result: What :func:`run_sim_session` returned.

    Returns:
        The capture, world and events digests, in that order.
    """
    paths: tuple[tuple[Literal["capture", "world", "events"], str], ...] = (
        ("capture", result["capture_path"]),
        ("world", result["world_path"]),
        ("events", result["events_path"]),
    )
    return [
        ControlArtifactDict(
            label=label,
            kind=kind,
            sha256=canonical_digest(_test_hooks.read_text(Path(path)), kind),
        )
        for kind, path in paths
    ]


def record_control(rounds: int, ghost_capture: Path, archive_dir: Path) -> ControlManifestDict:
    """Play the control set and digest everything it wrote.

    Args:
        rounds: Server ticks per session.
        ghost_capture: The archived capture the ghost session replays.
        archive_dir: Directory this recording's sessions are archived in;
            each session's events land under its own subdirectory so no
            session's ``latest`` stream overwrites another's.

    Returns:
        The manifest.
    """
    artifacts: list[ControlArtifactDict] = []
    for scenario in SCENARIOS:
        label = scenario["label"]
        result = run_sim_session(
            rounds,
            archive_dir=archive_dir,
            opponent=scenario["opponent"],
            practice=scenario["practice"],
            ferry=scenario["ferry"],
            larder=scenario["larder"],
            opponent_name=scenario["opponent_name"],
            stamp=f"control-{label}",
            layout=CONTROL_LAYOUT,
            population_seed=CONTROL_POPULATION_SEED,
            runs_root=str(archive_dir / label),
        )
        artifacts.extend(_digest_session(label, result))
    ghost = run_sim_session(
        rounds,
        archive_dir=archive_dir,
        ghost=str(ghost_capture),
        stamp=f"control-{GHOST_LABEL}",
        layout=CONTROL_LAYOUT,
        population_seed=CONTROL_POPULATION_SEED,
        runs_root=str(archive_dir / GHOST_LABEL),
    )
    artifacts.extend(_digest_session(GHOST_LABEL, ghost))
    return ControlManifestDict(
        rounds=rounds,
        layout=CONTROL_LAYOUT,
        population_seed=CONTROL_POPULATION_SEED,
        ghost_capture=ghost_capture.as_posix(),
        artifacts=artifacts,
    )


def compare_manifests(before: ControlManifestDict, after: ControlManifestDict) -> list[str]:
    """Say every way two recordings differ.

    Args:
        before: The recording of the tree before the change.
        after: The recording of the tree after it.

    Returns:
        One line per difference; empty when the two are identical.
    """
    differences: list[str] = []
    for field in ("rounds", "layout", "population_seed", "ghost_capture"):
        if before[field] != after[field]:
            differences.append(f"played differently: {field} {before[field]!r} vs {after[field]!r}")
    old = {(a["label"], a["kind"]): a["sha256"] for a in before["artifacts"]}
    new = {(a["label"], a["kind"]): a["sha256"] for a in after["artifacts"]}
    for key in sorted(old.keys() | new.keys()):
        label, kind = key
        if key not in new:
            differences.append(f"{label} {kind}: only before")
        elif key not in old:
            differences.append(f"{label} {kind}: only after")
        elif old[key] != new[key]:
            differences.append(f"{label} {kind}: {old[key][:12]} vs {new[key][:12]}")
    return differences


def _flag(argv: list[str], flag: str) -> str | None:
    """Read the value after one flag, or None when the flag is absent.

    Args:
        argv: The tokens after the subcommand's positionals.
        flag: The flag to read.

    Returns:
        The flag's value, or None.

    Raises:
        SystemExit: If the flag is the last token, with no value after it.
    """
    if flag not in argv:
        return None
    index = argv.index(flag) + 1
    if index >= len(argv):
        sys.stdout.write(f"SIM_CONTROL_USAGE: {flag} needs a value\n")
        raise SystemExit(2)
    return argv[index]


def _record(argv: list[str]) -> int:
    """Run ``record MANIFEST [--rounds R] [--ghost CAPTURE]``.

    Args:
        argv: The tokens after ``record``.

    Returns:
        0 once the manifest is written.

    Raises:
        SystemExit: If the manifest path is missing or the ghost capture
            does not exist; a control without its ghost is not the
            control, so it is refused rather than played short.
    """
    if not argv or argv[0].startswith("--"):
        sys.stdout.write("SIM_CONTROL_USAGE: record MANIFEST [--rounds R] [--ghost CAPTURE]\n")
        raise SystemExit(2)
    manifest_path = Path(argv[0])
    rounds_text = _flag(argv[1:], "--rounds")
    rounds = int(rounds_text) if rounds_text is not None else DEFAULT_ROUNDS
    ghost_text = _flag(argv[1:], "--ghost")
    ghost_capture = Path(ghost_text) if ghost_text is not None else DEFAULT_GHOST_CAPTURE
    if not _test_hooks.path_exists(ghost_capture):
        sys.stdout.write(
            f"SIM_CONTROL_GHOST_MISSING: {ghost_capture} does not exist; the archived "
            "captures live on diphtheria at /mnt/archive-a/austinpc/tankpitbot-runs/runs\n"
        )
        raise SystemExit(1)
    archive_dir = CONTROL_ROOT / manifest_path.stem
    manifest = record_control(rounds, ghost_capture, archive_dir)
    _test_hooks.write_text(manifest_path, encode_manifest(manifest))
    sys.stdout.write(
        f"recorded {len(manifest['artifacts'])} artifacts from "
        f"{len(SCENARIOS) + 1} sessions x {rounds} rounds: {manifest_path}\n"
    )
    return 0


def _compare(argv: list[str]) -> int:
    """Run ``compare BEFORE AFTER``.

    Args:
        argv: The tokens after ``compare``.

    Returns:
        0 when every artifact is identical, 1 when any differs.

    Raises:
        SystemExit: If the two manifest paths are not given.
    """
    if len(argv) != 2:
        sys.stdout.write("SIM_CONTROL_USAGE: compare BEFORE AFTER\n")
        raise SystemExit(2)
    before = decode_manifest(_test_hooks.read_text(Path(argv[0])))
    after = decode_manifest(_test_hooks.read_text(Path(argv[1])))
    differences = compare_manifests(before, after)
    if differences:
        sys.stdout.write(f"DIVERGED on {len(differences)}:\n")
        for line in differences:
            sys.stdout.write(f"  {line}\n")
        return 1
    sys.stdout.write(f"IDENTICAL: all {len(after['artifacts'])} artifacts\n")
    return 0


def main(argv: list[str]) -> int:
    """Dispatch to ``record`` or ``compare``.

    Args:
        argv: Command-line tokens excluding the program name.

    Returns:
        The subcommand's exit code.

    Raises:
        SystemExit: If no known subcommand is named.
    """
    script_hooks.setup_rich_logging(level=script_hooks.LogLevel.WARNING)
    if argv and argv[0] == "record":
        return _record(argv[1:])
    if argv and argv[0] == "compare":
        return _compare(argv[1:])
    sys.stdout.write("SIM_CONTROL_USAGE: sim_control record|compare ...\n")
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "CLOCK_KEYS",
    "CONTROL_LAYOUT",
    "CONTROL_POPULATION_SEED",
    "CONTROL_ROOT",
    "DEFAULT_GHOST_CAPTURE",
    "DEFAULT_ROUNDS",
    "GHOST_LABEL",
    "ControlArtifactDict",
    "ControlManifestDict",
    "canonical_digest",
    "compare_manifests",
    "decode_manifest",
    "encode_manifest",
    "main",
    "record_control",
    "strip_clock",
]
