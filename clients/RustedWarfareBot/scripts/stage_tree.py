"""Describe the game tree, pack it, and write the record staging is held to.

Three outputs from one act, because they must agree and there is no honest way
to produce them apart:

* the **document** (:mod:`rw_bot.staged_tree`), saying which depot every file
  came from and which of them were checked against Steam's own installed copy;
* the **archive** (:mod:`rw_bot.tree_archive`), one reproducible file, which
  is what a stage manifest can actually name;
* the **digest record**, the ``--expect-from`` document ``hpc3-stage``
  requires -- written here, by this act, and read there by another.

WHY THE REFERENCE IS STEAM'S OWN INSTALL. Content depot 647961 declares no
``oslist``, so its files are the same bytes on Windows and on Linux. The
installed Windows copy is therefore an independent second copy of most of this
tree, placed by Steam rather than by anybody here, and comparing against it
turns "I assembled this" into a measurement for 1,516 of 1,774 files --
1,445 at the same path, 59 under the same name in the other platform's runtime
directory, and 12 as byte-identical copies under another name.

WHAT IT REFUSES. A file present in both at one path with different bytes stops
the run: no digest and no transfer check would ever notice that, because each
matches the copy it came from. The one exception is the settings file the game
rewrites on every boot, which is expected to differ -- and is nonetheless
STAGED, because the runner resets every worker's copy from it and a tree
without it fails on the compute node.

Run as ``python -m scripts.stage_tree --tree <dir> --reference <dir>
--build-id <id> --content-manifest <gid> --linux-manifest <gid>
--archive <path> --out <path> --digests <path> --manifest <path>
--destination <dir>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hpc3.contracts.stage import encode_stage_manifest
from platform_core.cli_args import parse_single_flags, require_flag
from platform_core.json_utils import dump_json_str

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks
from rw_bot.harness.clone import VOLATILE_FILES, entries_to_copy
from rw_bot.harness.jvm import tool_path
from rw_bot.stage_record import stage_manifest
from rw_bot.staged_tree import (
    APP_ID,
    CONTENT_DEPOT,
    LINUX_DEPOT,
    ROLE_CONTENT,
    ROLE_LINUX_RUNTIME,
    DepotRef,
    StagedTreeSpec,
    classify_entries,
    count_origins,
    encode_staged_tree_spec,
    render_digest_record,
)
from rw_bot.tree_archive import write_archive
from rw_bot.tree_identity import TreeEntry, tree_digest, tree_entries

#: The platform the staged tree is for. Stated rather than read from the
#: running interpreter: this describes a tree destined for a Linux node and is
#: run on a Windows workstation, so the two are deliberately different.
TARGET_PLATFORM = "linux"

#: Indent the document is written with. Committed and read by people.
DOCUMENT_INDENT = 2

#: Flags. Every one required: a document that defaulted a manifest GID would
#: pin nothing while looking as though it had.
REQUIRED_FLAGS = (
    "--tree",
    "--reference",
    "--build-id",
    "--content-manifest",
    "--linux-manifest",
    "--archive",
    "--out",
    "--digests",
    # Emitted rather than hand-kept, for the same reason every other digest
    # here is: a manifest naming an archive it was typed from is one edit
    # away from naming a different one. See :mod:`rw_bot.stage_record`.
    "--manifest",
    "--destination",
)

_RUNTIME_NOT_EXECUTABLE = "RW-STAGE-001"
_NO_RESET_STATE = "RW-STAGE-002"

EXIT_OK = 0


class StageTreeError(RwBotError):
    """The tree could not be prepared for staging.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what was wrong.
    """


def stageable_entries(entries: Sequence[TreeEntry]) -> tuple[TreeEntry, ...]:
    """Drop the entries a worker's copy would never take.

    The game rebuilds ``saves`` and ``cache`` on boot and never reads the
    32-bit JVM, so a clone takes none of the three
    (:mod:`rw_bot.harness.clone`). Asked of the same function the clone check
    asks, so the staged tree and a worker's copy cannot disagree about what
    belongs in a game directory.

    ``preferences.ini`` is DELIBERATELY KEPT, and getting that wrong is what
    this docstring used to say. The clone does not merely tolerate it -- the
    runner resets every worker by copying it out of the source directory, so a
    staged tree without it fails that reset on the compute node. What is
    staged is this workstation's copy, which is what makes it the experiment's
    pinned starting state rather than whatever the first node left behind.

    Args:
        entries: Every file found under the tree.

    Returns:
        Those whose top-level entry a clone would copy, in the order given.
    """
    top_level = sorted({entry["path"].split("/")[0] for entry in entries})
    kept = set(entries_to_copy(top_level))
    return tuple(entry for entry in entries if entry["path"].split("/")[0] in kept)


def check_reset_state_present(entries: Sequence[TreeEntry]) -> None:
    """Raise unless the tree carries every file a clone is reset from.

    :func:`rw_bot.harness.runner.reset_volatile` copies each of these out of
    the source directory into every worker's copy. A staged tree missing one
    fails there, on a compute node, once per member -- and the message would
    be about a file nobody staged rather than about a tree nobody checked.

    Args:
        entries: The entries about to be staged.

    Raises:
        StageTreeError: ``RW-STAGE-002`` when one is absent.
    """
    staged = {entry["path"] for entry in entries}
    absent = [name for name in VOLATILE_FILES if name not in staged]
    if absent:
        raise StageTreeError(
            _NO_RESET_STATE,
            f"the tree does not carry {', '.join(absent)}, which every worker's copy is "
            "reset from; each match on the cluster would fail on a file nobody staged",
        )


def depots(content_manifest: str, linux_manifest: str) -> list[DepotRef]:
    """Name the depots the tree draws on, each pinned by manifest.

    Args:
        content_manifest: Manifest GID of the platform-neutral content depot.
        linux_manifest: Manifest GID of the Linux runtime depot.

    Returns:
        The two references, content first.
    """
    return [
        DepotRef(depot_id=CONTENT_DEPOT, manifest=content_manifest, role=ROLE_CONTENT),
        DepotRef(depot_id=LINUX_DEPOT, manifest=linux_manifest, role=ROLE_LINUX_RUNTIME),
    ]


def check_runtime_is_executable(executables: Sequence[str]) -> None:
    """Raise unless the archive would extract a runnable ``java``.

    The tree was assembled on Windows, which has no executable bit, so the
    archive decides it -- by ELF magic, in
    :mod:`rw_bot.tree_archive`. If that rule ever fails to fire on the one
    file every match runs, the archive extracts a JVM nobody can execute and
    every member dies as permission denied on a compute node.

    It has never been seen precisely because it cannot be: WSL2 mounts
    ``/mnt/c`` with everything 777, so the workstation proof of the Linux
    launch ran on a filesystem where the bit did not matter.

    Args:
        executables: Members the archive wrote executable.

    Raises:
        StageTreeError: ``RW-STAGE-001`` when the bundled ``java`` is not
            among them.
    """
    java = tool_path("java", TARGET_PLATFORM)
    if java not in executables:
        raise StageTreeError(
            _RUNTIME_NOT_EXECUTABLE,
            f"{java} would extract without its executable bit, so every match on the "
            "cluster would fail to start it; the archive marks a member executable when "
            "it begins with the ELF magic number, and this one did not",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Describe, pack and vouch for the game tree.

    Args:
        argv: Argument list excluding the program name. ``None`` reads the
            process arguments.

    Returns:
        :data:`EXIT_OK`.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        TreeIdentityError: When either tree is absent, empty, or holds a
            symbolic link.
        StagedTreeError: ``RW-TREE-101`` when a file the game does not rewrite
            differs between the tree and the reference.
        StageTreeError: ``RW-STAGE-001`` when the packed runtime would not be
            executable, or ``RW-STAGE-002`` when the tree does not carry the
            file every worker's copy is reset from.
        OSError: When a file cannot be read or an output cannot be written.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    parsed = parse_single_flags(tokens, REQUIRED_FLAGS)
    for flag in REQUIRED_FLAGS:
        require_flag(parsed, flag)

    tree_root = Path(parsed["--tree"])
    reference_root = Path(parsed["--reference"])
    entries = stageable_entries(tree_entries(tree_root))
    check_reset_state_present(entries)
    # Unfiltered: the reference is only ever read FROM, and a rename may
    # legitimately resolve against any file Steam ships.
    classified = classify_entries(entries, tree_entries(reference_root), rewritten=VOLATILE_FILES)

    archive_path = Path(parsed["--archive"])
    # Real I/O rather than the hooked maker, because the archive beside it is
    # written with a real tarfile: a faked directory and a real file would be
    # two halves of one act disagreeing about where it happened.
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive = write_archive(tree_root, entries, archive_path)
    check_runtime_is_executable(archive["executables"])

    spec = StagedTreeSpec(
        app_id=APP_ID,
        build_id=parsed["--build-id"],
        depots=depots(parsed["--content-manifest"], parsed["--linux-manifest"]),
        reference=str(reference_root),
        entries=list(classified),
        tree_sha256=tree_digest(entries),
        archive_name=archive_path.name,
        archive_sha256=archive["sha256"],
        archive_size_bytes=archive["size_bytes"],
    )
    manifest = stage_manifest(
        parsed["--destination"],
        archive_path.name,
        archive,
        {
            "app_id": APP_ID,
            "build_id": spec["build_id"],
            "content_depot": f"{CONTENT_DEPOT}@{parsed['--content-manifest']}",
            "linux_depot": f"{LINUX_DEPOT}@{parsed['--linux-manifest']}",
            "tree_sha256": spec["tree_sha256"],
            "entries": str(len(classified)),
            "described_by": str(Path(parsed["--out"]).as_posix()),
        },
    )
    # Every output is the point of the run, so their directories are this
    # command's to make rather than one more thing for a caller to get right
    # before it will work at all.
    for flag in ("--out", "--digests", "--manifest"):
        _test_hooks.make_dirs(Path(parsed[flag]).parent)
    _test_hooks.write_text_lines(
        Path(parsed["--out"]),
        dump_json_str(encode_staged_tree_spec(spec), indent=DOCUMENT_INDENT).splitlines(),
    )
    _test_hooks.write_text_lines(Path(parsed["--digests"]), list(render_digest_record(spec)))
    _test_hooks.write_text_lines(
        Path(parsed["--manifest"]),
        dump_json_str(encode_stage_manifest(manifest), indent=DOCUMENT_INDENT).splitlines(),
    )

    counts = count_origins(classified)
    _test_hooks.write_line(
        f"[tree] {len(classified)} file(s): "
        + ", ".join(f"{counts[origin]} {origin}" for origin in sorted(counts))
    )
    _test_hooks.write_line(f"[tree] identity {spec['tree_sha256']}")
    _test_hooks.write_line(
        f"[tree] archive {spec['archive_name']} "
        f"{spec['archive_size_bytes']} bytes {spec['archive_sha256']}"
    )
    _test_hooks.write_line(f"[tree] {len(archive['executables'])} member(s) marked executable")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "DOCUMENT_INDENT",
    "EXIT_OK",
    "REQUIRED_FLAGS",
    "TARGET_PLATFORM",
    "StageTreeError",
    "check_reset_state_present",
    "check_runtime_is_executable",
    "depots",
    "main",
    "stageable_entries",
]
