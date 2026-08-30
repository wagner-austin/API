"""Freeze the code a batch runs, pack it, and write the record it is staged to.

THE FREEZE MOVES TO BEFORE SUBMISSION, and that is the whole point. On a
workstation a batch freezes its own snapshot at launch, because the working
tree is editable while the batch runs -- an edit landed mid-batch used to mean
later matches ran different code from earlier ones. A compute node has the
opposite problem: :func:`~rw_bot.harness.runner.prepare_tree` copies
``src/rw_bot``, ``scripts``, ``doctrines`` and the agent jar from
REPOSITORY-RELATIVE paths, and a node has no repository. The freeze there
would copy nothing and every member would fail importing its own planner.

So it is done here, once, and staged. That is strictly better provenance than
the workstation path rather than a workaround for it: the frozen tree becomes
a digest-pinned artifact like the game, named in the record, verified on both
sides of the transfer, and re-checkable on the cluster with coreutils.

THE AGENT JAR IS WHY THIS CANNOT BE SKIPPED. The Linux depot ships a JRE with
no ``javac`` and no ``jar`` (:func:`~rw_bot.harness.jvm.bundled_tools`), so a
node cannot build the Java agent a match attaches. It has to arrive built, and
this is what carries it.

WHAT THE PROVENANCE IS. Not depots -- this tree comes from a repository, so
the honest pin is the commit. Given rather than read from ``git``: the tree
being frozen is the WORKING one, which may carry edits the commit does not,
and a command that quietly stamped HEAD onto a dirty tree would print a
reassuring lie. The caller states what it is staging.

Run as ``python -m scripts.stage_payload --tree <dir> --commit <sha>
--archive <path> --out <path> --digests <path> --manifest <path>
--destination <dir>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hpc3.contracts.stage import encode_stage_manifest
from platform_core.cli_args import parse_single_flags, require_flag
from platform_core.json_utils import JSONValue, dump_json_str

from rw_bot.harness import _test_hooks
from rw_bot.harness.agent_build import JAVA_RELEASE
from rw_bot.harness.results_layout import PINNED_GAME_DIR, TRACE_ROOT
from rw_bot.harness.runner import (
    TREE_SOURCES,
    check_frozen_tree,
    decode_sweep_config,
    prepare_tree,
)
from rw_bot.stage_record import stage_manifest
from rw_bot.tree_archive import write_archive
from rw_bot.tree_identity import digest_record, encode_tree_entry, tree_digest, tree_entries

#: Indent the document is written with. Committed and read by people.
DOCUMENT_INDENT = 2

#: What a Java release number adds to reach its class-file version: 8 is 52,
#: 11 is 55. The relation the depot's ceiling is expressed in.
CLASS_FILE_OFFSET = 44

#: What the manifest says about the agent jar it carries.
#:
#: Read from the builder rather than typed, because the number is the whole
#: claim: the Linux depot's JRE 1.8.0_131 reads class-file versions up to 52
#: and a jar built one release higher is a ``FATAL ERROR in native method``
#: before the game starts, on every member of every campaign.
AGENT_BYTECODE = (
    f"java {JAVA_RELEASE} "
    f"(class {int(JAVA_RELEASE) + CLASS_FILE_OFFSET}), the Linux depot JRE ceiling"
)

REQUIRED_FLAGS = (
    "--tree",
    "--commit",
    "--archive",
    "--out",
    "--digests",
    # Emitted rather than hand-kept. The manifest naming this archive was
    # written by hand and went stale the first time the tree gained a file:
    # the payload grew from 407 entries to 409, the archive changed digest
    # and length, and the manifest went on naming the old ones.
    "--manifest",
    "--destination",
)

EXIT_OK = 0


def freeze_config(tree: str) -> dict[str, str | int]:
    """Build the batch configuration :func:`prepare_tree` freezes from.

    Every field but the tree is inert here -- nothing is played -- but they
    are given real values rather than placeholders, because the decoder
    refuses empties and a payload of lies is harder to read later than one
    that says what it is.

    Args:
        tree: Where to write the frozen snapshot.

    Returns:
        The payload the sweep config decoder reads.
    """
    return {
        "out_dir": tree,
        "traces": TRACE_ROOT,
        "workers": 1,
        "lockstep": 1,
        "clone_prefix": ".game-w",
        "source_game_dir": PINNED_GAME_DIR,
        "tree": tree,
        "pin_delta": 0,
        "fast_forward": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze, pack and vouch for the code a batch runs.

    Args:
        argv: Argument list excluding the program name. ``None`` reads the
            process arguments.

    Returns:
        :data:`EXIT_OK`.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        SweepError: ``RW-SWEEP-006`` when the frozen tree is incomplete.
        TreeIdentityError: When the tree is empty or holds a symbolic link.
        OSError: When a source cannot be read or an output cannot be written.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    parsed = parse_single_flags(tokens, REQUIRED_FLAGS)
    for flag in REQUIRED_FLAGS:
        require_flag(parsed, flag)

    root = Path(parsed["--tree"])
    prepare_tree(decode_sweep_config(freeze_config(parsed["--tree"])))
    check_frozen_tree(root)

    entries = tree_entries(root)
    archive_path = Path(parsed["--archive"])
    # Real I/O rather than the hooked maker, because the archive beside it is
    # written with a real tarfile: a faked directory and a real file would be
    # two halves of one act disagreeing about where it happened.
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive = write_archive(root, entries, archive_path)
    identity = tree_digest(entries)

    document: dict[str, JSONValue] = {
        "git_commit": parsed["--commit"],
        "carries": list(TREE_SOURCES),
        "entries": [encode_tree_entry(entry) for entry in entries],
        "tree_sha256": identity,
        "archive_name": archive_path.name,
        "archive_sha256": archive["sha256"],
        "archive_size_bytes": archive["size_bytes"],
    }
    manifest = stage_manifest(
        parsed["--destination"],
        archive_path.name,
        archive,
        {
            "git_commit": parsed["--commit"],
            "carries": ", ".join(TREE_SOURCES),
            "tree_sha256": identity,
            "entries": str(len(entries)),
            "agent_bytecode": AGENT_BYTECODE,
            "described_by": str(Path(parsed["--out"]).as_posix()),
        },
    )
    # Every output is the point of the run, so their directories are this
    # command's to make rather than one more thing for a caller to get right
    # before it will work at all.
    for flag in ("--out", "--digests", "--manifest"):
        _test_hooks.make_dirs(Path(parsed[flag]).parent)
    _test_hooks.write_text_lines(
        Path(parsed["--manifest"]),
        dump_json_str(encode_stage_manifest(manifest), indent=DOCUMENT_INDENT).splitlines(),
    )
    _test_hooks.write_text_lines(
        Path(parsed["--out"]), dump_json_str(document, indent=DOCUMENT_INDENT).splitlines()
    )
    _test_hooks.write_text_lines(
        Path(parsed["--digests"]),
        list(
            digest_record(
                [
                    f"# rw_bot payload at commit {parsed['--commit']}",
                    f"# carries {', '.join(TREE_SOURCES)} and src/rw_bot",
                    f"# tree {identity}",
                ],
                archive_path.name,
                archive["sha256"],
                entries,
            )
        ),
    )

    _test_hooks.write_line(f"[payload] {len(entries)} file(s) frozen at {root}")
    _test_hooks.write_line(f"[payload] identity {identity}")
    _test_hooks.write_line(
        f"[payload] archive {archive_path.name} {archive['size_bytes']} bytes {archive['sha256']}"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "AGENT_BYTECODE",
    "CLASS_FILE_OFFSET",
    "DOCUMENT_INDENT",
    "EXIT_OK",
    "REQUIRED_FLAGS",
    "freeze_config",
    "main",
]
