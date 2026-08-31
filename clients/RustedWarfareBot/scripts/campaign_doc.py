"""Write the campaign document a batch becomes on the cluster.

Reads the job file a sweep already has and emits the document
``hpc3-campaign`` converges on -- one member per match, each declaring the
scorecard it writes. Nothing is submitted here: this produces the document,
and the cluster tooling computes what of it is still missing.

THE DOCUMENT CARRIES ONLY WHAT A BATCH CAN SAY FOR ITSELF. ``hpc3`` merges it
with the project's declared defaults from its own workspace -- the partition,
the wall clock, the image, the environment -- which live on the submitting
machine and not in this repository. Writing them here would be this package
guessing at another's configuration, and the guess would be silently wrong the
first time that configuration changed.

So this emits ``project``, ``name``, ``experiment`` and ``members``, which is
exactly the set ``hpc3.contracts.run.SWEEP_IDENTITY_FIELDS`` names as a
sweep's own, and validation happens where the workspace is.

THE ONE CLUSTER FACT IT NEEDS IS READ, NOT TYPED. A member's paths must be
absolute: ``sbatch`` sets no working directory and the campaign's existence
check runs from a login shell's ``$HOME``, so a relative artifact reads as
missing forever and the campaign resubmits the whole batch on every pass. The
root those paths hang off is the workspace's own ``root``, so this takes
``--config`` and reads it -- which also means the project must already be
declared there. A hand-typed root would be a second copy of the one fact both
sides have to agree on, and it would be wrong on the day the workspace moved.

Run as ``python -m scripts.campaign_doc --config <hpc3.json> --jobs <file>
--batch <name> --out <path>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hpc3.contracts.workspace import decode_workspace, require_project_config
from platform_core.cli_args import parse_single_flags, require_flag
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from rw_bot.harness import _test_hooks
from rw_bot.harness.campaign import campaign_members
from rw_bot.harness.match import MatchConfig, decode_match_config
from rw_bot.harness.sweep import SweepJob, parse_jobs
from scripts.sweep import DEFAULT_LOCKSTEP

#: Where an environment keeps its interpreter, appended to the project's
#: declared ``env_path``. Absolute, because the environment the experiment is
#: pinned to is the image's virtualenv and whatever bare ``python`` resolves
#: to on a compute node is not it -- but read from the workspace rather than
#: written out, so a project that moves its environment moves the command
#: with it instead of emitting one that points at where it used to be.
INTERPRETER_SUFFIX = "bin/python"

#: The hpc3 project a rusted campaign belongs to. Its defaults -- partition,
#: wall clock, image, root, environment -- are declared in the workspace,
#: which is where a statement about the cluster belongs.
PROJECT = "rusted"

#: Indent the document is written with. Committed beside the job file and read
#: by people, so it is formatted for them.
DOCUMENT_INDENT = 2

#: Flags naming the workspace to read the root from, the batch to emit, and
#: where to put the document.
REQUIRED_FLAGS = (
    "--config",
    "--jobs",
    "--batch",
    "--map",
    "--difficulty",
    "--fast-forward",
    "--out",
)

#: Opponents asked for, matching the sweep entry point's. The engine caps the
#: count by the map's own team count, so a two-player map is a duel regardless.
DUEL_OPPONENTS = 1

EXIT_OK = 0


def experiment_of(
    batch: str, jobs: Sequence[SweepJob], lockstep: int, fast_forward: int, match: MatchConfig
) -> dict[str, str]:
    """Describe what this campaign IS, for the ledger.

    A job id and a name say which row in ``squeue`` a member was and nothing
    about which experiment it belonged to. These are the facts that
    distinguish one batch of this project from another.

    Args:
        batch: The sweep's name.
        jobs: Every match the file describes.
        lockstep: Engine frames between samples.
        fast_forward: Wall-clock multiple every member runs at, zero for
            realtime. Recorded because pace is part of the regime the batch
            ran under, and a ledger row that omitted it would let a
            fast-forwarded batch read as comparable to a realtime one.
        match: Which match every member plays. Recorded because the map
            decides the opponent count and is therefore part of what the
            batch measured, not a runtime detail -- two batches on different
            maps are not comparable and the ledger should say which was which.

    Returns:
        The experiment's key/value pairs, every value a string because that is
        what the ledger stores.
    """
    return {
        "batch": batch,
        "matches": str(len(jobs)),
        "arms": ",".join(sorted({job["label"] for job in jobs})),
        "lockstep": str(lockstep),
        "fast_forward": str(fast_forward),
        "map": match["map_path"],
        "difficulty": str(match["difficulty"]),
    }


def interpreter_of(env_path: str) -> str:
    """Return the Python a member runs under, given its declared environment.

    Args:
        env_path: The project's ``env_path``, from the workspace. Inside the
            image when the project declares one.

    Returns:
        The absolute interpreter path.
    """
    return f"{env_path.rstrip('/')}/{INTERPRETER_SUFFIX}"


def campaign_document(
    root: str,
    env_path: str,
    jobs_file: str,
    batch: str,
    jobs: Sequence[SweepJob],
    lockstep: int,
    fast_forward: int,
    match: MatchConfig,
) -> dict[str, JSONValue]:
    """Build the document a batch is submitted as.

    Args:
        root: The cluster root, read from the workspace.
        env_path: The project's environment, read from the workspace.
        jobs_file: The batch's job file, relative to the repository root.
        batch: The sweep's name, which names the campaign.
        jobs: Every match the file describes.
        lockstep: Engine frames between samples.
        fast_forward: Wall-clock multiple every member runs at, zero for
            realtime.
        match: Which match every member plays.

    Returns:
        The document, carrying only a sweep's own identity fields.

    Raises:
        ValueError: When the batch has no jobs.
    """
    members: list[JSONValue] = [
        {
            "suffix": member["suffix"],
            "command": member["command"],
            "artifact": member["artifact"],
        }
        for member in campaign_members(
            interpreter_of(env_path),
            root,
            PROJECT,
            jobs_file,
            batch,
            jobs,
            lockstep,
            fast_forward,
            match,
        )
    ]
    return {
        "project": PROJECT,
        "name": batch,
        "experiment": dict(experiment_of(batch, jobs, lockstep, fast_forward, match)),
        "members": members,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write a batch's campaign document.

    Args:
        argv: Argument list excluding the program name. ``None`` reads the
            process arguments.

    Returns:
        :data:`EXIT_OK`.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        JSONTypeError: When the workspace document is malformed.
        AppError: When the workspace does not declare :data:`PROJECT`. Refused
            here rather than at submission, because a document naming a
            project the workspace has never heard of cannot be resolved into
            a job at all.
        SweepError: When the job file is malformed.
        ValueError: When the batch has no jobs.
        OSError: When a file cannot be read or the document written.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    parsed = parse_single_flags(tokens, REQUIRED_FLAGS)
    for flag in REQUIRED_FLAGS:
        require_flag(parsed, flag)

    raw = "\n".join(_test_hooks.read_text_lines(Path(parsed["--config"])))
    workspace = decode_workspace(load_json_str(raw), config_dir=Path(parsed["--config"]).parent)
    # Read for its refusal as much as for its value: a workspace with no
    # `rusted` project would otherwise emit a document that submits nothing.
    config = require_project_config(workspace, PROJECT)

    jobs_file = parsed["--jobs"]
    batch = parsed["--batch"]
    jobs = parse_jobs(_test_hooks.read_text_lines(Path(jobs_file)))
    match = decode_match_config(
        {
            "map_path": parsed["--map"],
            "opponents": DUEL_OPPONENTS,
            "difficulty": int(parsed["--difficulty"]),
        }
    )
    document = campaign_document(
        workspace["root"],
        config["env_path"],
        jobs_file,
        batch,
        jobs,
        DEFAULT_LOCKSTEP,
        int(parsed["--fast-forward"]),
        match,
    )
    _test_hooks.write_text_lines(
        Path(parsed["--out"]), dump_json_str(document, indent=DOCUMENT_INDENT).splitlines()
    )
    _test_hooks.write_line(f"[campaign] {len(jobs)} member(s) -> {parsed['--out']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "DOCUMENT_INDENT",
    "EXIT_OK",
    "INTERPRETER_SUFFIX",
    "PROJECT",
    "REQUIRED_FLAGS",
    "campaign_document",
    "experiment_of",
    "interpreter_of",
    "main",
]
