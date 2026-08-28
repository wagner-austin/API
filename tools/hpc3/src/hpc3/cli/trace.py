"""CLI: connect a result back to the run that produced it.

Usage:
    hpc3-trace --config hpc3.json --match 07ab4976...   # which job trained this?
    hpc3-trace --config hpc3.json --job 55519937        # what did this job train?

Every other command in this package looks forward: submit a thing, watch it,
account for it. This one looks backward, which is the direction the question
actually gets asked in. An outcome file turns up months later and the question
is what corpus it came from, at which seed, from which base model. Answering
that from job names means trusting a string somebody typed.

The ledger is the only record that has both halves -- the job id the cluster
knows and the experiment identity the run declared -- so it is the only place
the question can be answered without re-deriving anything.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.experiment import format_experiment, matches
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core import ledger

_FLAGS = (_config.CONFIG_FLAG, "--match", "--job")


def _select(entries: Sequence[LedgerEntry], parsed: dict[str, str]) -> list[LedgerEntry]:
    """Choose the entries the caller asked about.

    Args:
        entries: Everything the ledger holds.
        parsed: Flags already read from the command line.

    Returns:
        Matching entries, oldest first.

    Raises:
        ValueError: If neither ``--match`` nor ``--job`` was given, or both
            were. Answering a question the caller did not ask is worse than
            refusing, because the answer looks authoritative either way.
    """
    has_match = "--match" in parsed
    has_job = "--job" in parsed
    if has_match == has_job:
        raise ValueError("exactly one of --match or --job is required")

    if has_job:
        wanted = parsed["--job"]
        return [entry for entry in entries if entry["job_id"] == wanted]

    needle = parsed["--match"]
    return [entry for entry in entries if matches(entry["experiment"], needle)]


def describe_image(digest: str | None) -> str:
    """Say which software produced a run, including when the row cannot.

    Args:
        digest: The recorded digest, ``""`` for a directory run, or None for
            a row written before the field existed.

    Returns:
        A line naming the image, or naming which of the two kinds of absence
        this is. They are reported differently on purpose: "ran outside any
        image" is an answer and "this row does not record it" is not, and a
        reader that saw one string for both would take the second for the
        first.
    """
    if digest is None:
        return "image unrecorded -- this row predates the field, do not read it as none"
    if digest == "":
        return "image none (directory environment)"
    return f"image {digest}"


def describe_artifact(artifact: str | None) -> str:
    """Say where the run was told to write its result.

    Args:
        artifact: The declared path, or None when the row names none.

    Returns:
        A line naming the path, or saying plainly that the row does not.
        Saying so is the point: an index whose artifact column is unset is
        the state that cannot answer "which file holds the answer", and it
        must be visible rather than rendered as a missing line.
    """
    if artifact is None:
        return "artifact not declared -- this row cannot say where the result went"
    return f"artifact {artifact}"


def main(argv: Sequence[str] | None = None) -> int:
    """Report which recorded runs match a job id or an identity value.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when at least one entry matched, 1 when none did. A
        question with no answer is not an error -- the run may predate the
        ledger -- but it must not read as "nothing was ever run", so the
        exit code distinguishes them.

    Raises:
        ValueError: If ``--config`` is missing, an argument is unknown, or
            neither/both of ``--match`` and ``--job`` were given.
        JSONTypeError: If the ledger holds a malformed record.
        AppError: If a record names a partition this cluster does not have.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)

    entries = ledger.read(pathlib.Path(workspace["ledger"]), workspace_cluster(workspace))
    found = _select(entries, parsed)

    for entry in found:
        _test_hooks.emit(f"{entry['job_id']} {entry['name']} submitted {entry['submitted_at']}")
        _test_hooks.emit(f"  {format_experiment(entry['experiment'])}")
        _test_hooks.emit(f"  {describe_image(entry['image_digest'])}")
        _test_hooks.emit(f"  {describe_artifact(entry['artifact'])}")
        _test_hooks.emit(f"  logs {entry['log_dir']}")

    if found == []:
        _test_hooks.emit(f"no recorded run matches; the ledger holds {len(entries)} entry(s)")
        return 1

    _test_hooks.emit(f"{len(found)} of {len(entries)} recorded run(s) match")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
