"""Refuse a run whose command names a cluster file that is not there.

WHY THIS EXISTS. ``hpc3-stage`` puts a corpus on the cluster behind four
checks: the local digest, the cluster-side digest after transfer, an
``--expect-from`` record published independently of the manifest, and a
certification file written beside the data. The PAYLOAD gets none of them.
It is copied by hand, and it is the document that decides what trains --
model, strategy, learning rate, seed, and which corpus digest to read.

Measured, 2026-09-07: job 55806418 was submitted with both corpora staged
and certified, and died fourteen seconds later on

    FileNotFoundError: '/pub/.../payloads/code-style-qlora-v2.json'

because nothing stages a payload. Preflight had passed: it answers "would
the scheduler admit this", which is a different question from "do the files
this command names exist".

THE ABSENT FILE IS THE LOUD CASE. A WRONG file present is the quiet one, and
this does not solve that -- see the task for the certification work. What
this removes is the class where the answer was knowable before a job id was
issued and nobody asked.

INPUTS ARE TOLD FROM OUTPUTS BY THE FLAG, NOT BY THE DIRECTORY. A run command
names both: ``--payload`` and ``--spec`` take files that must already exist,
while ``--record`` and ``--out-dir`` name what the job will WRITE, and
requiring those to exist would refuse every first run of everything.
``payloads/`` versus ``results/`` is a naming convention, and a convention
believed by a checker is exactly the material this whole class of defect is
made of. The flag is a fact about the command; the directory is a habit.
"""

from __future__ import annotations

import re

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core.remote import run_remote

#: Flags whose value is a file the job READS. A run naming one of these must
#: have that file on the cluster before it is submitted.
INPUT_FLAGS: tuple[str, ...] = ("--payload", "--spec", "--items", "--corpus")

#: Flags whose value the job WRITES. Never required to exist. Kept as an
#: explicit list rather than "anything not an input" so that a flag nobody
#: has classified is reported instead of silently trusted.
OUTPUT_FLAGS: tuple[str, ...] = (
    "--record",
    "--out-dir",
    "--artifacts-dir",
    "--corpus-dir",
    "--out",
)

#: A cluster path: the shared filesystem every project stages beneath.
_CLUSTER_PATH = re.compile(r"^/pub/[^\s]+$")

_TOKEN = re.compile(r"\S+")


def declared_inputs(command: str) -> tuple[str, ...]:
    """Find every cluster file a command declares it will read.

    Args:
        command: The run document's command line.

    Returns:
        Each ``/pub`` path given to an input flag, in command order and
        deduplicated. A flag whose value is not a cluster path -- a relative
        path, a URL, a bare name -- contributes nothing: this checks the
        shared filesystem, which is the only place it can check.
    """
    tokens: list[str] = _TOKEN.findall(command)
    found: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token not in INPUT_FLAGS:
            continue
        value = tokens[index + 1]
        if _CLUSTER_PATH.match(value) and value not in found:
            found.append(value)
    return tuple(found)


def present_on_cluster(host: str, paths: tuple[str, ...]) -> frozenset[str]:
    """Ask the cluster which of these paths it holds.

    One ssh round trip for the whole set rather than one per path: the check
    sits in front of every submission, and a per-path probe would make the
    common case -- everything present -- the slowest.

    Args:
        host: SSH destination, an alias from the user's ssh config.
        paths: Cluster paths to test. An empty tuple asks nothing and
            reaches no network.

    Returns:
        The subset that exists. A path the cluster does not report back is
        absent; nothing is assumed present on a quiet answer.
    """
    if not paths:
        return frozenset()
    listed = " ".join(f"'{path}'" for path in paths)
    # `; true` is load-bearing. Without it the loop inherits the exit
    # status of its last `[ -e ]`, so an ABSENT file makes the whole ssh
    # command exit 1 and run_remote raises REMOTE_COMMAND_FAILED --
    # reporting an infrastructure fault for the ordinary case this exists
    # to detect, and sending the reader to the network instead of to the
    # unstaged file. Absence is an ANSWER here, not an error.
    command = f'for p in {listed}; do [ -e "$p" ] && echo "$p"; done; true'
    reported: list[str] = run_remote(host, command).splitlines()
    return frozenset(line.strip() for line in reported if line.strip())


def missing_inputs(paths: tuple[str, ...], present: frozenset[str]) -> tuple[str, ...]:
    """Name the declared inputs the cluster does not hold.

    Args:
        paths: Cluster paths the command declares it will read.
        present: The subset of them that exist, as the cluster reported it.

    Returns:
        The absent ones, in the order they were declared.
    """
    return tuple(path for path in paths if path not in present)


def require_inputs_present(command: str, present: frozenset[str]) -> None:
    """Refuse a command whose declared inputs are not on the cluster.

    Args:
        command: The run document's command line.
        present: Declared inputs the cluster reported as existing.

    Raises:
        AppError: With ``RUN_INPUT_MISSING``, naming every absent path and
            what to do. Raised before submission, so the refusal costs no
            job id and no GPU seconds -- which is the whole point, since the
            alternative is learning the same thing from a dead job's log.
    """
    absent = missing_inputs(declared_inputs(command), present)
    if not absent:
        return
    listed = "\n  ".join(absent)
    raise AppError(
        Hpc3ErrorCode.RUN_INPUT_MISSING,
        f"the run declares {len(absent)} input file(s) the cluster does not hold:\n"
        f"  {listed}\n"
        "Stage or copy them before submitting. A corpus goes through hpc3-stage, "
        "which certifies it; payloads and specs are copied directly and are not "
        "certified by anything, which is why this check exists.",
    )


__all__ = [
    "INPUT_FLAGS",
    "OUTPUT_FLAGS",
    "declared_inputs",
    "missing_inputs",
    "present_on_cluster",
    "require_inputs_present",
]
