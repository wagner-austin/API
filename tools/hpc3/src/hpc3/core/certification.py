"""Refuse a run whose declared inputs no record vouches for.

WHY THIS EXISTS, AND WHY EXISTENCE WAS NOT ENOUGH. :mod:`hpc3.core.inputs`
refuses a run naming a ``/pub`` file the cluster does not hold. That closed the
loud half: job 55806418 died in fourteen seconds on an absent payload. The
quiet half is a payload that IS there and is the wrong one -- stale from a
previous run, edited in place, or drifted from the repo copy. It stages
nothing, verifies nothing, trains to completion, and is comparable to nothing.

``hpc3-stage`` already solves this for corpora, and solves it generically:
:class:`~hpc3.contracts.stage.StageManifest` names files and a destination,
:func:`~hpc3.core.stage.stage_manifest` verifies each file's digest locally,
sends it, re-digests it ON THE CLUSTER, and writes a ``*-digests.txt`` record
beside the data. Nothing in that path is corpus-specific. A payload could
always have been staged this way; nothing ever checked that it had been.

So this module is the consumer side, and it deliberately mirrors the rule
Model-Trainer already applies to corpora in
``model_trainer.cluster.preflight.check_corpus_certified``: the file's ACTUAL
digest must appear in a certification record sitting beside it. That consumer
additionally requires the bytes to hash to the FILENAME, because a corpus is
addressed by digest (``corpus_dir/<file_id>``). A payload is not --
``code-style-qlora-v2.json`` is a name -- so that leg does not transfer and is
not imitated here. What transfers is the part that matters: a later reader can
prove which bytes a run used.

PER-PROJECT, NOT GLOBAL, AND THAT IS A MEASUREMENT RATHER THAN A COMPROMISE.
Surveyed 2026-09-09 across every committed run document: ten unique declared
inputs, spanning code-style, floor and mi, and not one of their directories
held a ``*-digests.txt`` at all. Certification here is absent, not merely
uneven. An unconditional check would have refused the next submission of every
registered project, which ``ProjectConfig.certified_inputs`` exists to avoid
without hiding the fact -- a project answering ``false`` is recording that its
inputs arrive by a route that records nothing.
"""

from __future__ import annotations

import re

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core.remote import run_remote
from hpc3.core.stage import CERTIFICATION_SUFFIX

#: One ``<digest>  <name>`` row, as `sha256sum` prints it and as
#: :func:`hpc3.core.stage.certification_text` writes it.
#:
#: The name is a SINGLE token deliberately. A certification record ends with a
#: free-form ``provenance ...`` line, and matching ``\\S.*`` would let a
#: sentence that happens to open with 64 hex characters register as a file row.
#: A filename has no spaces; prose does.
_DIGEST_LINE = re.compile(r"^([0-9a-f]{64})\s+(\S+)$")


def certification_probe(paths: tuple[str, ...]) -> str:
    """Build the one shell command that answers the whole question.

    For each path it emits the file's digest, then the contents of every
    ``*-digests.txt`` beside it. Both halves in one round trip, because this
    runs in front of every submission and the alternative is two.

    Args:
        paths: Cluster paths to ask about.

    Returns:
        The command text.
    """
    parts: list[str] = []
    for path in paths:
        quoted = f"'{path}'"
        # `; true` on both, for the reason inputs.py documents: a missing file
        # or an empty glob must be an ANSWER, not an ssh failure that sends
        # the reader to the network instead of to the unstaged file.
        parts.append(f'echo "FILE {path}"; sha256sum {quoted} 2>/dev/null; true')
        parts.append(
            f'echo "RECORDS {path}"; '
            f'cat "$(dirname {quoted})"/*{CERTIFICATION_SUFFIX} 2>/dev/null; true'
        )
    return "; ".join(parts)


def parse_probe(
    output: str, paths: tuple[str, ...]
) -> dict[str, tuple[str, frozenset[tuple[str, str]]]]:
    """Read the probe's answer back into one entry per path.

    THE RECORD'S NAME COLUMN IS KEPT, and that is the whole point. An earlier
    version collected only the digests and asked whether the file's digest
    appeared ANYWHERE in the directory's records. That proves "these bytes are
    known here", which is a weaker claim than it reads as: two files staged
    into one directory both appear in one record, so a pair holding EACH
    OTHER'S bytes satisfies it. For the generation specs that is precisely the
    confound the arms must not have -- base and candidate differing in exactly
    one field -- and the check would have waved a swap through. Caught in audit
    2026-09-09; the name was already captured by the pattern and simply never
    read.

    Args:
        output: The probe's stdout.
        paths: The paths asked about, used to key the result.

    Returns:
        For each path, its digest on the cluster -- empty string when the
        file is absent or unreadable -- and every ``(name, digest)`` pair its
        neighbouring certification records assert.
    """
    found: dict[str, tuple[str, frozenset[tuple[str, str]]]] = {
        path: ("", frozenset()) for path in paths
    }
    current = ""
    mode = ""
    recorded: set[tuple[str, str]] = set()
    digest = ""

    def _flush() -> None:
        if current:
            found[current] = (digest, frozenset(recorded))

    for line in output.splitlines():
        if line.startswith("FILE "):
            _flush()
            current = line[len("FILE ") :]
            mode, digest, recorded = "file", "", set()
            continue
        if line.startswith("RECORDS "):
            mode = "records"
            continue
        if not current:
            continue
        matched = _DIGEST_LINE.match(line.strip())
        if not matched:
            continue
        if mode == "file":
            digest = matched.group(1)
        else:
            recorded.add((matched.group(2), matched.group(1)))
    _flush()
    return found


def uncertified(
    paths: tuple[str, ...], probed: dict[str, tuple[str, frozenset[tuple[str, str]]]]
) -> tuple[str, ...]:
    """Name the declared inputs no record beside them vouches for.

    A file is admitted only when a record asserts ITS OWN NAME against the
    digest it actually has. Matching the digest alone would admit any file
    whose bytes are recorded somewhere in the directory, which two files
    staged together can satisfy by holding each other's contents.

    A path whose file is absent is NOT reported here. Absence is
    :func:`hpc3.core.inputs.require_inputs_present`'s question, and reporting
    it twice would tell a reader to stage a file they already know is missing
    while burying the ones that are present and unvouched.

    Args:
        paths: Cluster paths the command declares it will read.
        probed: What the cluster reported for each.

    Returns:
        The unvouched paths, in declaration order.
    """
    return tuple(
        path
        for path in paths
        if probed[path][0] and (path.rsplit("/", 1)[-1], probed[path][0]) not in probed[path][1]
    )


def require_inputs_certified(host: str, paths: tuple[str, ...]) -> None:
    """Refuse a run whose declared inputs no certification admits.

    Args:
        host: SSH destination.
        paths: Cluster paths the command declares it will read. An empty
            tuple asks nothing and reaches no network.

    Raises:
        AppError: With ``RUN_INPUT_UNCERTIFIED``, naming every unvouched path
            and how to fix it. Raised before submission, so the refusal costs
            no job id and no GPU seconds.
    """
    if not paths:
        return
    probed = parse_probe(run_remote(host, certification_probe(paths)), paths)
    absent = uncertified(paths, probed)
    if not absent:
        return
    listed = "\n  ".join(f"{path}\n    digest {probed[path][0]}" for path in absent)
    raise AppError(
        Hpc3ErrorCode.RUN_INPUT_UNCERTIFIED,
        f"the run declares {len(absent)} input file(s) that no certification record "
        f"beside them names:\n  {listed}\n"
        f"Each directory was searched for *{CERTIFICATION_SUFFIX}. Stage the file with "
        "hpc3-stage, which digests it locally, re-digests it on the cluster and writes "
        "that record -- a file copied up by hand has none, which is the difference this "
        "check exists to see. This project declares certified_inputs; a project whose "
        "inputs arrive by an unrecorded route should declare it false rather than be "
        "waved through.",
    )


__all__ = [
    "certification_probe",
    "parse_probe",
    "require_inputs_certified",
    "uncertified",
]
