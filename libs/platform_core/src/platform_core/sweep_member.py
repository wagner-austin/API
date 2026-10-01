"""One member of a sweep, and the artifact every run must declare.

A sweep is one template run several ways: the members differ only in the
payload command and the path that command writes its result to. The template
-- partition, GPUs, wall clock, the per-user QOS ceilings it is checked
against -- is the submitter's business and stays in ``hpc3``. The member is
not: a payload that BUILDS sweep documents (``rw_bot``'s campaign harness)
writes members, and a payload's image runs on a compute node that must never
submit anything. Keeping the member here, in the library both sides already
ship, lets the payload speak the document without carrying the dispatcher.

:func:`require_artifact_in_command` moved with it because the member is one of
its two callers -- the other is ``hpc3``'s job spec, which imports it from
here -- and a member cannot be decoded without it.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from platform_core.json_utils import JSONTypeError, JSONValue, require_str


class SweepMember(TypedDict):
    """One variation on the sweep's template.

    Attributes:
        suffix: Appended to the template's name to make this job's name.
            Distinct across the sweep, because the name determines the log
            filenames and two jobs sharing one would interleave into the
            same file.
        command: Payload for this member, replacing the template's.
        artifact: Where this member was told to write its result, or None.
            Per member rather than per sweep, for the same reason the command
            is: six arms writing to one path are five results nobody can
            read. Checked against this member's OWN command, so a suffix
            edited in the command and not the path fails here.
    """

    suffix: str
    command: str
    artifact: str | None


def require_artifact_in_command(obj: dict[str, JSONValue], command: str) -> str | None:
    """Read the declared artifact path, and refuse one the command never writes.

    The ledger is an index: job -> image -> artifact. A declaration nobody
    checks turns that into a confident wrong answer -- a reader follows the
    path, finds nothing, and cannot tell whether the run failed, wrote
    somewhere else, or was never going to write at all.

    The check is deliberately a substring test rather than a parse. This
    contract does not know any payload's flags, and it should not: the claim
    being verified is only that the path the ledger will publish is a path
    this command mentions. That catches the failure that actually happens --
    an output path edited in one place and not the other -- without pretending
    to understand what the command does with it.

    The key is REQUIRED; only its value may be null. Absent and null used to
    read alike, and the result was an index with its answer column empty: of
    130 recorded runs, 8 carried an image digest -- which fills itself in from
    the spec -- and ONE named where its result went. Every probe run in
    ``runs/`` writes ``--out /pub/wagnera3/probe/<name>.json`` and none of them
    said so, so `hpc3-trace` could reach the job and the image and then stop.

    Requiring the key does not force a fiction on a run that produces nothing.
    It forces the author to SAY which of the two they mean, once, in the spec
    -- and writing ``"artifact": null`` next to a command with an ``--out``
    flag is a claim somebody has to make on purpose rather than a field they
    never noticed.

    Args:
        obj: The run document being decoded.
        command: The command this run will execute, already validated.

    Returns:
        The declared path, or None when the run states it produces nothing
        durable. A directory is a legitimate answer where a run writes several
        files into one place: the reader follows it and finds them.

    Raises:
        JSONTypeError: If ``artifact`` is absent, is present but is not a
            non-empty string or null, or names a path its own command does
            not contain.
    """
    if "artifact" not in obj:
        raise JSONTypeError(
            "Field 'artifact' is required. Name the path this run writes its result to -- "
            "the ledger publishes it, so `hpc3-trace` can answer 'which file holds this "
            "run's answer'. Write null to state that the run produces nothing durable."
        )
    artifact = obj["artifact"]
    if artifact is None:
        return None
    if not isinstance(artifact, str):
        raise JSONTypeError(
            f"Field 'artifact' must be a string or null, got {type(artifact).__name__}"
        )
    if artifact == "":
        raise JSONTypeError("Field 'artifact' must name a path or be null, not an empty string")
    if artifact not in command:
        raise JSONTypeError(
            f"Field 'artifact' names {artifact!r}, which does not appear in this run's "
            "command. The ledger publishes this path as where the result will be, so a "
            "declaration the command does not honour would index a file nobody writes."
        )
    return artifact


def _require_nonempty_str(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required string field that must not be empty.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def encode_sweep_member(member: SweepMember) -> dict[str, JSONValue]:
    """Encode one sweep member to a JSON object.

    Args:
        member: Member to encode.

    Returns:
        JSON-serialisable mapping carrying every field, ``artifact`` included
        when it is None: the key is required on decode, so an encoder that
        dropped a null would write a document its own decoder refuses.
    """
    return {
        "suffix": member["suffix"],
        "command": member["command"],
        "artifact": member["artifact"],
    }


def decode_sweep_member(value: JSONValue) -> SweepMember:
    """Decode and validate a JSON value into one sweep member.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated member.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing,
            mistyped, empty, or -- for the suffix -- carries a character that
            would leave the job name unusable as a filename.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"sweep member must be a JSON object, got {type(value).__name__}")
    suffix = _require_nonempty_str(value, "suffix")
    if "/" in suffix or "\\" in suffix:
        raise JSONTypeError(f"Field 'suffix' must not contain a path separator, got {suffix!r}")
    command = _require_nonempty_str(value, "command")
    return SweepMember(
        suffix=suffix,
        command=command,
        artifact=require_artifact_in_command(value, command),
    )


__all__ = [
    "SweepMember",
    "decode_sweep_member",
    "encode_sweep_member",
    "require_artifact_in_command",
]
