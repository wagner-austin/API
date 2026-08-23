"""What a run IS, as opposed to which row in the queue it was.

A ledger entry records a job: its id, its name, where its logs went. That is
enough to find the job and not enough to say what it produced. With six arms in
flight the only thing linking a job to the corpus it trained on is a name
string, and a name is a label somebody typed. ``arm-b-43`` mistyped as
``arm-b-42`` gives two jobs claiming one identity, no error anywhere, and a
reconciliation that has to go back through log files.

That matters most where it is least visible. Tracing an outcome file back to
the run and the corpus that produced it is the whole job of a provenance
record, and the tool that submits the run is both the natural place to capture
that link and, until now, the one place it was not captured.

So a run declares what identifies it, the declaration is required, and it goes
into the ledger beside the job id. ``hpc3-trace`` then answers "which job
trained ``07ab4976``" from the ledger directly.

The pairs are free-form for the same reason provenance is: what identifies a
run differs per project. A transformer arm is a corpus digest plus a seed plus
a base model; a gradient-boosting sweep is a dataset and a set of
hyperparameters; an annotation job is an input spectrum and a database
version. A fixed schema would force half of those to write fields that do not
apply, and a fabricated field is worse than an absent one.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue

MAX_COMMENT_LENGTH = 120
"""How much of the experiment fits in Slurm's ``--comment`` beside the rest.

The comment already carries project, hardware and environment. What is left is
enough for a digest and a seed, which is what a person reading ``scontrol show
job`` is looking for; the ledger holds the whole record.
"""


def require_experiment(obj: dict[str, JSONValue], key: str) -> dict[str, str]:
    """Read and validate a run's identity record.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The identity pairs, exactly as written.

    Raises:
        JSONTypeError: If the field is missing, is not an object, is empty, or
            holds a non-string, empty key or empty value. Required because a
            run that cannot say what it is produces a result nobody can trace
            back to it, and that is discovered months later.
    """
    raw = obj.get(key)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{key}' must be a JSON object, got {type(raw).__name__}")
    if raw == {}:
        raise JSONTypeError(
            f"Field '{key}' must record at least one thing identifying this run -- "
            "the corpus digest, the seed, the base model. A job id says which "
            "queue row it was, not which result it produced."
        )

    experiment: dict[str, str] = {}
    for name, detail in raw.items():
        if not isinstance(detail, str):
            raise JSONTypeError(
                f"Field '{key}' must map names to strings; {name!r} maps to {type(detail).__name__}"
            )
        if name == "" or detail == "":
            raise JSONTypeError(
                f"Field '{key}' must not hold an empty name or value; got {name!r}: {detail!r}"
            )
        experiment[name] = detail
    return experiment


def encode_experiment(experiment: dict[str, str]) -> dict[str, JSONValue]:
    """Encode a run's identity record to a JSON object.

    Args:
        experiment: The pairs to encode.

    Returns:
        JSON-serialisable mapping carrying every pair unchanged.
    """
    return dict(experiment)


def format_experiment(experiment: dict[str, str]) -> str:
    """Render an identity record for a report or a log line.

    Args:
        experiment: The pairs to render.

    Returns:
        ``key=value`` pairs joined by spaces, in sorted key order so two runs
        of the same experiment produce the same line.
    """
    return " ".join(f"{name}={experiment[name]}" for name in sorted(experiment))


def comment_fragment(experiment: dict[str, str]) -> str:
    """Render an identity record compactly enough for ``sbatch --comment``.

    Args:
        experiment: The pairs to render.

    Returns:
        ``key=value`` pairs joined by commas, with no spaces because Slurm
        takes the comment as a single token, truncated at
        :data:`MAX_COMMENT_LENGTH` with a trailing ``…`` when it does not fit.
        Truncation is safe: this is a convenience for someone reading
        ``scontrol show job``, and the ledger holds the full record.
    """
    rendered = ",".join(f"{name}={experiment[name]}" for name in sorted(experiment))
    if len(rendered) <= MAX_COMMENT_LENGTH:
        return rendered
    return rendered[: MAX_COMMENT_LENGTH - 1] + "…"


def matches(experiment: dict[str, str], needle: str) -> bool:
    """Report whether a record names a value or key the caller is looking for.

    Args:
        experiment: The record to search.
        needle: What to look for.

    Returns:
        True when any key or value equals the needle exactly. Exact rather
        than substring: a digest prefix that matches two corpora would answer
        "which job trained this" with two jobs and no warning.
    """
    return needle in experiment or needle in experiment.values()


__all__ = [
    "MAX_COMMENT_LENGTH",
    "comment_fragment",
    "encode_experiment",
    "format_experiment",
    "matches",
    "require_experiment",
]
