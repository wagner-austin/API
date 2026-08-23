"""Where a staged file came from, and the record that says so.

Staging proves the bytes on the cluster are the bytes named in the manifest.
It cannot prove the manifest names the right bytes, and that is the gap this
module and :mod:`hpc3.core.expected` close between them.

The failure is specific and it passes every transport check: emit a corpus
from the wrong source state, and you get a manifest whose digests match the
files you just made, staged and verified end to end, and comparable to nothing
already published. The extraction ablation's arms were emitted over 733 wiki
pages; the wiki now holds 776. Re-emitting from the current tree produces a
self-consistent manifest for a corpus that is not the one the published arms
used, and nothing anywhere reports a problem.

Two halves answer it, and they are deliberately different in kind:

* **Provenance** is the *record* -- free-form key/value pairs saying what
  produced these bytes. It is not machine-checkable, because what identifies a
  source differs per project: a commit for a corpus emitted from a repository,
  an instrument run for a spectrum, a seed and a generator version for
  synthetic data. Forcing a fixed schema on that would mean projects writing
  ``"none"`` into fields that do not apply, and a fabricated field is worse
  than an absent one.
* **The expected-digest check** is the *enforcement*, in
  :mod:`hpc3.core.expected`. It holds the manifest against an external record
  of digests that were published, which is a real check precisely because the
  record is not written by the same act that stages.

Provenance is required and must not be empty. A staging operation that cannot
say where its bytes came from is the one worth refusing.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue


def require_provenance(obj: dict[str, JSONValue], key: str) -> dict[str, str]:
    """Read and validate a manifest's provenance record.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The provenance pairs, exactly as written. Keys are not normalised:
        this is a record for a human reading it later, and ``wiki_commit``
        should come back out as ``wiki_commit``.

    Raises:
        JSONTypeError: If the field is missing, is not an object, is empty, or
            holds a non-string, empty key or empty value. An empty record
            would satisfy the requirement while saying nothing, which is the
            outcome requiring it at all was meant to prevent.
    """
    raw = obj.get(key)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{key}' must be a JSON object, got {type(raw).__name__}")
    if raw == {}:
        raise JSONTypeError(
            f"Field '{key}' must record at least one fact about where these bytes "
            "came from; a manifest that cannot say is the one worth refusing."
        )

    provenance: dict[str, str] = {}
    for name, detail in raw.items():
        if not isinstance(detail, str):
            raise JSONTypeError(
                f"Field '{key}' must map names to strings; {name!r} maps to {type(detail).__name__}"
            )
        if name == "" or detail == "":
            raise JSONTypeError(
                f"Field '{key}' must not hold an empty name or value; got {name!r}: {detail!r}"
            )
        provenance[name] = detail
    return provenance


def encode_provenance(provenance: dict[str, str]) -> dict[str, JSONValue]:
    """Encode a provenance record to a JSON object.

    Args:
        provenance: The pairs to encode.

    Returns:
        JSON-serialisable mapping carrying every pair unchanged.
    """
    return dict(provenance)


def format_provenance(provenance: dict[str, str]) -> str:
    """Render a provenance record for a one-line report.

    Args:
        provenance: The pairs to render.

    Returns:
        ``key=value`` pairs joined by spaces, in sorted key order so two runs
        of the same staging produce the same line.
    """
    return " ".join(f"{name}={provenance[name]}" for name in sorted(provenance))


__all__ = ["encode_provenance", "format_provenance", "require_provenance"]
