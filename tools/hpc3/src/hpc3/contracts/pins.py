"""Declaring exactly what an environment must contain.

A path proves nothing about contents. Two environments built for one project
-- one on the stack a published result used, one on current releases -- differ
by a few characters in a path and by a major version underneath. Both exist,
both pass an existence check, and a run against the wrong one produces a
number that is not comparable to the results it was meant to extend.

A pin is therefore a declaration a project makes about its environment, and
:mod:`hpc3.core.env_probe` holds the environment to it by asking the
environment rather than the path.

Names are stored normalised. ``importlib.metadata`` reports whatever a
distribution called itself, so ``Typing_Extensions`` and ``typing-extensions``
arrive as different strings for one package; normalising at decode means the
comparison later is a dictionary lookup rather than a guess.
"""

from __future__ import annotations

import re

from platform_core.json_utils import JSONTypeError, JSONValue

_SEPARATORS = re.compile(r"[-_.]+")


def normalise_name(name: str) -> str:
    """Reduce a distribution name to its PEP 503 comparison form.

    Args:
        name: Distribution name as written anywhere.

    Returns:
        Lowercased, with runs of ``-``, ``_`` and ``.`` collapsed to a single
        hyphen.
    """
    return _SEPARATORS.sub("-", name).lower()


def require_pinned_packages(obj: dict[str, JSONValue], key: str) -> dict[str, str]:
    """Read and validate a map of distribution name to required version.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        Required versions, keyed by normalised distribution name. An empty
        map is valid and means the project declared no pins.

    Raises:
        JSONTypeError: If the field is missing, is not an object, holds a
            non-string version, or holds an empty name or version. An empty
            version would compare equal to nothing and silently fail every
            run; a missing field is an unasked question rather than an answer
            of "none", which is why the field is required even when empty.
    """
    raw = obj.get(key)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{key}' must be a JSON object, got {type(raw).__name__}")

    pinned: dict[str, str] = {}
    for name, version in raw.items():
        if not isinstance(version, str):
            raise JSONTypeError(
                f"Field '{key}' must map names to version strings; "
                f"{name!r} maps to {type(version).__name__}"
            )
        if name == "" or version == "":
            raise JSONTypeError(
                f"Field '{key}' must not hold an empty name or version; got {name!r}: {version!r}"
            )
        pinned[normalise_name(name)] = version
    return pinned


def encode_pinned_packages(pinned: dict[str, str]) -> dict[str, JSONValue]:
    """Encode pinned versions to a JSON object.

    Args:
        pinned: Required versions, keyed by normalised name.

    Returns:
        JSON-serialisable mapping. Names are emitted normalised, which is what
        was decoded rather than what was written, so a round trip produces an
        equivalent declaration rather than identical bytes.
    """
    return dict(pinned)


__all__ = ["encode_pinned_packages", "normalise_name", "require_pinned_packages"]
