"""The one place a covenant evaluation's status is written down.

The sibling of :mod:`platform_core.risk_tiers`, and it was left standing when
that one was collapsed -- which is the failure worth naming here, because the
two sets sit in the same event payloads and were written the same way. Fixing
one and not its twin is how the second becomes the only copy nobody is
watching.

WHAT WAS THERE. Two aliases -- ``EvaluationStatus`` in covenant-radar-api's
streaming schemas and ``EvaluationStatusValue`` in its Google AI schemas --
plus a ``VALID_EVALUATION_STATUSES`` tuple beside the second, five inline
``Literal["OK", "BREACH", "WARNING"]`` annotations, and THREE narrowing
implementations: ``_parse_evaluation_status`` twice, byte-identical, and
``_require_evaluation_status`` once, which raised ValueError where the other
two raised JSONTypeError.

WHY THE GUARD WATCHES ``evaluation_status`` AND NOT ``status``, WHICH IS THE
NAME HALF THESE SITES ACTUALLY USE. ``status`` names 28 DISTINCT Literal sets
across this monorepo -- job states, probe outcomes, health, wandb, and a
covenant-domain ``OK/BREACH/NEAR_BREACH`` that is a different three-member set
in the same domain as this one. Watching it would report every one of them as
drift. So the guard covers the five ``evaluation_status`` sites and cannot
cover the four spelled ``status``; those are held only by this module being
the single narrowing they all call.

That is a real limit of keying a rule on a field name, and it is recorded
rather than papered over: a reader who "helpfully" adds ``status`` to the
watched names will turn the guard into noise on 28 unrelated sets.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONObject, JSONTypeError, require_str

EVALUATION_STATUSES: tuple[Literal["OK", "BREACH", "WARNING"], ...] = (
    "OK",
    "BREACH",
    "WARNING",
)
"""Every declared evaluation status.

``OK`` is within covenant terms, ``BREACH`` is outside them, and ``WARNING``
is inside them but within the configured tolerance band. Distinct from
``covenant_domain``'s per-covenant ``OK/BREACH/NEAR_BREACH``: that one grades
a single covenant result, this one grades a period's evaluation as a whole.
The two are not interchangeable and neither is a superset of the other.
"""


def as_evaluation_status(raw: str, field: str) -> Literal["OK", "BREACH", "WARNING"]:
    """Narrow a string to an evaluation status, or refuse it.

    The single narrowing in the monorepo. Three paths need it -- the covenant
    metrics event decoder, the streaming evaluation decoder, and the Google AI
    response reader -- and each had its own copy before this one.

    Written as a chain rather than a membership test because mypy will not
    narrow a ``str`` to a member of a variadic tuple through ``in``, and
    ``typing.get_args`` is unavailable under this package's
    ``disallow_any_expr``.

    Args:
        raw: The string to narrow, from outside this process.
        field: Name of the field it came from, for the refusal.

    Returns:
        The same status, typed.

    Raises:
        JSONTypeError: If the string names no declared status. One of the
            three copies this replaces raised ValueError instead, so a service
            disagreed with itself about the type of its own refusal depending
            on which decoder read the payload.
    """
    if raw == "OK":
        return "OK"
    if raw == "BREACH":
        return "BREACH"
    if raw == "WARNING":
        return "WARNING"
    declared = ", ".join(EVALUATION_STATUSES)
    raise JSONTypeError(f"Field '{field}' must be one of [{declared}], got '{raw}'")


def require_evaluation_status(obj: JSONObject, field: str) -> Literal["OK", "BREACH", "WARNING"]:
    """Read a required evaluation-status field, narrowing it to the Literal.

    Args:
        obj: JSON object to read from.
        field: Name of the field holding the status.

    Returns:
        The status, typed.

    Raises:
        JSONTypeError: If the field is absent, is not a string, or names no
            declared status.
    """
    return as_evaluation_status(require_str(obj, field), field)


__all__ = ["EVALUATION_STATUSES", "as_evaluation_status", "require_evaluation_status"]
