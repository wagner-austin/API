"""Narrowing an untrusted word to a member of a StrEnum vocabulary.

A closed set of words in this monorepo is a :class:`enum.StrEnum`, one
definition whose members are the same strings on the wire, rather than a
module-level ``X = Literal[...]``, which is a type alias the operator's "no
type alias" covers (MCPs board task 1374feba). Every decoder of such a word
narrows it here, so each vocabulary has one refusal message and no decoder
keeps its own ``if raw == "a": return "a"`` chain.

Three entry points, one lookup:

* :func:`find_member` answers "which member carries this word, if any" for a
  boundary where a word outside the vocabulary is a legitimate other case.
* :func:`as_member` narrows a word already read as ``str`` and refuses one
  outside the vocabulary with :class:`JSONTypeError`.
* :func:`require_member` reads ``obj[key]`` as a string first, the same shape
  as :func:`platform_core.json_utils.require_str`.

The lookup goes through the members' values because ``word in members``
raises for a non-member on Python 3.11.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

from platform_core.json_utils import JSONObject, JSONTypeError, require_str

MemberT = TypeVar("MemberT", bound=StrEnum)


def find_member(text: str, members: type[MemberT]) -> MemberT | None:
    """Return the member of ``members`` whose value is ``text``, if any.

    Args:
        text: The candidate word.
        members: The vocabulary.

    Returns:
        The member carrying ``text``, or ``None`` when no member does.
    """
    for member in members:
        if member.value == text:
            return member
    return None


def as_member(raw: str, field: str, members: type[MemberT]) -> MemberT:
    """Narrow ``raw`` to a member of ``members``, or refuse it.

    Args:
        raw: The word to narrow.
        field: The field name the refusal message names.
        members: The vocabulary.

    Returns:
        The member whose value is ``raw``.

    Raises:
        JSONTypeError: When no member carries ``raw``; the message names the
            field, the word and every admitted word in declaration order.
    """
    member = find_member(raw, members)
    if member is None:
        admitted = ", ".join(f"'{declared.value}'" for declared in members)
        raise JSONTypeError(f"Invalid {field} '{raw}': must be one of {admitted}")
    return member


def require_member(obj: JSONObject, key: str, members: type[MemberT]) -> MemberT:
    """Read ``obj[key]`` as a string and narrow it to a member of ``members``.

    Args:
        obj: The decoded JSON object.
        key: The field to read.
        members: The vocabulary.

    Returns:
        The member whose value is ``obj[key]``.

    Raises:
        JSONTypeError: When ``key`` is absent or not a string (from
            :func:`platform_core.json_utils.require_str`), or when no member
            carries its value.
    """
    return as_member(require_str(obj, key), key, members)


__all__ = ["MemberT", "as_member", "find_member", "require_member"]
