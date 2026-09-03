"""Field readers shared by the workspace contract and the project contract.

Two modules need them and neither owns them: a project reads ``env_path`` and
``repo`` with the same rules the workspace reads ``host`` and ``ledger`` with.
Leaving them in either module would make the other import it for two
functions and put a cycle between them, since a workspace holds projects.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue, require_int, require_str


def require_nonempty_str(obj: dict[str, JSONValue], key: str) -> str:
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


def require_positive(obj: dict[str, JSONValue], key: str) -> int:
    """Read a required integer field that must be at least one.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not an integer, or below one.
    """
    value = require_int(obj, key)
    if value < 1:
        raise JSONTypeError(f"Field '{key}' must be at least 1, got {value}")
    return value


__all__ = ["require_nonempty_str", "require_positive"]
