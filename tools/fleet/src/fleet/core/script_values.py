"""Refusing a roster value that a rendered PowerShell script cannot carry.

Lifted out of :mod:`fleet.core.runner_audit` when the rebuild's renderers
(:mod:`fleet.core.runner_base_render`) became its second caller: both embed
roster values inside single-quoted PowerShell strings, and one rule decides
which values may go there.
"""

from __future__ import annotations

#: Characters a value must not carry to be embedded in a rendered script.
#:
#: Values land inside single-quoted PowerShell strings, where a quote ends the
#: string and CR/LF end the statement. Escaping is refused in favour of
#: rejection: every legitimate service name, path and task name in the roster
#: is plain, so a value carrying one of these is a roster error to surface,
#: not a case to accommodate.
UNSCRIPTABLE: tuple[str, ...] = ("'", '"', "\r", "\n", "`", "$")


def scriptable(value: str, *, label: str) -> str:
    """Refuse a value the rendered script could not carry verbatim.

    Args:
        value: The roster value about to be embedded.
        label: What the value is, for the error.

    Returns:
        The value, unchanged.

    Raises:
        ValueError: Deliberately a plain ValueError -- this is a
            roster-content precondition of the renderer, not a JSON-shape
            fault, and the message names the field to fix.
    """
    for forbidden in UNSCRIPTABLE:
        if forbidden in value:
            raise ValueError(
                f"{label} {value!r} contains {forbidden!r}, which cannot be embedded in "
                "a rendered script verbatim; rename the item rather than escaping it"
            )
    return value


__all__ = ["UNSCRIPTABLE", "scriptable"]
