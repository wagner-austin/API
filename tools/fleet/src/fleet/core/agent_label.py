"""The board's agent-label grammar, held runner-side for every label the fleet embeds.

The queue only length-checks a job's ``submitted_by``, and the runner puts
that label into things a parser reads: the hub's ``build-bases`` make line
(:mod:`fleet.core.rebuild`) and, since MCPs board task 6c4516af A4, every
build script, which exports it as ``BOARD_AGENT_LABEL`` so a hold the suite
takes (MCPs' disposable deploy) is attributed to whoever asked for the run.
A label outside the board's kebab alphabet cannot be embedded in a shell or
PowerShell script verbatim, so it is refused before any script is rendered.
"""

from __future__ import annotations

import re
from typing import Final

#: The board's agent-label grammar.
AGENT_LABEL_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")

#: The variable a build exports the submitter's label under, the name MCPs'
#: fleet hold reads its actor from.
AGENT_LABEL_VARIABLE: Final = "BOARD_AGENT_LABEL"


def require_agent_label(label: str) -> str:
    """Refuse a label outside the board's grammar.

    Args:
        label: The job's submitting agent label.

    Returns:
        The label, unchanged.

    Raises:
        ValueError: When it is outside the grammar, naming it.
    """
    if AGENT_LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError(
            f"submitter label {label!r} is outside the board's kebab-case grammar, so no "
            "build script can export it"
        )
    return label


__all__ = ["AGENT_LABEL_PATTERN", "AGENT_LABEL_VARIABLE", "require_agent_label"]
