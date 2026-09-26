"""Who this bridge is on the board, and where its posts land.

The RULES behind both live in :mod:`platform_core.board`, shared with the
sibling bridges. What is here is only what is true of THIS bridge.

THE SESSION ID MUST NOT CHANGE, EVER. The board binds a session id to an
agent label on first write and never releases it (mig 415,
``assertSessionLabel``). Editing :data:`_SESSION_NAME` mints a different
id, after which every post is refused with ``TASK_IDENTITY_MISMATCH`` and
there is no way to unbind the label. The value is pinned as a literal in
``tests/test_identity.py`` -- not re-derived there -- so an edit fails a
test rather than a production cycle.
"""

from __future__ import annotations

from typing import Final

from platform_core.board import BoardIdentity, require_task_id, service_identity
from platform_core.config import _optional_env_str

#: The bridge's agent label: kebab-case, service-shaped, stable forever.
BRIDGE_AGENT: Final = "bridge-fleet-health-0926"

#: The fixed name the bridge's session id is derived from. NEVER EDIT.
_SESSION_NAME: Final = "corvis:fleet-health-wake:bridge"

#: What the board records as this bridge's location. The ``service://``
#: prefix is what the session audits read to know this is not a session
#: with a queue to read (MCPs ``session_audit.reading.SERVICE_CWD_PREFIX``).
_CWD: Final = "service://fleet-health-wake"

#: The reader name the journal's cursor file carries
#: (:func:`platform_core.journal_cursor.cursor_path`). NEVER EDIT: a new
#: name reads as a bridge that never ran and re-posts the whole journal.
CURSOR_READER: Final = "fleet-health-wake"

#: Environment variable naming the standing board task posts go to.
TASK_ID_VARIABLE: Final = "FLEET_HEALTH_WAKE_TASK_ID"

#: The identity every board write presents. Built once, never mutated.
IDENTITY: Final[BoardIdentity] = service_identity(
    agent=BRIDGE_AGENT, service=_SESSION_NAME, cwd=_CWD
)

#: What kind of writer the ledger records this as (MCPs mig 530): the
#: bridge's own name, since it has no harness and no transcript.
HARNESS: Final = "fleet-health-wake"

#: One sentence on the ledger row, for whoever reads it months from now.
PURPOSE: Final = (
    "the fleet health bridge, which posts each line the fleet audit appends to "
    "its health journal: nodes and rows that started failing or recovered."
)


def load_task_id() -> str:
    """Read the standing task's id from the environment.

    Returns:
        The task id.

    Raises:
        AppError: ``TASK_ID_MISSING`` from
            :func:`platform_core.board.require_task_id` when the variable
            is unset or blank.
    """
    return require_task_id(_optional_env_str(TASK_ID_VARIABLE), variable=TASK_ID_VARIABLE)


__all__ = [
    "BRIDGE_AGENT",
    "CURSOR_READER",
    "HARNESS",
    "IDENTITY",
    "PURPOSE",
    "TASK_ID_VARIABLE",
    "load_task_id",
]
