"""Who the bridge is on the board, and where its announcements land.

The board binds a session id to an agent label on first write, permanently
(mig 415, ``assertSessionLabel``). A service has no harness session, so it
mints a DETERMINISTIC one: a UUIDv5 of a fixed name. Every run of this
package on this installation therefore presents the same (label, session)
pair, which is exactly the one-session-one-label rule read as a service
contract -- restarts do not create identities, and a second label under the
same id is refused by the board itself.

The standing task the announcements land in is configuration, not
discovery. Finding it by title search would make every cycle depend on a
render grammar owned by another repository for something that never changes;
the id is created once by the operator (or the setup step in the README) and
exported beside the board credentials.
"""

from __future__ import annotations

import uuid
from typing import Final

from platform_core.config import _optional_env_str
from platform_core.error_codes import HpcWakeErrorCode
from platform_core.errors import AppError

#: The bridge's agent label: kebab-case, service-shaped, stable forever.
BRIDGE_AGENT: Final = "bridge-hpc-wake-0906"

#: The fixed name the bridge's session id is derived from.
_SESSION_NAME: Final = "corvis:hpc-wake:bridge"

#: The bridge's session id -- the same on every run, by construction.
BRIDGE_SESSION_ID: Final = str(uuid.uuid5(uuid.NAMESPACE_URL, _SESSION_NAME))

#: Environment variable naming the standing board task announcements go to.
TASK_ID_VARIABLE: Final = "HPC_WAKE_TASK_ID"

#: The board room the standing task lives in.
ROOM: Final = "main"


def load_task_id() -> str:
    """Read the standing task's id from the environment.

    Returns:
        The task id.

    Raises:
        AppError: ``TASK_ID_MISSING`` when the variable is unset or blank.
            Required rather than defaulted: an announcement posted to a
            guessed task is an announcement nobody is subscribed to, which
            reads exactly like the bridge working.
    """
    task_id = _optional_env_str(TASK_ID_VARIABLE)
    if task_id is None:
        raise AppError(
            code=HpcWakeErrorCode.TASK_ID_MISSING,
            message=(
                f"{TASK_ID_VARIABLE} is unset; it names the standing board task "
                "announcements are posted into. Create the task once and export "
                "its id beside the board credentials."
            ),
        )
    return task_id


__all__ = [
    "BRIDGE_AGENT",
    "BRIDGE_SESSION_ID",
    "ROOM",
    "TASK_ID_VARIABLE",
    "load_task_id",
]
