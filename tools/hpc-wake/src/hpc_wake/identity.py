"""Who this bridge is on the board, and where its announcements land.

The RULES behind both moved to :mod:`platform_core.board` on 2026-09-06, when
``tools/fleet-wake`` needed the same ones and copying them would have forked
the subtlest part of either package. What stays here is only what is true of
THIS bridge: its label, the name its session id is derived from, and the
variable naming its standing task.

THE SESSION ID MUST NOT CHANGE, EVER. The board bound
``bridge-hpc-wake-0906`` to ``b6048b2e-2e32-5247-a488-7b4ccc35f2cc`` on this
bridge's first write, permanently (mig 415). Editing :data:`_SESSION_NAME`
would mint a different id, the board would refuse every subsequent post with
``TASK_IDENTITY_MISMATCH``, and there is no way to unbind a label. The value
is pinned in ``tests/test_identity.py`` against the literal string rather than
against a re-derivation, so an edit fails a test instead of a production
cycle.
"""

from __future__ import annotations

from typing import Final

from platform_core.board import BoardIdentity, require_task_id, service_identity
from platform_core.config import _optional_env_str

#: The bridge's agent label: kebab-case, service-shaped, stable forever.
BRIDGE_AGENT: Final = "bridge-hpc-wake-0906"

#: The fixed name the bridge's session id is derived from. NEVER EDIT.
_SESSION_NAME: Final = "corvis:hpc-wake:bridge"

#: What the board records as this bridge's location.
_CWD: Final = "service://hpc-wake"

#: Environment variable naming the standing board task announcements go to.
TASK_ID_VARIABLE: Final = "HPC_WAKE_TASK_ID"

#: The identity every board write presents. Built once, never mutated.
IDENTITY: Final[BoardIdentity] = service_identity(
    agent=BRIDGE_AGENT, service=_SESSION_NAME, cwd=_CWD
)

#: What kind of writer the ledger records this as (MCPs mig 530).
#:
#: The bridge's own name, not ``claude-code``: it has no harness, no
#: transcript and no machine the observer watches, and a slug borrowed from
#: something else would put a row on the ledger that no process answers for.
HARNESS: Final = "hpc-wake"

#: One sentence on the ledger row, for whoever reads it months from now.
PURPOSE: Final = (
    "the Slurm bridge, which announces each newly terminal cluster job on the "
    "standing HPC task and mentions the session that submitted it."
)


def load_task_id() -> str:
    """Read the standing task's id from the environment.

    Returns:
        The task id.

    Raises:
        AppError: ``TASK_ID_MISSING`` from
            :func:`platform_core.board.require_task_id` when the variable is
            unset or blank.
    """
    return require_task_id(_optional_env_str(TASK_ID_VARIABLE), variable=TASK_ID_VARIABLE)


__all__ = [
    "BRIDGE_AGENT",
    "IDENTITY",
    "TASK_ID_VARIABLE",
    "load_task_id",
]
