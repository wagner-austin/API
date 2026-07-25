"""Headless Rusted Warfare client.

Two cooperating processes: a Java agent inside the game's JVM that dispatches
orders and serialises state, and this Python planner outside it. See
``wiki/pages/runtime-split-java-agent-python-brain.md`` for the split and
``wiki/index.md`` for everything else.

The shared failure base lives here rather than in a dedicated error module:
the monorepo forbids per-project ``errors.py`` files so that projects cannot
grow a parallel error framework alongside ``platform_core``. Concrete
exceptions are defined by the module that raises them, next to the code whose
contract they describe.
"""

from __future__ import annotations


class RwBotError(Exception):
    """Base for every failure this package raises.

    Carries a stable machine-readable code alongside a human-readable message
    so callers can branch on the code and operators can read the message. There
    is deliberately no catch-all code: a new failure mode gets a new one so it
    can be grepped, asserted on, and counted.

    Code format is ``RW-<AREA>-<NNN>``.

    Args:
        code: Stable identifier, e.g. ``"RW-BOOTLOG-001"``.
        message: Human-readable description of what went wrong.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


__all__ = ["RwBotError"]
