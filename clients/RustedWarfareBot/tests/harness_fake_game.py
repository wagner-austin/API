"""The engine-process fake, split from :mod:`tests.harness_fakes`.

The split is by role: :class:`FakeGame` stands in for a spawned engine
PROCESS, while ``FakeHost`` stands in for the machine around it. The two
change for different reasons -- a new engine exit behaviour touches this
file, a new filesystem or command hook touches the host -- and the host had
passed the 600-line ceiling.
"""

from __future__ import annotations


class FakeGame:
    """An engine process that never started.

    Attributes:
        pid: The process id the launcher will fell.
        exit_status: What :meth:`poll` reports; ``None`` is a live engine.
    """

    def __init__(self, pid: int, exit_status: int | None = None) -> None:
        self.pid = pid
        self.exit_status = exit_status

    def poll(self) -> int | None:
        """Report the engine's state.

        Returns:
            The exit status this fake was told to die with, or ``None`` while
            it plays the living.
        """
        return self.exit_status


__all__ = ["FakeGame"]
