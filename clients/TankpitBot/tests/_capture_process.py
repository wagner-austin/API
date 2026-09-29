"""A real capture child whose waits record their bound instead of spending it.

``stream/capture.py`` ends a helper with ``terminate`` and then
``wait(PROCESS_END_TIMEOUT_SECONDS)``, escalating to ``kill`` and a second
bounded wait. On the Linux fleet that bound is deliberate: a helper stuck in
uninterruptible sleep fails the stop loudly. A test that runs that stop
against a REAL child on a loaded Windows host would instead be betting on the
host's load. There, ``terminate`` is already ``TerminateProcess``, which
returns before the child has run down, and a venv's ``python.exe`` is a
launcher tearing down a second process. Measured on austinpc on 2026-09-29
(board tasks 06fc3195 and 839e14c7), the wait after a kill took over a second
in 48 of 1,601 reaps under load, and 4.83 s at worst.

So this handle leaves every operation to the operating system except one:
``wait`` records the timeout it was asked for, which a test asserts is the
production bound, and then waits for the child's real end with no timeout.
Once ``terminate`` or ``kill`` has been called that end is guaranteed, and
inside a test function pytest-timeout still bounds a real hang. The branch
where the bound expires is proven without a host race by the
``_StuckProcess`` stand-in in ``tests/stream/test_capture.py``.
"""

from __future__ import annotations

from tankpit_bot.stream import _test_hooks as stream_hooks


class PatientCaptureProcess:
    """A protocol-complete handle over a real child, with waits recorded.

    Attributes:
        wait_timeouts: The ``timeout`` of every ``wait`` call, in order.
    """

    def __init__(self, process: stream_hooks.CaptureProcessProtocol) -> None:
        """Wrap a real child.

        Args:
            process: The handle the production spawner returned.
        """
        self._process = process
        self.wait_timeouts: list[float | None] = []

    @property
    def pid(self) -> int:
        """The real child's pid."""
        return self._process.pid

    def poll(self) -> int | None:
        """Delegate to the real child.

        Returns:
            The exit code, or None while it runs.
        """
        return self._process.poll()

    def terminate(self) -> None:
        """Ask the real child to end."""
        self._process.terminate()

    def kill(self) -> None:
        """End the real child."""
        self._process.kill()

    def wait(self, timeout: float | None = None) -> int:
        """Record the bound asked for, then wait for the real end.

        Args:
            timeout: The bound the caller asked for; recorded, not spent.

        Returns:
            The real exit code.
        """
        self.wait_timeouts.append(timeout)
        return self._process.wait()


__all__ = ["PatientCaptureProcess"]
