"""The deadline bounds the call even when the call carries a payload.

Board task 1e57ebe5. Every other deadline case in this package drives a
child that reads nothing, so the pipe never fills and the payload never
matters; those cases pass whether or not a bulk payload is bounded, which
is why a staging send sat forty minutes against a 120-second bound on
2026-09-24 with the suite green throughout. The fixtures and the defect
shared an assumption.

So these cases are written the opposite way round. The child STOPS
READING while a payload larger than any pipe buffer is handed over. Under
``subprocess.run(input=...)`` the parent's own write blocks with no clock
in the path; under the shape
:func:`fleet.core._test_hooks._default_run` uses, the payload is a file
the child reads itself and the deadline governs what is left.

THE ELAPSED BOUND IS THE ASSERTION THAT DISCRIMINATES, AND ``timed_out``
IS NOT. Measured 2026-09-24 against the old shape at exactly the size and
deadline below: the call returned after 60.2 SECONDS against a
one-second deadline, because a blocked write ends when the CHILD does,
and the child here sleeps sixty. It then raises ``TimeoutExpired`` like
any other late command, so ``timed_out`` is True under the defect too. A
case asserting only the flag passes on broken code. What the defect
cannot survive is being asked to finish inside a bound.

The second case is the other half and is not decoration: a call that
bounded itself by delivering less than it was given would satisfy the
first case and corrupt every staging send.
"""

from __future__ import annotations

import sys
import time

from fleet.core import _test_hooks

#: Larger than any operating-system pipe buffer, so a child that does not
#: read is guaranteed to stop the writer rather than merely slow it. Pipe
#: buffers are measured in kilobytes; this is eight megabytes, and it stays
#: small enough that writing it to a temporary file costs a test nothing.
PIPE_FILLING_BYTES = 8_000_000

#: The deadline under test. One second, so a bound that applies is obvious
#: against the ceiling below.
DEADLINE_SECONDS = 1

#: What the whole call must finish inside. Twenty times the deadline, and
#: a third of the sixty seconds the unbounded shape was measured taking,
#: so the two cannot be confused on a slow machine either way. Generous on
#: purpose: this asserts that a clock governed the call, not how fast the
#: machine is.
CEILING_SECONDS = 20

#: A child that never reads its standard input and outlives the deadline.
DEAF_CHILD = (sys.executable, "-c", "import time; time.sleep(60)")

#: A child that reads its standard input to EOF and reports how much it got.
COUNTING_CHILD = (
    sys.executable,
    "-c",
    "import sys; print(len(sys.stdin.buffer.read()))",
)


class TestADeadlineBoundsAPayloadCarryingCall:
    def test_a_child_that_stops_reading_does_not_outlive_the_deadline(self) -> None:
        """The regression. The payload cannot fit the pipe and nothing drains it.

        That is the shape a fleet staging send takes when a node stops
        consuming mid-transfer. Measured against the old implementation at
        this size and deadline: 60.2 s, ended by the child rather than by
        the clock, with ``timed_out`` still True. Only the elapsed
        assertion below separates the two.
        """
        started = time.monotonic()
        result = _test_hooks._default_run(
            list(DEAF_CHILD),
            timeout_seconds=DEADLINE_SECONDS,
            stdin_bytes=b"x" * PIPE_FILLING_BYTES,
        )
        elapsed = time.monotonic() - started

        assert result["timed_out"] is True
        assert result["returncode"] == _test_hooks.TIMED_OUT_RETURNCODE
        assert result["stderr"].endswith(f"timed out after {DEADLINE_SECONDS} s")
        # THE ONE ASSERTION THE DEFECT CANNOT PASS. Everything above it is
        # true of the broken implementation as well; a clock ending the
        # call rather than the child outliving it is the whole difference.
        assert elapsed < CEILING_SECONDS, (
            f"the call took {elapsed:.1f}s against a {DEADLINE_SECONDS}s deadline, "
            "so the deadline did not govern it"
        )

    def test_a_reading_child_still_receives_every_byte(self) -> None:
        """Bounding the call must not have been bought by delivering less."""
        result = _test_hooks._default_run(
            list(COUNTING_CHILD),
            timeout_seconds=60,
            stdin_bytes=b"x" * PIPE_FILLING_BYTES,
        )

        assert result["timed_out"] is False
        assert result["returncode"] == 0
        assert result["stdout"].strip() == str(PIPE_FILLING_BYTES)
