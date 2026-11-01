"""The slowest steps of a docker build its transcript timed (MCPs board task eb6586e4).

tools/maketools-execution builds each service's images as an install step
before its timed check, and the verdict's ``phases=`` gives each service's
build one total: grandma-api's was 189 s cold on loki on 2026-11-01. The
task's audit asked which step INSIDE that build took the time, as a reading
a program wrote, and a ``--quiet`` build prints none. Built with
``--progress plain``, BuildKit prints each step's header and, when it
finishes, its seconds::

    #20 [runtime-base 5/9] RUN pip install --extra-index-url https://...
    #20 DONE 33.0s

This reads those pairs off the transcript tail and names the slowest steps
for the verdict's ``steps=`` field. A step is keyed by its number, and a
header line reprinted when BuildKit interleaves two steps names it again, so
the latest header for a number wins; a build that reuses numbers after
another build reads correctly for the same reason. Internal steps (loading
the build definition, exporting the image) carry no ``stage i/n`` header and
are not counted, nor is a ``CACHED`` step, which prints no ``DONE`` line.
"""

from __future__ import annotations

import re
from typing import Final

#: A step's header: its number, its stage, its index and count in that
#: stage, its instruction and the instruction's first word. BuildKit pads
#: the stage name when stages of different lengths build together
#: (``[builder  2/10]``).
STEP_HEADER: Final[re.Pattern[str]] = re.compile(r"^#(\d+) \[([\w.-]+) +(\d+)/(\d+)\] (\S+) ?(\S*)")

#: A step's last line: its number and its seconds, to a tenth.
STEP_DONE: Final[re.Pattern[str]] = re.compile(r"^#(\d+) DONE (\d+)\.(\d)s$")

#: What a step's word may keep: a FROM's image loses its tag and digest, and
#: every other character the verdict line or its comma list would split on.
WORD_KEPT: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._/-]")

#: How many of the slowest steps a verdict names.
SLOWEST_STEPS: Final = 3


def _step_name(header: re.Match[str]) -> str:
    """Name one step as the verdict carries it.

    Args:
        header: The step's header line, matched by :data:`STEP_HEADER`.

    Returns:
        ``<stage>-<index>/<count>-<instruction>[-<word>]``, for example
        ``runtime-base-5/9-RUN-pip``.
    """
    word = WORD_KEPT.sub("", f"{header.group(6)}".split(":", 1)[0].split("@", 1)[0])
    instruction = f"{header.group(5)}" if word == "" else f"{header.group(5)}-{word}"
    return f"{header.group(2)}-{header.group(3)}/{header.group(4)}-{instruction}"


def read_build_steps(tail: str) -> str:
    """Read the slowest timed build steps off a transcript tail.

    Args:
        tail: The transcript's last lines, colours stripped.

    Returns:
        ``<step>:<s>s,...`` for the :data:`SLOWEST_STEPS` slowest steps,
        slowest first, seconds rounded and a tie broken by name, else
        ``unread`` (no build in the tail printed its steps).
    """
    names: dict[str, str] = {}
    # Each step as (-tenths of a second, name), so the natural order is
    # slowest first.
    timed: list[tuple[int, str]] = []
    for line in tail.splitlines():
        header = STEP_HEADER.match(line)
        if header is not None:
            names[f"{header.group(1)}"] = _step_name(header)
            continue
        done = STEP_DONE.match(line)
        if done is not None and f"{done.group(1)}" in names:
            tenths = int(done.group(2)) * 10 + int(done.group(3))
            timed.append((-tenths, names[f"{done.group(1)}"]))
    if not timed:
        return "unread"
    ranked = sorted(timed)[:SLOWEST_STEPS]
    return ",".join(f"{name}:{round(-tenths / 10)}s" for tenths, name in ranked)


__all__ = ["SLOWEST_STEPS", "STEP_DONE", "STEP_HEADER", "WORD_KEPT", "read_build_steps"]
