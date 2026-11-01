"""The slowest steps of a docker build its transcript timed (MCPs board task eb6586e4).

tools/maketools-execution builds each service's images as an install step
before its timed check, and the verdict's ``phases=`` gives each service's
build one total: grandma-api's was 189 s cold on loki on 2026-11-01. The
task's audit asked which step INSIDE that build took the time, as a reading
a program wrote, and a ``--quiet`` build prints none. Built with
``--progress plain``, BuildKit prints each step's title and, when it
finishes, its seconds::

    #20 [runtime-base 5/9] RUN pip install --extra-index-url https://...
    #20 DONE 33.0s
    #25 exporting to image
    #25 exporting layers 52.1s done
    #25 DONE 65.7s

This reads those pairs off the transcript tail and names the slowest steps
for the verdict's ``steps=`` field. EVERY titled step counts, the export
included: the first reading skipped steps without a ``[stage i/n]`` title,
and so named a 57 s apt-get the slowest while the 65.7 s export went unread
(fleet job 47f84926, the task's review). A stage step's title is reprinted
whenever BuildKit resumes it and names it again; any other step is named by
its FIRST line, since the lines after it (``exporting layers``, ``naming
to``) report its progress under the same number. Each build numbers its
steps from ``#1`` again, so its first title, which loads the build
definition, starts a fresh set of names.
``CACHED`` steps print no ``DONE`` line and are not counted.
"""

from __future__ import annotations

import re
from typing import Final

#: A step's line that is neither its end nor timed output: its number and
#: the rest of the line.
STEP_TITLE: Final[re.Pattern[str]] = re.compile(r"^#(\d+) (?!DONE |CACHED$|ERROR|\d+\.\d+ )(.+)$")

#: A stage step's title: its stage, its index and count in that stage, its
#: instruction and the instruction's first word. BuildKit pads the stage
#: name when stages of different lengths build together (``[builder  2/10]``).
STAGE_TITLE: Final[re.Pattern[str]] = re.compile(r"^\[([\w.-]+) +(\d+)/(\d+)\] (\S+) ?(\S*)")

#: A step's last line: its number and its seconds, to a tenth.
STEP_DONE: Final[re.Pattern[str]] = re.compile(r"^#(\d+) DONE (\d+)\.(\d)s$")

#: What a name may keep: a FROM's image loses its tag and digest, and every
#: other character the verdict line or its comma list would split on.
WORD_KEPT: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._/-]")

#: The title every build's first step carries, which starts a build's names.
BUILD_START: Final = "[internal] load build definition"

#: How many words of a step without a stage name it.
TITLE_WORDS: Final = 3

#: How many of the slowest steps a verdict names.
SLOWEST_STEPS: Final = 3


def _kept(word: str) -> str:
    """One word as a name keeps it.

    Args:
        word: A word of a step's title.

    Returns:
        The word before any ``:`` tag or ``@`` digest, without the
        characters :data:`WORD_KEPT` drops.
    """
    return WORD_KEPT.sub("", word.split(":", 1)[0].split("@", 1)[0])


def _step_name(title: str) -> str:
    """Name one step as the verdict carries it.

    Args:
        title: The step's title, after its number.

    Returns:
        ``<stage>-<index>/<count>-<instruction>[-<word>]`` for a stage step,
        for example ``runtime-base-5/9-RUN-pip``, else the title's first
        :data:`TITLE_WORDS` words joined by dashes, for example
        ``exporting-to-image``.
    """
    stage = STAGE_TITLE.match(title)
    if stage is None:
        words = [_kept(word) for word in title.split()[:TITLE_WORDS]]
        return "-".join(word for word in words if word != "")
    word = _kept(f"{stage.group(5)}")
    instruction = f"{stage.group(4)}" if word == "" else f"{stage.group(4)}-{word}"
    return f"{stage.group(1)}-{stage.group(2)}/{stage.group(3)}-{instruction}"


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
        title = STEP_TITLE.match(line)
        if title is not None:
            number, text = f"{title.group(1)}", f"{title.group(2)}"
            if number == "1" and text.startswith(BUILD_START):
                names.clear()
            if STAGE_TITLE.match(text) is not None or number not in names:
                names[number] = _step_name(text)
            continue
        done = STEP_DONE.match(line)
        if done is not None and f"{done.group(1)}" in names:
            tenths = int(done.group(2)) * 10 + int(done.group(3))
            timed.append((-tenths, names[f"{done.group(1)}"]))
    if not timed:
        return "unread"
    ranked = sorted(timed)[:SLOWEST_STEPS]
    return ",".join(f"{name}:{round(-tenths / 10)}s" for tenths, name in ranked)


__all__ = [
    "BUILD_START",
    "SLOWEST_STEPS",
    "STAGE_TITLE",
    "STEP_DONE",
    "STEP_TITLE",
    "TITLE_WORDS",
    "WORD_KEPT",
    "read_build_steps",
]
