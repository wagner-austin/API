"""How much of a failed remote command's stderr a failure carries (MCPs board task ee3d6cdc)."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, remote
from fleet.core.stderr_tail import NO_STDERR, TAIL_CHARS, TAIL_LINES, stderr_tail
from tests.conftest import FakeRun, failed

#: One of the per-file warnings fleet job af6c4d99's init-repo step wrote 7778 of.
WARNING = (
    "warning: in the working copy of 'Makefile', LF will be replaced by CRLF "
    "the next time Git touches it"
)

#: How a tail that left something out begins.
LEFT_OUT = "earlier line(s) and {chars} character(s) of stderr left out]"

#: ssh's own last word in af6c4d99.
BROKEN_PIPE = "client_loop: send disconnect: Broken pipe"


def test_a_short_stderr_is_carried_whole() -> None:
    assert stderr_tail("  first\nsecond\n") == "first\nsecond"


def test_no_stderr_is_said_so() -> None:
    assert stderr_tail(" \n ") == NO_STDERR


def test_a_long_stderr_keeps_its_last_lines_and_says_how_many_it_left_out() -> None:
    lines = [f"line {index}" for index in range(TAIL_LINES + 5)]

    tail = stderr_tail("\n".join(lines))

    assert tail == f"[5 {LEFT_OUT.format(chars=0)} " + "\n".join(lines[5:])


def test_one_very_long_last_line_is_kept_from_its_end() -> None:
    tail = stderr_tail("x" * (TAIL_CHARS + 7) + "END")

    assert tail == f"[0 {LEFT_OUT.format(chars=10)} {'x' * (TAIL_CHARS - 3)}END"


def test_af6c4d99_is_unreachable_and_names_the_broken_pipe_not_the_flood() -> None:
    """The real failure: init-repo's git add wrote 7778 warnings, then the
    tunnel broke and ssh exited 255 with the broken pipe as its last line."""
    stderr = "\n".join([WARNING] * 7777 + [BROKEN_PIPE]) + "\n"
    _test_hooks.run = FakeRun([failed(remote.SSH_FAILURE, stderr)])

    with pytest.raises(AppError) as raised:
        remote.run_ssh("loki", ("powershell", "-File", "C:/fleet/stage/run.stage/init-repo.ps1"))

    assert raised.value.code is FleetErrorCode.NODE_UNREACHABLE
    message = raised.value.message
    assert message.endswith(BROKEN_PIPE)
    assert f"[7758 {LEFT_OUT.format(chars=0)}" in message
    assert len(message) < TAIL_CHARS + 400
