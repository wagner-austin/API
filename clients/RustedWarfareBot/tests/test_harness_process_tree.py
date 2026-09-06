"""Spawning a match so it can be felled, and felling it so nothing survives.

The POSIX arm of this was verified against a real Linux before it was written
down: a parent spawned with ``start_new_session`` becomes a process-group
leader, its child inherits the group, and ``kill -s KILL -- -<pid>`` returns 0
and fells both. What the tests here pin is that the module composes exactly
that, from either platform, without needing one to run.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from rw_bot.harness.process_tree import (
    ABOVE_NORMAL_PRIORITY_CLASS,
    NO_CREATION_FLAGS,
    fell_command,
    holder_is_an_orphaned_engine,
    parse_port_listener,
    parse_process_name,
    port_listener_command,
    process_name_command,
    spawn_isolation,
)
from rw_bot.platform_id import WINDOWS

LINUX = "linux"

#: Captured from a real ``netstat -ano -p TCP`` on this machine while a socket
#: held the port. Written out verbatim rather than shaped by hand: a parser
#: tested against invented output tests the invention.
WINDOWS_NETSTAT = (
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    127.0.0.1:27510        0.0.0.0:0              LISTENING       4242",
    "  TCP    127.0.0.1:27511        0.0.0.0:0              LISTENING       55116",
    "  TCP    127.0.0.1:51000        127.0.0.1:27511        ESTABLISHED     900",
)

#: Captured from a real ``ss -Hltnp`` inside WSL2 under the same conditions.
POSIX_SS = (
    'LISTEN 0      1           127.0.0.1:27510 0.0.0.0:* users:(("java",pid=4242,fd=5))   ',
    'LISTEN 0      1           127.0.0.1:27511 0.0.0.0:* users:(("python3",pid=3753678,fd=3))   ',
)


class TestSpawnIsolation:
    def test_windows_ranks_the_match_and_takes_no_session(self) -> None:
        """The priority class is the load-bearing half there: without it a
        co-tenant spike deschedules the batch work rather than the sample
        stream (log 2026-08-10)."""
        assert spawn_isolation(WINDOWS) == {
            "creationflags": ABOVE_NORMAL_PRIORITY_CLASS,
            "start_new_session": False,
        }

    def test_posix_takes_a_session_and_no_flags(self) -> None:
        """The session is the load-bearing half there: it is what makes the
        match a process group, and the group is what fell_command signals."""
        assert spawn_isolation(LINUX) == {
            "creationflags": NO_CREATION_FLAGS,
            "start_new_session": True,
        }

    def test_posix_asks_for_no_creation_flags_because_anything_else_raises(
        self,
    ) -> None:
        """Not a default: the standard library rejects a non-zero
        ``creationflags`` off Windows outright, so zero is the only legal
        answer there."""
        assert spawn_isolation(LINUX)["creationflags"] == 0

    def test_posix_does_not_try_to_raise_priority(self) -> None:
        """Raising it means a negative nice value, which needs privileges a
        batch job does not have -- and the contention the Windows class
        survives is a workstation condition, not a scheduled core's."""
        assert spawn_isolation(LINUX)["creationflags"] != ABOVE_NORMAL_PRIORITY_CLASS

    @pytest.mark.skipif(sys.platform != WINDOWS, reason="the constant exists on Windows only")
    def test_the_written_out_constant_matches_the_standard_library(self) -> None:
        """The value is written out because the NAME exists only in a Windows
        Python, and importing it would break the module on the platform this
        port is for. This is what stops the literal drifting from the constant
        it stands in for.

        The inner guard is for the TYPE CHECKER, not the runtime -- the marker
        above already stops this running off Windows. mypy analyses a skipped
        test's body like any other, so on a Linux runner it resolves the
        attribute against Linux stubs, does not find it, and fails the lint.
        Narrowing on ``sys.platform`` makes the block statically unreachable
        there while leaving it fully checked on Windows.

        The comparison must be against the LITERAL ``"win32"``. mypy narrows
        this on the syntactic form alone: ``sys.platform == WINDOWS``, using
        this package's own constant, does not narrow and leaves the lint red.
        That is why the string is spelled out here and nowhere else.
        """
        if sys.platform == "win32":
            assert ABOVE_NORMAL_PRIORITY_CLASS == subprocess.ABOVE_NORMAL_PRIORITY_CLASS


class TestFellingTheTree:
    def test_windows_walks_the_process_table(self) -> None:
        """/T fells the tree, /F skips the polite close request an unattended
        engine has nothing to answer with."""
        assert fell_command(1234, WINDOWS) == ("taskkill", "/PID", "1234", "/T", "/F")

    def test_posix_signals_the_negated_pid_as_a_group(self) -> None:
        """Verified against a real Linux: this exact vector returned 0 and
        killed both the parent and the child it had spawned."""
        assert fell_command(1234, LINUX) == ("kill", "-s", "KILL", "--", "-1234")

    def test_posix_stops_option_parsing_before_the_negative_pid(self) -> None:
        """Without the separator the negated pid reads as a bundle of flags
        rather than as a process group."""
        vector = fell_command(99, LINUX)
        assert vector[vector.index("-99") - 1] == "--"

    @pytest.mark.parametrize("pid", [0, -1, -1234])
    @pytest.mark.parametrize("platform", [WINDOWS, LINUX])
    def test_a_non_positive_pid_is_refused_on_either_platform(
        self, pid: int, platform: str
    ) -> None:
        """Negating these produces a POSIX signal target that means something
        else entirely: 0 is "every process in my own group", which would fell
        the harness itself, and -1 is "every process I may signal"."""
        with pytest.raises(ValueError, match="felled by its root pid"):
            fell_command(pid, platform)

    def test_the_two_platforms_share_no_mechanism_at_all(self) -> None:
        """Stated because it is the reason this is composition rather than a
        flag: there is no common command to parameterise."""
        assert fell_command(7, WINDOWS)[0] != fell_command(7, LINUX)[0]


class TestFindingWhoHoldsAPort:
    def test_both_forms_suppress_headers(self) -> None:
        assert port_listener_command(WINDOWS) == ("netstat", "-ano", "-p", "TCP")
        assert port_listener_command(LINUX) == ("ss", "-Hltnp")

    def test_windows_reads_the_owning_pid_off_a_real_row(self) -> None:
        assert parse_port_listener(WINDOWS_NETSTAT, 27511, WINDOWS) == 55116

    def test_posix_reads_the_owning_pid_off_a_real_row(self) -> None:
        assert parse_port_listener(POSIX_SS, 27511, LINUX) == 3753678

    def test_the_right_row_is_picked_out_of_several(self) -> None:
        """Every worker's leased port shows up in one listing, so picking the
        first LISTEN row would fell another live match."""
        assert parse_port_listener(WINDOWS_NETSTAT, 27510, WINDOWS) == 4242
        assert parse_port_listener(POSIX_SS, 27510, LINUX) == 4242

    def test_a_connection_to_the_port_is_not_a_holder_of_it(self) -> None:
        """The established row names 27511 as its FOREIGN address. Matching
        the port anywhere on the line would read the planner's own connection
        as the engine and fell the match this launch just started."""
        established: tuple[str, ...] = (
            "  TCP    127.0.0.1:51000        127.0.0.1:27511        ESTABLISHED     900",
        )
        assert parse_port_listener(established, 27511, WINDOWS) is None

    def test_a_free_port_has_no_holder(self) -> None:
        assert parse_port_listener(WINDOWS_NETSTAT, 27999, WINDOWS) is None
        assert parse_port_listener(POSIX_SS, 27999, LINUX) is None

    def test_a_posix_socket_this_user_cannot_see_into_reads_as_free(self) -> None:
        """``ss`` omits the users:() field for another user's socket. Reading
        that as "no holder" is deliberate: a port this harness cannot see into
        is not one it may fell."""
        opaque = ("LISTEN 0      1           127.0.0.1:27511 0.0.0.0:*",)
        assert parse_port_listener(opaque, 27511, LINUX) is None


class TestNamingTheHolder:
    def test_each_platform_asks_its_own_way(self) -> None:
        assert process_name_command(55116, WINDOWS) == (
            "tasklist",
            "/FI",
            "PID eq 55116",
            "/FO",
            "CSV",
            "/NH",
        )
        assert process_name_command(55116, LINUX) == ("ps", "-p", "55116", "-o", "comm=")

    @pytest.mark.parametrize("platform", [WINDOWS, LINUX])
    def test_a_non_positive_pid_is_refused(self, platform: str) -> None:
        with pytest.raises(ValueError, match="named by a positive pid"):
            process_name_command(0, platform)

    def test_windows_names_it_out_of_the_csv_row(self) -> None:
        assert parse_process_name(('"java.exe","55116","Services","0","13,488 K"',), WINDOWS) == (
            "java.exe"
        )

    def test_posix_names_it_off_the_bare_line(self) -> None:
        assert parse_process_name(("java\n",), LINUX) == "java"

    def test_windows_reports_a_departed_process_as_absent(self) -> None:
        """tasklist exits 0 whether or not the pid exists, so the notice text
        is the only signal that it is gone."""
        gone = ("INFO: No tasks are running which match the specified criteria.",)
        assert parse_process_name(gone, WINDOWS) is None

    def test_posix_reports_a_departed_process_as_absent(self) -> None:
        """ps prints nothing at all and exits 1."""
        assert parse_process_name(("", "  "), LINUX) is None


class TestWhetherTheHolderMayBeFelled:
    def test_a_jvm_holding_a_match_port_is_always_an_orphan(self) -> None:
        """Only a JVM can legitimately hold one, and this launch has not
        started its own engine yet."""
        assert holder_is_an_orphaned_engine("java.exe", WINDOWS) is True
        assert holder_is_an_orphaned_engine("java", LINUX) is True

    def test_anything_else_is_left_alone_so_the_bind_fails_loudly(self) -> None:
        """Felling a bystander is worse than failing to start: the match that
        does not start is reported, and the process that vanishes is not."""
        assert holder_is_an_orphaned_engine("python3", LINUX) is False
        assert holder_is_an_orphaned_engine("code.exe", WINDOWS) is False

    def test_an_unnameable_holder_is_left_alone(self) -> None:
        """The pid was listening a moment ago and is gone now, or is another
        user's. Either way there is nothing to fell."""
        assert holder_is_an_orphaned_engine(None, WINDOWS) is False
        assert holder_is_an_orphaned_engine(None, LINUX) is False

    def test_the_windows_suffix_is_not_ignored(self) -> None:
        """The name each tool reports carries the platform's own spelling, so
        the comparison has to use the platform's own spelling too."""
        assert holder_is_an_orphaned_engine("java", WINDOWS) is False
        assert holder_is_an_orphaned_engine("java.exe", LINUX) is False
