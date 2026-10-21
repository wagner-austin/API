"""Felling a match so nothing survives, and finding who holds a match port.

The fell was verified against a real Linux before it was written down: a
parent spawned with ``start_new_session`` becomes a process-group leader, its
child inherits the group, and ``kill -s KILL -- -<pid>`` returns 0 and fells
both. What the tests here pin is that the module composes exactly that.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.process_tree import (
    ENGINE_PROCESS_NAME,
    PORT_LISTENER_COMMAND,
    fell_command,
    holder_is_an_orphaned_engine,
    parse_port_listener,
    parse_process_name,
    process_name_command,
)

#: Captured from a real ``ss -Hltnp`` while two sockets held their ports.
#: Written out verbatim rather than shaped by hand: a parser tested against
#: invented output tests the invention.
REAL_SS = (
    'LISTEN 0      1           127.0.0.1:27510 0.0.0.0:* users:(("java",pid=4242,fd=5))   ',
    'LISTEN 0      1           127.0.0.1:27511 0.0.0.0:* users:(("python3",pid=3753678,fd=3))   ',
)


class TestFellingTheTree:
    def test_it_signals_the_negated_pid_as_a_group(self) -> None:
        """Verified against a real Linux: this exact vector returned 0 and
        killed both the parent and the child it had spawned."""
        assert fell_command(1234) == ("kill", "-s", "KILL", "--", "-1234")

    def test_it_stops_option_parsing_before_the_negative_pid(self) -> None:
        """Without the separator the negated pid reads as a bundle of flags
        rather than as a process group."""
        vector = fell_command(99)
        assert vector[vector.index("-99") - 1] == "--"

    @pytest.mark.parametrize("pid", [0, -1, -1234])
    def test_a_non_positive_pid_is_refused(self, pid: int) -> None:
        """Negating these produces a signal target that means something else
        entirely: 0 is "every process in my own group", which would fell the
        harness itself, and -1 is "every process I may signal"."""
        with pytest.raises(ValueError, match="felled by its root pid"):
            fell_command(pid)


class TestFindingWhoHoldsAPort:
    def test_the_listing_suppresses_its_header(self) -> None:
        assert PORT_LISTENER_COMMAND == ("ss", "-Hltnp")

    def test_it_reads_the_owning_pid_off_a_real_row(self) -> None:
        assert parse_port_listener(REAL_SS, 27511) == 3753678

    def test_the_right_row_is_picked_out_of_several(self) -> None:
        """Every worker's leased port shows up in one listing, so picking the
        first LISTEN row would fell another live match."""
        assert parse_port_listener(REAL_SS, 27510) == 4242

    def test_a_connection_to_the_port_is_not_a_holder_of_it(self) -> None:
        """An established row names the port as its PEER. Matching the port
        anywhere on the line would read the planner's own connection as the
        engine and fell the match this launch just started."""
        established = (
            'ESTAB 0 0 127.0.0.1:51000 127.0.0.1:27511 users:(("python3",pid=900,fd=3))',
        )
        assert parse_port_listener(established, 27511) is None

    def test_a_free_port_has_no_holder(self) -> None:
        assert parse_port_listener(REAL_SS, 27999) is None

    def test_a_short_row_is_not_read_past_its_end(self) -> None:
        assert parse_port_listener(("LISTEN 0 1",), 27511) is None

    def test_a_socket_this_user_cannot_see_into_reads_as_free(self) -> None:
        """``ss`` omits the users:() field for another user's socket. Reading
        that as "no holder" is deliberate: a port this harness cannot see into
        is not one it may fell."""
        opaque = ("LISTEN 0      1           127.0.0.1:27511 0.0.0.0:*",)
        assert parse_port_listener(opaque, 27511) is None


class TestNamingTheHolder:
    def test_it_asks_ps_for_the_bare_command_name(self) -> None:
        assert process_name_command(55116) == ("ps", "-p", "55116", "-o", "comm=")

    def test_a_non_positive_pid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="named by a positive pid"):
            process_name_command(0)

    def test_it_names_it_off_the_bare_line(self) -> None:
        assert parse_process_name(("java\n",)) == "java"

    def test_blank_lines_before_the_name_are_skipped(self) -> None:
        assert parse_process_name(("", "java")) == "java"

    def test_a_departed_process_is_absent(self) -> None:
        """ps prints nothing at all and exits 1."""
        assert parse_process_name(("", "  ")) is None


class TestWhetherTheHolderMayBeFelled:
    def test_a_jvm_holding_a_match_port_is_always_an_orphan(self) -> None:
        """Only a JVM can legitimately hold one, and this launch has not
        started its own engine yet."""
        assert ENGINE_PROCESS_NAME == "java"
        assert holder_is_an_orphaned_engine("java") is True

    def test_anything_else_is_left_alone_so_the_bind_fails_loudly(self) -> None:
        """Felling a bystander is worse than failing to start: the match that
        does not start is reported, and the process that vanishes is not."""
        assert holder_is_an_orphaned_engine("python3") is False

    def test_an_unnameable_holder_is_left_alone(self) -> None:
        """The pid was listening a moment ago and is gone now, or is another
        user's. Either way there is nothing to fell."""
        assert holder_is_an_orphaned_engine(None) is False
