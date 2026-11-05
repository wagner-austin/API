"""The ssh seam: what every remote command, send and copy carries.

Split from ``test_core_io.py`` at the 600-line ceiling. `FakeRun` is given
to ``_test_hooks.run`` and satisfies ``RunProtocol``; nothing is patched.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, remote
from tests.conftest import FakeRun, failed, ok, timed_out


class TestRemote:
    def test_a_command_returns_its_stdout(self) -> None:
        runner = FakeRun([ok("done")])
        _test_hooks.run = runner

        assert remote.run_ssh("lavender", ("echo", "hi")) == "done"
        assert runner.calls[0][0] == "ssh"
        assert "BatchMode=yes" in runner.calls[0]
        assert runner.calls[0][-2:] == ("echo", "hi")

    def test_every_ssh_carries_the_keepalive_and_a_deadline(self) -> None:
        """The two bounds a dead or silent peer cannot escape (board tasks
        35940277 and 41ac6ed2): ssh's own keepalive for a peer that went
        away, and the seam's deadline for one that answers and never
        finishes. Both the run and the send carry them."""
        runner = FakeRun([ok(""), ok("")])
        _test_hooks.run = runner

        remote.run_ssh("pendragon", ("hostname",))
        remote.send_script("pendragon", "/s/probe.sh", "date")

        for call in runner.calls:
            # Every `-o Key=Value` pair the invocation carries, by key.
            settings = [call[index + 1] for index, word in enumerate(call) if word == "-o"]
            options = dict(setting.split("=", 1) for setting in settings)
            assert options["ServerAliveInterval"] == "15"
            assert options["ServerAliveCountMax"] == "4"
            assert options["ConnectTimeout"] == "10"
        assert runner.timeouts == [remote.SSH_TIMEOUT_SECONDS] * 2 == [120, 120]

    def test_a_copy_has_the_deadline_its_size_takes_at_the_floor_rate(self) -> None:
        """A flat two minutes cut the hub's 240 MB companion bundle on loki,
        which copies at about 2 MB/s (fleet job 851300a6, MCPs board task
        bfe2b3da), so a copy's deadline grows with the file."""
        runner = FakeRun([ok(""), ok("")])
        _test_hooks.run = runner

        remote.send_file("loki", pathlib.Path("/hub/a.tgz"), "/stage/a.tgz", size_bytes=13)
        remote.send_file(
            "loki", pathlib.Path("/hub/b.bundle"), "/stage/b.bundle", size_bytes=239_780_392
        )

        # 13 bytes take one second at the floor rate, the bundle 458 (457.3);
        # at loki's measured rate the bundle copies in about 115 s of its 578.
        assert runner.timeouts == [121, 578]
        assert remote.send_deadline_seconds(0) == remote.SSH_TIMEOUT_SECONDS

    def test_a_peer_that_stops_answering_is_unreachable_with_the_elapsed_seconds(self) -> None:
        _test_hooks.run = FakeRun([timed_out(120)])

        with pytest.raises(AppError) as excinfo:
            remote.run_ssh("pendragon", ("/bin/sh", "observe.sh"))

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert excinfo.value.message == (
            "ssh to pendragon timed out while running `/bin/sh observe.sh`: timed out after 120 s"
        )

    def test_ssh_failing_to_reach_the_node_is_its_own_code(self) -> None:
        """255 is ssh's own status, and the fix is the tailnet not the work."""
        _test_hooks.run = FakeRun([failed(255, "Connection timed out")])

        with pytest.raises(AppError) as excinfo:
            remote.run_ssh("pendragon", ("echo", "hi"))

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "Connection timed out" in excinfo.value.message

    def test_a_remote_command_failing_is_a_dispatch_failure(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "make: *** [check] Error 2")])

        with pytest.raises(AppError) as excinfo:
            remote.run_ssh("lavender", ("make", "check"))

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "Error 2" in excinfo.value.message

    def test_a_failure_with_no_stderr_still_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "")])

        with pytest.raises(AppError, match="<no stderr>"):
            remote.run_ssh("lavender", ("make", "check"))

    def test_a_script_body_is_streamed_over_stdin_through_mkdir_and_cat(self) -> None:
        """Never an argument, so no shell between here and the disk sees it:
        one argument for the remote login shell, the parent made in the same
        round trip."""
        runner = FakeRun([ok("")])
        _test_hooks.run = runner

        remote.send_script("diphtheria", "/home/c/stage/probe.sh", "echo hi")

        assert runner.stdin[0] == b"echo hi"
        assert runner.calls[0][-1] == (
            "mkdir -p \"$(dirname '/home/c/stage/probe.sh')\" && cat > '/home/c/stage/probe.sh'"
        )

    def test_an_unreachable_node_during_send_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "no route to host")])

        with pytest.raises(AppError) as excinfo:
            remote.send_script("pendragon", "/s/probe.sh", "x")

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE

    def test_a_root_file_is_streamed_into_install_as_root_s_alone(self) -> None:
        """A registration token, so never an argument and never readable by
        anyone but root: install writes it at 0600, its parents made."""
        runner = FakeRun([ok("")])
        _test_hooks.run = runner

        remote.send_root_file("lavender", "/var/lib/tokens/RUNNER_TOKEN_API", "TOK\n")

        assert runner.stdin[0] == b"TOK\n"
        assert runner.calls[0][-1] == (
            "sudo -n install -D -m 0600 /dev/stdin /var/lib/tokens/RUNNER_TOKEN_API"
        )

    def test_a_root_file_whose_write_fails_says_which_path(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "sudo: a password is required")])

        with pytest.raises(AppError) as excinfo:
            remote.send_root_file("lavender", "/var/lib/tokens/T", "x")

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "writing /var/lib/tokens/T as root" in excinfo.value.message
        assert "a password is required" in excinfo.value.message

    def test_a_root_file_path_a_shell_would_reinterpret_is_refused_before_any_call(self) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner

        with pytest.raises(ValueError, match="root file path '/var/lib/\\$x' contains '\\$'"):
            remote.send_root_file("lavender", "/var/lib/$x", "x")

        assert runner.calls == []

    def test_a_write_that_fails_is_a_dispatch_failure(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "cat: /s/probe.sh: Permission denied")])

        with pytest.raises(AppError) as excinfo:
            remote.send_script("lavender", "/s/probe.sh", "x")

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "Permission denied" in excinfo.value.message

    def test_run_script_sends_then_runs_by_path_through_bin_sh(self) -> None:
        """The bytes that run are the bytes that were sent."""
        runner = FakeRun([ok(""), ok("output")])
        _test_hooks.run = runner

        assert remote.run_script("diphtheria", "/s/p.sh", "body") == "output"
        assert runner.stdin[0] == b"body"
        assert runner.calls[1][-2:] == ("/bin/sh", "/s/p.sh")

    def test_a_root_payload_runs_under_sudo_bash_under_its_deadline(self) -> None:
        """Only the run takes the caller's deadline; the send is a short
        command like any other."""
        runner = FakeRun([ok(""), ok("installed")])
        _test_hooks.run = runner

        output = remote.run_root_script("colossus", "/s/p.sh", "body", timeout_seconds=900)

        assert output == "installed"
        assert runner.calls[1][-len(remote.ROOT_BASH) - 1 :] == (*remote.ROOT_BASH, "/s/p.sh")
        assert runner.timeouts == [remote.SSH_TIMEOUT_SECONDS, 900]
