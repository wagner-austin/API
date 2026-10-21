"""The options a hub gives every ssh and scp to a node (MCPs board task ee3d6cdc, A2).

A POSIX hub shares one connection per node across a job's calls, so loki's
tunnel is crossed once per job rather than once per call; a Windows hub,
whose OpenSSH has no multiplexing, keeps one connection per call. Every
call site sends :data:`fleet.core.ssh_options.SSH_OPTIONS`, the hub's own.
"""

from __future__ import annotations

import pathlib
import sys

from fleet.contracts.node import NodePlatform
from fleet.core import _test_hooks, remote
from fleet.core.ssh_options import (
    BASE_OPTIONS,
    MASTER_PERSIST_SECONDS,
    SSH_OPTIONS,
    ssh_options,
)
from tests.conftest import FakeRun, ok


def _settings(options: tuple[str, ...]) -> dict[str, str]:
    """Every ``-o Key=Value`` pair the options carry, by key."""
    pairs = [options[index + 1] for index, word in enumerate(options) if word == "-o"]
    return dict(pair.split("=", 1) for pair in pairs)


class TestAPosixHub:
    def test_shares_one_connection_per_node_and_keeps_it_past_its_last_session(self) -> None:
        settings = _settings(ssh_options("linux"))
        assert settings["ControlMaster"] == "auto"
        # %C is ssh's hash of the local host, the node, its port and its user.
        assert settings["ControlPath"] == "~/.ssh/fleet-mux-%C"
        assert settings["ControlPersist"] == str(MASTER_PERSIST_SECONDS) == "300"

    def test_keeps_every_bound_a_dead_or_silent_peer_meets(self) -> None:
        options = ssh_options("linux")
        assert options[: len(BASE_OPTIONS)] == BASE_OPTIONS
        settings = _settings(options)
        assert (
            settings["BatchMode"],
            settings["ConnectTimeout"],
            settings["ServerAliveInterval"],
            settings["ServerAliveCountMax"],
        ) == ("yes", "10", "15", "4")


class TestAWindowsHub:
    def test_opens_one_connection_per_call(self) -> None:
        assert ssh_options("win32") == BASE_OPTIONS
        assert not any(key.startswith("Control") for key in _settings(ssh_options("win32")))


class TestTheCallSites:
    def test_every_ssh_and_scp_carries_this_hubs_options(self) -> None:
        runner = FakeRun([ok(""), ok(""), ok("")])
        _test_hooks.run = runner

        remote.run_ssh("loki", ("hostname",))
        remote.send_script("loki", "C:/fleet/probe.ps1", "Get-Date", platform=NodePlatform.WINDOWS)
        remote.send_file("loki", pathlib.Path("/hub/a.tgz"), "C:/fleet/a.tgz", size_bytes=13)

        assert ssh_options(sys.platform) == SSH_OPTIONS
        for call in runner.calls:
            assert call[1 : 1 + len(SSH_OPTIONS)] == SSH_OPTIONS or (
                call[:2] == ("scp", "-q") and call[2 : 2 + len(SSH_OPTIONS)] == SSH_OPTIONS
            )
