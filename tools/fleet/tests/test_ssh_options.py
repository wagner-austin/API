"""The options the hub gives every ssh and scp to a node (MCPs board task ee3d6cdc, A2).

The hub shares one connection per node across a job's calls, so loki's
tunnel is crossed once per job rather than once per call. Every call site
sends :data:`fleet.core.ssh_options.SSH_OPTIONS`.
"""

from __future__ import annotations

import pathlib

from fleet.contracts.node import NodePlatform
from fleet.core import _test_hooks, remote
from fleet.core.ssh_options import BASE_OPTIONS, MASTER_PERSIST_SECONDS, SSH_OPTIONS
from tests.conftest import FakeRun, ok


def _settings(options: tuple[str, ...]) -> dict[str, str]:
    """Every ``-o Key=Value`` pair the options carry, by key."""
    pairs = [options[index + 1] for index, word in enumerate(options) if word == "-o"]
    return dict(pair.split("=", 1) for pair in pairs)


class TestTheOptions:
    def test_share_one_connection_per_node_and_keep_it_past_its_last_session(self) -> None:
        settings = _settings(SSH_OPTIONS)
        assert settings["ControlMaster"] == "auto"
        # %C is ssh's hash of the local host, the node, its port and its user.
        assert settings["ControlPath"] == "~/.ssh/fleet-mux-%C"
        assert settings["ControlPersist"] == str(MASTER_PERSIST_SECONDS) == "300"

    def test_keep_every_bound_a_dead_or_silent_peer_meets(self) -> None:
        assert SSH_OPTIONS[: len(BASE_OPTIONS)] == BASE_OPTIONS
        settings = _settings(SSH_OPTIONS)
        assert (
            settings["BatchMode"],
            settings["ConnectTimeout"],
            settings["ServerAliveInterval"],
            settings["ServerAliveCountMax"],
        ) == ("yes", "10", "15", "4")


class TestTheCallSites:
    def test_every_ssh_and_scp_carries_the_options(self) -> None:
        runner = FakeRun([ok(""), ok(""), ok("")])
        _test_hooks.run = runner

        remote.run_ssh("loki", ("hostname",))
        remote.send_script("loki", "C:/fleet/probe.ps1", "Get-Date", platform=NodePlatform.WINDOWS)
        remote.send_file("loki", pathlib.Path("/hub/a.tgz"), "C:/fleet/a.tgz", size_bytes=13)

        assert [call[0] for call in runner.calls] == ["ssh", "ssh", "scp"]
        for call in runner.calls:
            assert call[1 : 1 + len(SSH_OPTIONS)] == SSH_OPTIONS or (
                call[:2] == ("scp", "-q") and call[2 : 2 + len(SSH_OPTIONS)] == SSH_OPTIONS
            )
