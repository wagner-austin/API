"""The options the hub gives every ssh and scp to a node (MCPs board task ee3d6cdc, A2).

Every call opens its own connection and never shares one: over loki's
tunnel a shared connection hung 8 of 200 sends against none of 100 fresh
ones (measured 2026-10-21). Every call site sends
:data:`fleet.core.ssh_options.SSH_OPTIONS`.
"""

from __future__ import annotations

import pathlib

from fleet.core import _test_hooks, remote
from fleet.core.ssh_options import SSH_OPTIONS
from tests.conftest import FakeRun, ok


def _settings(options: tuple[str, ...]) -> dict[str, str]:
    """Every ``-o Key=Value`` pair the options carry, by key."""
    assert options[::2] == ("-o",) * (len(options) // 2)
    return dict(pair.split("=", 1) for pair in options[1::2])


class TestTheOptions:
    def test_open_a_connection_per_call_and_use_no_shared_one(self) -> None:
        settings = _settings(SSH_OPTIONS)
        # none is ssh's word for no control socket: no master is made, and
        # one the hub's ssh config or an earlier release names is not used.
        assert settings["ControlPath"] == "none"
        assert "ControlMaster" not in settings
        assert "ControlPersist" not in settings

    def test_keep_every_bound_a_dead_or_silent_peer_meets(self) -> None:
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

        remote.run_ssh("colossus", ("hostname",))
        remote.send_script("colossus", "/home/corvis/fleet/probe.sh", "date")
        remote.send_file(
            "colossus", pathlib.Path("/hub/a.tgz"), "/home/corvis/fleet/a.tgz", size_bytes=13
        )

        assert [call[0] for call in runner.calls] == ["ssh", "ssh", "scp"]
        for call in runner.calls:
            assert call[1 : 1 + len(SSH_OPTIONS)] == SSH_OPTIONS or (
                call[:2] == ("scp", "-q") and call[2 : 2 + len(SSH_OPTIONS)] == SSH_OPTIONS
            )
