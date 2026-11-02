"""Registering a NixOS runner host's runners, against fakes at the command seam.

MCPs board task 9aba868d.

Every command the registration runs is answered through the same
``_test_hooks.run`` seam production uses, so these assert what reaches the
host: the probe's text, each token's path and bytes, each restart, and the
order of all of them.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import _test_hooks, remote, runner_audit, runner_register
from fleet.core.runner_install import TOKEN_DIRECTORY, install_root_for
from tests._runner_fixtures import a_host, an_install, no_dropped_jobs, register_answers
from tests.conftest import FakeClock, FakeRun, FakeSleep, failed, ok

#: The probe's landing path on the fixture host.
_PROBE = "/home/corvis/fleet/runner-payloads/fleet-register-probe.sh"


def _host() -> HostRunnerSpec:
    """lavender with two API installs and one MCPs install.

    Returns:
        The spec.
    """
    return a_host(
        [
            an_install("wagner-austin/API"),
            an_install("wagner-austin/API", "lavender-2", slot=2),
            an_install("wagner-austin/MCPs"),
        ]
    )


def _clean(spec: HostRunnerSpec) -> str:
    """One OK line per expected check.

    Args:
        spec: The host.

    Returns:
        The transcript.
    """
    return (
        "\n".join(f"CHECK {check['check_id']} OK" for check in runner_audit.expected_checks(spec))
        + "\n"
    )


class TestTheProbe:
    def test_it_names_each_install_root_with_no_registration(self) -> None:
        assert runner_register.render_probe(_host()) == (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "for root in '/home/gharunner/actions-runner-api-1' "
            "'/home/gharunner/actions-runner-api-2' '/home/gharunner/actions-runner-mcps-1'; do\n"
            '    [ -f "$root/.runner" ] || echo "$root"\n'
            "done\n"
        )

    def test_a_root_a_script_cannot_carry_is_refused(self) -> None:
        install = an_install()
        install["workdir"] = "/home/gharunner/it's/_work"
        with pytest.raises(ValueError, match="install root"):
            runner_register.render_probe(a_host([install]))

    def test_the_installs_it_names_are_returned_in_roster_order(self) -> None:
        spec = _host()
        mcps = install_root_for(spec["installs"][2])
        api = install_root_for(spec["installs"][0])
        runner = FakeRun([ok(""), ok(f"{mcps}\n{api}\n")])
        _test_hooks.run = runner
        assert runner_register.unregistered(spec) == [spec["installs"][0], spec["installs"][2]]
        assert runner.stdin[0] == runner_register.render_probe(spec).encode("utf-8")
        assert tuple(runner.calls[1][-4:]) == (*remote.ROOT_BASH, _PROBE)
        assert runner.timeouts[1] == runner_register.PROBE_TIMEOUT_SECONDS


class TestRegister:
    def test_each_repository_s_token_is_written_once_and_each_unit_restarted(self) -> None:
        spec = _host()
        runner = FakeRun(register_answers(_clean(spec), spec))
        _test_hooks.run = runner
        report = runner_register.register(spec)
        assert report["steps"] == [
            f"tokens: minted for 2 repositories and written to {TOKEN_DIRECTORY}",
            "registered: 3 runner(s), actions.runner.wagner-austin-API.lavender.service, "
            "actions.runner.wagner-austin-API.lavender-2.service, "
            "actions.runner.wagner-austin-MCPs.lavender.service",
        ]
        assert [finding["ok"] for finding in report["findings"]] == [True] * (
            len(runner_audit.expected_checks(spec)) + 1
        )
        assert runner.calls[2][4] == "repos/wagner-austin/API/actions/runners/registration-token"
        assert runner.calls[3][-1] == (
            f"sudo -n install -D -m 0600 /dev/stdin {TOKEN_DIRECTORY}/RUNNER_TOKEN_API"
        )
        assert runner.stdin[3] == b"TOKEN0\n"
        assert runner.calls[4][4] == "repos/wagner-austin/MCPs/actions/runners/registration-token"
        assert runner.calls[5][-1] == (
            f"sudo -n install -D -m 0600 /dev/stdin {TOKEN_DIRECTORY}/RUNNER_TOKEN_MCPS"
        )
        assert runner.stdin[5] == b"TOKEN1\n"
        assert [tuple(call[-6:]) for call in runner.calls[6:9]] == [
            ("sudo", "-n", "systemctl", "restart", "--no-block", install["service"])
            for install in spec["installs"]
        ]
        assert len(runner.calls) == 9 + 4 + len(no_dropped_jobs(spec))

    def test_a_host_already_registered_mints_nothing_and_is_audited(self) -> None:
        spec = _host()
        runner = FakeRun([ok(""), ok(""), ok(""), ok(_clean(spec)), *no_dropped_jobs(spec)])
        _test_hooks.run = runner
        report = runner_register.register(spec)
        assert report["steps"] == ["registered: all 3 runner(s) already were; nothing minted"]
        assert len(runner.calls) == 4 + len(no_dropped_jobs(spec))
        assert not any("registration-token" in " ".join(call) for call in runner.calls)

    def test_a_runner_that_registers_late_is_waited_for(self) -> None:
        spec = a_host([an_install()])
        root = f"{install_root_for(spec['installs'][0])}\n"
        clock = FakeClock(1_000)
        sleep = FakeSleep(clock)
        _test_hooks.now = clock
        _test_hooks.sleep = sleep
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(root),
                ok("TOK\n"),
                ok(""),
                ok(""),
                ok(""),
                ok(root),
                ok(""),
                ok(""),
                ok(""),
                ok(_clean(spec)),
                *no_dropped_jobs(spec),
            ]
        )
        report = runner_register.register(spec)
        assert sleep.slept == [runner_register.REGISTER_POLL_SECONDS]
        assert report["steps"][-1] == (
            "registered: 1 runner(s), actions.runner.wagner-austin-API.lavender.service"
        )

    def test_a_runner_still_unregistered_at_the_deadline_is_named(self) -> None:
        spec = a_host([an_install()])
        root = f"{install_root_for(spec['installs'][0])}\n"
        clock = FakeClock(1_000)
        _test_hooks.now = clock
        _test_hooks.sleep = FakeSleep(clock)
        polls = runner_register.REGISTER_DEADLINE_SECONDS // runner_register.REGISTER_POLL_SECONDS
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(root),
                ok("TOK\n"),
                ok(""),
                ok(""),
                *(answer for _ in range(polls + 1) for answer in (ok(""), ok(root))),
            ]
        )
        with pytest.raises(AppError) as fault:
            runner_register.register(spec)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert fault.value.message == (
            "lavender: actions.runner.wagner-austin-API.lavender.service held no registration "
            f"{runner_register.REGISTER_DEADLINE_SECONDS}s after 1 restart(s); "
            "journalctl -u <unit> on the host says why"
        )

    def test_a_mint_gh_refuses_writes_nothing_to_the_host(self) -> None:
        spec = a_host([an_install()])
        root = f"{install_root_for(spec['installs'][0])}\n"
        runner = FakeRun([ok(""), ok(root), failed(1, "HTTP 403: Resource not accessible")])
        _test_hooks.run = runner
        with pytest.raises(AppError) as fault:
            runner_register.register(spec)
        assert fault.value.code is FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE
        assert len(runner.calls) == 3

    def test_an_audit_that_cannot_reach_the_host_is_refused(self) -> None:
        spec = _host()
        _test_hooks.run = FakeRun([ok(""), ok(""), failed(255, "No route to host")])
        with pytest.raises(AppError) as fault:
            runner_register.register(spec)
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert fault.value.message.startswith(
            "the audit after registering could not reach lavender: "
        )
        assert fault.value.message.endswith(
            "Every step ran; re-run the audit before trusting the host."
        )
