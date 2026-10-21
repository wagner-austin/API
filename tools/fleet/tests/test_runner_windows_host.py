"""A Windows runner host's audit and rebuild, over the command seam.

Every ssh and gh call answers through ``_test_hooks.run`` exactly as
:mod:`tests.test_core_io` describes, so each case asserts the commands
production runs, the bytes it sends and the deadlines it chose.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runner_windows import WindowsRunnerHost
from fleet.core import _test_hooks, dialect_windows, runner_windows_host, runner_windows_render
from fleet.core.runner_audit_types import AuditFinding, AuditOutcome
from tests._runner_fixtures import a_windows_host
from tests.conftest import FakeRun, failed, ok


def _clean(host: WindowsRunnerHost) -> str:
    """The audit's output on a host where every row holds.

    Args:
        host: The host.

    Returns:
        One OK line per expected check.
    """
    checks = runner_windows_render.expected_checks(host)
    return "".join(f"CHECK {check['check_id']} OK\n" for check in checks)


class TestTheAudit:
    def test_the_audit_is_sent_to_the_payload_directory_and_run_in_the_elevated_session(
        self,
    ) -> None:
        host = a_windows_host()
        runner = FakeRun([ok(""), ok(_clean(host))])
        _test_hooks.run = runner
        outcome = runner_windows_host.attempt_audit_windows_host(host)
        assert outcome == AuditOutcome(
            findings=[
                AuditFinding(check_id=check["check_id"], ok=True, detail="", reason=check["reason"])
                for check in runner_windows_render.expected_checks(host)
            ],
            reason="",
        )
        path = "C:/fleet/runner-payloads/runner-windows-audit.ps1"
        assert runner.stdin[0] == runner_windows_render.render_audit_script(host).encode("utf-8")
        assert runner.calls[1][-6:] == (*dialect_windows.POWERSHELL_INVOCATION, path)
        assert runner.timeouts[1] == runner_windows_host.AUDIT_TIMEOUT_SECONDS

    def test_an_unreachable_host_is_a_reason_not_a_crash(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "ssh: connect to host loki: No route to host")])
        outcome = runner_windows_host.attempt_audit_windows_host(a_windows_host())
        assert outcome["findings"] is None
        assert "No route to host" in outcome["reason"]

    def test_a_transcript_that_stops_early_is_refused(self) -> None:
        host = a_windows_host()
        _test_hooks.run = FakeRun([ok(""), ok(_clean(host).splitlines()[0] + "\n")])
        with pytest.raises(AppError) as fault:
            runner_windows_host.attempt_audit_windows_host(host)
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE


class TestTheRebuild:
    def test_tokens_are_minted_the_install_runs_with_them_and_the_host_is_audited(
        self,
    ) -> None:
        host = a_windows_host()
        installed = "runner: unpacked the release\n\nrunner: registered wagner-austin/chat\n"
        runner = FakeRun([ok("TOKEN1\n"), ok(""), ok(installed), ok(""), ok(_clean(host))])
        _test_hooks.run = runner
        report = runner_windows_host.rebuild_windows_host(host)
        assert report["steps"] == [
            "tokens: minted for 1 repositories",
            "runner: unpacked the release",
            "runner: registered wagner-austin/chat",
        ]
        # The base's four rows, the job hook's two, the one runner's three and
        # its one tool cache row.
        assert [finding["ok"] for finding in report["findings"]] == [True] * 10
        assert runner.calls[0][4] == "repos/wagner-austin/chat/actions/runners/registration-token"
        assert runner.stdin[1] == runner_windows_render.render_install_script(
            host, {"wagner-austin/chat": "TOKEN1"}
        ).encode("utf-8")
        assert runner.calls[2][-1] == "C:/fleet/runner-payloads/runner-windows-install.ps1"
        assert runner.timeouts[2] == runner_windows_host.INSTALL_TIMEOUT_SECONDS

    def test_a_host_that_stops_answering_after_its_install_is_raised(self) -> None:
        host = a_windows_host()
        _test_hooks.run = FakeRun(
            [ok("TOKEN1\n"), ok(""), ok("runner: done\n"), failed(255, "Connection reset")]
        )
        with pytest.raises(AppError) as fault:
            runner_windows_host.rebuild_windows_host(host)
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert fault.value.message.startswith(
            "loki stopped answering between its install and its audit: "
        )

    def test_an_install_that_fails_on_the_host_is_raised_before_any_audit(self) -> None:
        host = a_windows_host()
        runner = FakeRun(
            [ok("TOKEN1\n"), ok(""), failed(1, "RUNNER_CONFIG_FAILED: config.cmd exited 1")]
        )
        _test_hooks.run = runner
        with pytest.raises(AppError) as fault:
            runner_windows_host.rebuild_windows_host(host)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert len(runner.calls) == 3
