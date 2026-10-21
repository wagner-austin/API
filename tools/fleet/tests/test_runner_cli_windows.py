"""fleet-runners against a roster with a Windows runner host: what it does to loki, and refuses.

Every invocation goes through :func:`fleet.cli.runners.main` with a real
roster file and the command seam faked at ``_test_hooks.run``, as
tests/test_runner_cli.py's are.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.cli import runners
from fleet.contracts.runner_windows import decode_windows_runner_host
from fleet.core import _test_hooks, runner_windows_render
from tests._runner_fixtures import raw_windows_host
from tests._runner_roster_files import raw_host, write_roster
from tests.conftest import FakeRun, failed, ok

#: loki's one runner's service, as the fixture declares it.
_SERVICE = "actions.runner.wagner-austin-chat.loki"


def _clean() -> str:
    """loki's audit output when every row holds.

    Returns:
        One OK line per expected check: the base's four, then the runner's
        three and its tool cache row.
    """
    checks = runner_windows_render.expected_checks(decode_windows_runner_host(raw_windows_host()))
    return "".join(f"CHECK {check['check_id']} OK\n" for check in checks)


def _roster(tmp_path: pathlib.Path) -> str:
    """lavender and loki.

    Args:
        tmp_path: The case's directory.

    Returns:
        The roster's path.
    """
    return write_roster(tmp_path, [raw_host("lavender")], [raw_windows_host()])


class TestTheAudit:
    def test_loki_alone_is_audited_and_its_repository_s_registrations_held_to_the_roster(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = FakeRun([ok(""), ok(_clean()), ok("7 loki online false\n")])
        _test_hooks.run = runner
        with caplog.at_level("INFO"):
            assert runners.main(["--spec", _roster(tmp_path), "--host", "loki"]) == 0
        assert [record.getMessage() for record in caplog.records] == [
            "loki OK execution-policy:LocalMachine:RemoteSigned",
            "loki OK long-paths",
            "loki OK git-longpaths",
            "loki OK machine-env:POETRY_CACHE_DIR",
            "loki OK job-hook",
            "loki OK job-hook-variable",
            f"loki OK service:{_SERVICE}",
            f"loki OK account:{_SERVICE}",
            f"loki OK recovery:{_SERVICE}",
            f"loki OK python:{_SERVICE}:3.11.9",
            "roster OK registrations:wagner-austin/chat",
        ]
        assert runner.calls[2][3] == "repos/wagner-austin/chat/actions/runners"

    def test_an_unreachable_windows_host_is_a_line_and_a_fault(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        unreachable = failed(255, "ssh: connect to host loki: No route to host")
        _test_hooks.run = FakeRun([unreachable, ok("7 loki online false\n")])
        with caplog.at_level("INFO"):
            assert runners.main(["--spec", _roster(tmp_path), "--host", "loki"]) == 1
        messages = [record.getMessage() for record in caplog.records]
        assert messages[0].startswith("loki UNREACHABLE ")
        assert messages[-1] == "1 fault(s); the fleet does not match its roster"

    def test_an_unknown_host_names_the_windows_hosts_too(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", _roster(tmp_path), "--host", "pendragon"])
        assert fault.value.code is FleetErrorCode.RUNNER_HOST_UNKNOWN
        assert fault.value.message == (
            "the roster declares no host named 'pendragon'; it declares: lavender, loki"
        )


class TestTheRebuild:
    def test_loki_is_rebuilt_from_the_roster_and_verified(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _test_hooks.run = FakeRun(
            [ok("TOKEN1\n"), ok(""), ok("runner: registered\n"), ok(""), ok(_clean())]
        )
        with caplog.at_level("INFO"):
            code = runners.main(["--spec", _roster(tmp_path), "--host", "loki", "--rebuild", "yes"])
        assert code == 0
        messages = [record.getMessage() for record in caplog.records]
        assert messages[:2] == [
            "loki rebuild: tokens: minted for 1 repositories",
            "loki rebuild: runner: registered",
        ]
        assert messages[-1] == "rebuild verified: every check green on loki"

    def test_drift_after_the_rebuild_exits_one(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        drifted = _clean().replace(
            f"CHECK account:{_SERVICE} OK",
            f"CHECK account:{_SERVICE} DRIFT sc.exe qc exited 0: START_TYPE : 3 DEMAND_START",
        )
        _test_hooks.run = FakeRun([ok("TOKEN1\n"), ok(""), ok(""), ok(""), ok(drifted)])
        with caplog.at_level("INFO"):
            code = runners.main(["--spec", _roster(tmp_path), "--host", "loki", "--rebuild", "yes"])
        assert code == 1
        assert caplog.records[-1].getMessage() == (
            "1 fault(s) after the rebuild; the drift lines name the repairs"
        )


class TestTheLinuxOnlyModes:
    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--render", "out"), ("--sample", "load.jsonl"), ("--onboard", "wagner-austin/x")],
    )
    def test_a_linux_only_mode_refuses_a_windows_host_by_name(
        self, tmp_path: pathlib.Path, flag: str, value: str
    ) -> None:
        with pytest.raises(ValueError) as refused:
            runners.main(["--spec", _roster(tmp_path), "--host", "loki", flag, value])
        assert str(refused.value) == (
            f"{flag} covers Linux runner hosts; loki is a Windows runner host, whose runners "
            "--rebuild yes lays from the roster and the audit reads"
        )

    def test_a_sample_of_every_host_samples_the_linux_ones(self, tmp_path: pathlib.Path) -> None:
        record = tmp_path / "load.jsonl"
        runner = FakeRun([ok("1 0 /sbin/init\n")])
        _test_hooks.run = runner
        assert runners.main(["--spec", _roster(tmp_path), "--sample", str(record)]) == 0
        assert len(runner.calls) == 1
        assert "lavender" in runner.calls[0]
