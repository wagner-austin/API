"""Registrations held to the roster: the reading, the audit row and the prune.

GitHub answers through ``_test_hooks.run`` exactly as :mod:`tests.test_core_io`
describes, so each case's argv is the one production would run.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import RunnerSpec
from fleet.core import _test_hooks, runner_registrations
from fleet.core.runner_github import GitHubRunner
from tests._runner_fixtures import a_host, an_install
from tests.conftest import FakeRun, failed, ok


def _roster() -> RunnerSpec:
    """Two hosts, lavender serving API and MCPs, sedona serving API.

    Returns:
        The roster.
    """
    lavender = a_host()
    lavender["installs"] = [an_install("wagner-austin/API"), an_install("wagner-austin/MCPs")]
    sedona = a_host()
    sedona["name"] = "sedona"
    sedona_install = an_install("wagner-austin/API")
    sedona_install["runner_name"] = "sedona"
    sedona["installs"] = [sedona_install]
    return RunnerSpec(hosts=[lavender, sedona], windows_hosts=[])


class TestTheReading:
    def test_every_host_s_installs_are_declared_by_repository(self) -> None:
        assert runner_registrations.declared_names(_roster()) == {
            "wagner-austin/API": {"lavender", "sedona"},
            "wagner-austin/MCPs": {"lavender"},
        }

    def test_a_registration_another_host_declares_is_not_undeclared(self) -> None:
        runner = FakeRun([ok("1 lavender online false\n2 sedona offline false\n")])
        _test_hooks.run = runner
        readings = runner_registrations.read_registrations(_roster(), ["wagner-austin/API"])
        assert readings == [
            runner_registrations.RegistrationReading(
                repo="wagner-austin/API", examined=2, declared=2, undeclared=[]
            )
        ]
        assert runner.calls[0][3] == "repos/wagner-austin/API/actions/runners"

    def test_an_undeclared_registration_is_read_with_github_s_reading(self) -> None:
        _test_hooks.run = FakeRun([ok("1 lavender online false\n23 lavender-wsl offline true\n")])
        readings = runner_registrations.read_registrations(_roster(), ["wagner-austin/MCPs"])
        assert readings[0]["undeclared"] == [
            runner_registrations.Registration(
                repo="wagner-austin/MCPs",
                name="lavender-wsl",
                github=GitHubRunner(id=23, status="offline", busy=True),
            )
        ]


class TestTheRow:
    def test_a_repository_whose_every_registration_is_declared_reads_ok(self) -> None:
        finding = runner_registrations.registration_finding(
            runner_registrations.RegistrationReading(
                repo="wagner-austin/API", examined=2, declared=2, undeclared=[]
            )
        )
        assert finding == {
            "check_id": "registrations:wagner-austin/API",
            "ok": True,
            "detail": "examined 2 registration(s) of wagner-austin/API against the 2 runner "
            "name(s) the roster declares for it; every one is declared",
            "reason": runner_registrations.REGISTRATIONS_REASON,
        }

    def test_a_repository_with_no_registrations_says_it_examined_none(self) -> None:
        finding = runner_registrations.registration_finding(
            runner_registrations.RegistrationReading(
                repo="wagner-austin/API", examined=0, declared=2, undeclared=[]
            )
        )
        assert finding["detail"].startswith("examined 0 registration(s) of wagner-austin/API")

    def test_each_undeclared_registration_is_named_with_its_state(self) -> None:
        undeclared = [
            runner_registrations.Registration(
                repo="wagner-austin/API",
                name="lavender-wsl",
                github=GitHubRunner(id=21, status="offline", busy=False),
            ),
            runner_registrations.Registration(
                repo="wagner-austin/API",
                name="loki",
                github=GitHubRunner(id=30, status="online", busy=True),
            ),
        ]
        finding = runner_registrations.registration_finding(
            runner_registrations.RegistrationReading(
                repo="wagner-austin/API", examined=4, declared=2, undeclared=undeclared
            )
        )
        assert finding["ok"] is False
        assert finding["detail"] == (
            "examined 4 registration(s) of wagner-austin/API against the 2 runner name(s) the "
            "roster declares for it; 2 no roster host declares: lavender-wsl (id 21, offline, "
            "idle), loki (id 30, online, busy); fleet-runners --prune yes removes the offline "
            "idle ones"
        )


class TestThePrune:
    def test_only_the_offline_idle_undeclared_are_deleted(self) -> None:
        runner = FakeRun(
            [
                ok(
                    "1 lavender online false\n21 lavender-wsl offline false\n"
                    "22 lavender-wsl-2 offline true\n30 loki online false\n"
                ),
                ok("1 lavender online false\n"),
                ok(""),
            ]
        )
        _test_hooks.run = runner
        report = runner_registrations.prune_registrations(_roster())
        assert [each["name"] for each in report["removed"]] == ["lavender-wsl"]
        assert [each["name"] for each in report["kept"]] == ["lavender-wsl-2", "loki"]
        assert report["examined"] == 5
        assert [call[3] for call in runner.calls] == [
            "repos/wagner-austin/API/actions/runners",
            "repos/wagner-austin/MCPs/actions/runners",
            "DELETE",
        ]
        assert runner.calls[2][4] == "repos/wagner-austin/API/actions/runners/21"

    def test_a_refused_deletion_stops_the_prune(self) -> None:
        runner = FakeRun(
            [
                ok("21 lavender-wsl offline false\n"),
                ok("23 lavender-wsl offline false\n"),
                failed(1, "HTTP 403"),
            ]
        )
        _test_hooks.run = runner
        with pytest.raises(AppError) as fault:
            runner_registrations.prune_registrations(_roster())
        assert fault.value.code is FleetErrorCode.RUNNER_DEREGISTRATION_REFUSED
        assert len(runner.calls) == 3
