"""The drain and the resume of a runner host, against fakes at the command seam.

MCPs board task f19d66a5. Every ``gh`` call and every ssh command is answered through the same
``_test_hooks.run`` seam production uses, and the clock moves only when the
code sleeps, so these assert the order of what reaches GitHub and the host:
labels removed before any unit stops, no unit stopped while GitHub reads its
runner busy or before the settle has passed since the labels went, and every
stopped unit started and every label restored when the drain times out.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import _test_hooks, runner_drain
from fleet.core.runner_github import CUSTOM_LABELS_JQ
from tests._runner_fixtures import a_host, an_install
from tests.conftest import FakeClock, FakeRun, FakeSleep, ok

#: The fixture host's two units.
_API = "actions.runner.wagner-austin-API.lavender.service"
_MCPS = "actions.runner.wagner-austin-MCPs.lavender.service"

#: Where the clock starts.
_START = 1_000


def _host() -> HostRunnerSpec:
    """lavender with one API install and one MCPs install, each labelled lavender.

    Returns:
        The spec.
    """
    return a_host([an_install("wagner-austin/API"), an_install("wagner-austin/MCPs")])


def _listed(runner_id: int, *, busy: bool, status: str = "online") -> str:
    """One repository's runner list holding its lavender runner.

    Args:
        runner_id: GitHub's id for it.
        busy: Whether GitHub holds a job on it.
        status: ``online`` or ``offline``.

    Returns:
        The list as ``gh`` prints it.
    """
    return f"{runner_id} lavender {status} {'true' if busy else 'false'}\n"


def _clock() -> FakeClock:
    """A clock at :data:`_START` that each sleep moves.

    Returns:
        The clock, already bound with its sleep into the hooks.
    """
    clock = FakeClock(_START)
    _test_hooks.now = clock
    _test_hooks.sleep = FakeSleep(clock)
    return clock


def _checked(api_labels: str = "lavender\n", mcps_labels: str = "lavender\n") -> list[str]:
    """The four answers of the check every act begins with, both runners listed.

    Args:
        api_labels: The API runner's custom labels as ``gh`` prints them.
        mcps_labels: The MCPs runner's.

    Returns:
        The two lists, then the two label reads.
    """
    return [_listed(1, busy=False), _listed(2, busy=False), api_labels, mcps_labels]


def _tail(call: tuple[str, ...], words: int) -> tuple[str, ...]:
    """The last words of a recorded argv, past ssh's own.

    Args:
        call: A recorded argv.
        words: How many to keep.

    Returns:
        Those words.
    """
    return tuple(call[-words:])


def _stop(service: str) -> tuple[str, ...]:
    """The stop a drain sends for one unit.

    Args:
        service: The unit.

    Returns:
        The remote words.
    """
    return ("sudo", "-n", "systemctl", "stop", service)


def _show(service: str) -> tuple[str, ...]:
    """The state read that follows every start and stop.

    Args:
        service: The unit.

    Returns:
        The remote words.
    """
    return ("systemctl", "show", "-p", "ActiveState", "--value", service)


class TestDrain:
    def test_labels_go_first_and_each_unit_stops_after_the_settle(self) -> None:
        _clock()
        idle = [_listed(1, busy=False), _listed(2, busy=False)]
        answers = [*_checked(), "", "", *idle, *idle, *idle, "", "inactive\n", "", "inactive\n"]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        report = runner_drain.drain(_host())
        assert [call[2:5] for call in runner.calls[4:6]] == [
            ("-X", "DELETE", "repos/wagner-austin/API/actions/runners/1/labels"),
            ("-X", "DELETE", "repos/wagner-austin/MCPs/actions/runners/2/labels"),
        ]
        assert runner.calls[4][-1] == CUSTOM_LABELS_JQ
        assert [_tail(call, 5) for call in runner.calls[12:16:2]] == [_stop(_API), _stop(_MCPS)]
        assert [_tail(call, 6) for call in runner.calls[13:17:2]] == [_show(_API), _show(_MCPS)]
        assert len(runner.calls) == len(answers)
        assert _test_hooks.now() == _START + runner_drain.DRAIN_SETTLE_SECONDS
        assert report["steps"] == [
            "labels: removed from 2 runner(s); GitHub assigns them nothing new",
            f"stopped: {_API}, idle for 30s with no label",
            f"stopped: {_MCPS}, idle for 30s with no label",
        ]

    def test_a_busy_reading_restarts_the_settle_and_nothing_stops_while_busy(self) -> None:
        _clock()
        spec = a_host([an_install("wagner-austin/MCPs")])
        busy, idle = _listed(2, busy=True), _listed(2, busy=False)
        readings = [idle, busy, idle, idle, idle]
        answers = [_listed(2, busy=False), "lavender\n", "", *readings, "", "inactive\n"]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        runner_drain.drain(spec)
        listing = "repos/wagner-austin/MCPs/actions/runners"
        assert all(call[3] == listing for call in runner.calls[3:8])
        assert _tail(runner.calls[8], 5) == _stop(_MCPS)
        assert _test_hooks.now() == _START + 4 * runner_drain.DRAIN_POLL_SECONDS
        assert len(runner.calls) == len(answers)

    def test_a_runner_still_busy_at_the_deadline_puts_the_host_back(self) -> None:
        _clock()
        spec = _host()
        spec["job_timeout_minutes"] = 1
        both = [_listed(1, busy=False), _listed(2, busy=True)]
        polls = (60 + runner_drain.DRAIN_MARGIN_SECONDS) // runner_drain.DRAIN_POLL_SECONDS + 1
        mcps_only = [_listed(2, busy=True)] * (polls - 3)
        answers = [
            *_checked(),
            "",
            "",
            *both,
            *both,
            *both,
            "",
            "inactive\n",
            *mcps_only,
            "",
            "active\n",
            "lavender\n",
            "lavender\n",
        ]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        with pytest.raises(AppError) as fault:
            runner_drain.drain(spec)
        assert fault.value.code is FleetErrorCode.RUNNER_DRAIN_TIMEOUT
        assert fault.value.message == (
            f"lavender: {_MCPS} still held a job 360s after the labels went, past the host's 1 "
            "minute job timeout; the 1 stopped unit(s) were started again and every runner's "
            "labels restored, so the host serves as it did"
        )
        assert len(runner.calls) == len(answers)
        restore = runner.calls[-4:]
        assert _tail(restore[0], 5) == ("sudo", "-n", "systemctl", "start", _API)
        assert _tail(restore[1], 6) == _show(_API)
        assert [call[2:5] for call in restore[2:]] == [
            ("-X", "PUT", "repos/wagner-austin/API/actions/runners/1/labels"),
            ("-X", "PUT", "repos/wagner-austin/MCPs/actions/runners/2/labels"),
        ]
        assert restore[2][5:7] == ("-f", "labels[]=lavender")

    def test_labels_other_than_the_roster_s_change_nothing(self) -> None:
        _clock()
        runner = FakeRun([ok(text) for text in _checked(mcps_labels="lavender\ngpu\n")])
        _test_hooks.run = runner
        with pytest.raises(AppError) as fault:
            runner_drain.drain(_host())
        assert fault.value.code is FleetErrorCode.RUNNER_LABELS_DRIFTED
        assert fault.value.message == (
            "lavender: runner lavender of wagner-austin/MCPs answers to custom labels "
            "['lavender', 'gpu'], and the roster declares ['lavender']; nothing was changed, "
            "since removing and restoring the roster's would rewrite a pool the roster does "
            "not describe"
        )
        assert len(runner.calls) == 4

    def test_a_drain_refuses_a_runner_left_with_no_label(self) -> None:
        _clock()
        runner = FakeRun([ok(text) for text in _checked(api_labels="")][:3])
        _test_hooks.run = runner
        with pytest.raises(AppError) as fault:
            runner_drain.drain(_host())
        assert fault.value.code is FleetErrorCode.RUNNER_LABELS_DRIFTED
        assert "answers to custom labels []" in fault.value.message
        assert len(runner.calls) == 3

    def test_an_unregistered_install_is_stopped_at_once_if_it_runs(self) -> None:
        _clock()
        idle = _listed(1, busy=False)
        answers = [
            idle,
            "",
            "lavender\n",
            "",
            "active\n",
            "",
            "inactive\n",
            idle,
            idle,
            idle,
            "",
            "inactive\n",
        ]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        report = runner_drain.drain(_host())
        assert _tail(runner.calls[4], 6) == _show(_MCPS)
        assert _tail(runner.calls[5], 5) == _stop(_MCPS)
        assert report["steps"] == [
            "labels: removed from 1 runner(s); GitHub assigns them nothing new",
            f"stopped: {_MCPS}, which GitHub has not registered",
            f"stopped: {_API}, idle for 30s with no label",
        ]
        assert len(runner.calls) == len(answers)

    def test_an_unregistered_install_already_stopped_is_left(self) -> None:
        _clock()
        idle = _listed(1, busy=False)
        answers = [idle, "", "lavender\n", "", "inactive\n", idle, idle, idle, "", "inactive\n"]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        runner_drain.drain(_host())
        assert not any(_tail(call, 5) == _stop(_MCPS) for call in runner.calls)
        assert len(runner.calls) == len(answers)

    def test_a_unit_that_does_not_stop_is_refused_by_name(self) -> None:
        _clock()
        spec = a_host([an_install("wagner-austin/MCPs")])
        idle = _listed(2, busy=False)
        answers = [idle, "lavender\n", "", idle, idle, idle, "", "failed\n"]
        _test_hooks.run = FakeRun([ok(text) for text in answers])
        with pytest.raises(AppError) as fault:
            runner_drain.drain(spec)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert fault.value.message == (
            f"lavender: {_MCPS} reads failed after systemctl stop, not inactive; journalctl -u "
            "<unit> on the host says why"
        )

    def test_a_runner_github_stops_listing_mid_drain_is_refused(self) -> None:
        _clock()
        spec = a_host([an_install("wagner-austin/MCPs")])
        answers = [_listed(2, busy=False), "lavender\n", "", ""]
        _test_hooks.run = FakeRun([ok(text) for text in answers])
        with pytest.raises(AppError) as fault:
            runner_drain.drain(spec)
        assert fault.value.code is FleetErrorCode.RUNNER_LABELS_DRIFTED
        assert fault.value.message == (
            f"lavender: GitHub no longer lists the runner of {_MCPS}, which it listed when "
            "this act began; fleet-runners --resume yes --host lavender puts the host back once "
            "it is registered again"
        )


class TestResume:
    def test_each_stopped_unit_starts_then_labels_return_and_each_reads_online(self) -> None:
        _clock()
        online = [_listed(1, busy=False), _listed(2, busy=False)]
        answers = [
            *_checked(api_labels="", mcps_labels=""),
            "inactive\n",
            "",
            "active\n",
            "active\n",
            "lavender\n",
            "lavender\n",
            *online,
        ]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        report = runner_drain.resume(_host())
        assert _tail(runner.calls[5], 5) == ("sudo", "-n", "systemctl", "start", _API)
        assert [call[2:5] for call in runner.calls[8:10]] == [
            ("-X", "PUT", "repos/wagner-austin/API/actions/runners/1/labels"),
            ("-X", "PUT", "repos/wagner-austin/MCPs/actions/runners/2/labels"),
        ]
        assert len(runner.calls) == len(answers)
        assert report["steps"] == [
            f"started: {_API}",
            "labels: the roster's restored on 2 runner(s)",
            "online: all 2 runner(s), as GitHub reads them",
        ]

    def test_a_host_whose_switch_started_every_unit_starts_none(self) -> None:
        _clock()
        online = [_listed(1, busy=False), _listed(2, busy=False)]
        answers = [*_checked(), "active\n", "active\n", "lavender\n", "lavender\n", *online]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        report = runner_drain.resume(_host())
        assert report["steps"][0] == "started: none, every unit was active"
        assert len(runner.calls) == len(answers)

    def test_an_unregistered_install_is_started_and_named_for_register(self) -> None:
        _clock()
        answers = [
            _listed(1, busy=False),
            "",
            "",
            "active\n",
            "inactive\n",
            "",
            "active\n",
            "lavender\n",
            _listed(1, busy=False),
        ]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        report = runner_drain.resume(_host())
        assert report["steps"] == [
            f"started: {_MCPS}",
            f"unregistered: {_MCPS}; fleet-runners --register yes --host lavender registers "
            "them with the roster's labels",
            "labels: the roster's restored on 1 runner(s)",
            "online: all 1 runner(s), as GitHub reads them",
        ]
        assert len(runner.calls) == len(answers)

    def test_a_runner_still_offline_at_the_deadline_is_named(self) -> None:
        _clock()
        spec = a_host([an_install("wagner-austin/MCPs")])
        polls = runner_drain.RESUME_DEADLINE_SECONDS // runner_drain.RESUME_POLL_SECONDS + 1
        offline = [_listed(2, busy=False, status="offline")] * polls
        answers = [_listed(2, busy=False), "", "active\n", "lavender\n", *offline]
        runner = FakeRun([ok(text) for text in answers])
        _test_hooks.run = runner
        with pytest.raises(AppError) as fault:
            runner_drain.resume(spec)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert fault.value.message == (
            f"lavender: GitHub reads {_MCPS} offline 300s after their labels were restored; "
            "journalctl -u <unit> on the host says why"
        )
        assert len(runner.calls) == len(answers)

    def test_labels_other_than_the_roster_s_or_none_change_nothing(self) -> None:
        _clock()
        runner = FakeRun([ok(text) for text in _checked(api_labels="stale\n")][:3])
        _test_hooks.run = runner
        with pytest.raises(AppError) as fault:
            runner_drain.resume(_host())
        assert fault.value.code is FleetErrorCode.RUNNER_LABELS_DRIFTED
        assert "answers to custom labels ['stale']" in fault.value.message
        assert len(runner.calls) == 3
