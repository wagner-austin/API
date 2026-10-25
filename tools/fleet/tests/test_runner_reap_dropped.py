"""The hub's reap pass and its read of the rolled roster (MCPs board task 715b9ec5, A2).

The fakes answer through ``_test_hooks.run``, ``now`` and ``sleep`` exactly
as :mod:`tests.test_core_io` describes. A host's reading is one ``gh``
runner list per repository, then the job scope helper's ``--workers`` lines;
a reap is one ssh running the reaper's ``--dropped`` mode.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str

from fleet.cli import hub_passes
from fleet.contracts.runners import RunnerSpec, encode_runner_spec
from fleet.core import _test_hooks, declared, rolled, runner_reap_dropped
from fleet.core.runner_dropped_check import HostWorker
from fleet.core.runner_reaper_render import REAPER_PATH
from tests.conftest import FakeClock, FakeRun, FakeSleep, failed, ok
from tests.test_runner_dropped_check import API_BUSY, INSTALLS, X_BUSY, _answer, _host, _worker

#: GitHub reading API's first runner offline, every other runner online and busy.
API_ONE_OFFLINE = "1 lavender offline false\n2 lavender-2 online true\n3 lavender-3 online true\n"

#: The Worker the offline runner holds, past the grace.
HELD = _worker(41, 6300)

#: The Worker's description in the pass's lines.
NAMED = "wagner-austin/API lavender Runner.Worker 41 (offline busy=false, 6300 s)"

#: The unit holding it.
HELD_UNIT = INSTALLS[0]["service"]


def _roster() -> RunnerSpec:
    """lavender with its four runners.

    Returns:
        The roster.
    """
    return RunnerSpec(hosts=[_host()])


def _reading(answer: str, github: str = API_ONE_OFFLINE) -> list[_test_hooks.CommandResult]:
    """One reading of lavender: the two ``gh`` lists, then the helper's answer.

    Args:
        answer: The helper's ``--workers`` lines.
        github: GitHub's reading of API's runners.

    Returns:
        The three replies.
    """
    return [ok(github), ok(X_BUSY), ok(answer)]


#: lavender's helper answer with the dropped Worker, the rest idle.
WITH_HELD = _answer((HELD,), (), (), ())


def _sleeper() -> FakeSleep:
    """Bind a fake clock and sleep, so the pass's wait passes no time.

    Returns:
        The sleep, which records what was waited.
    """
    clock = FakeClock(1_000)
    sleeper = FakeSleep(clock)
    _test_hooks.now = clock
    _test_hooks.sleep = sleeper
    return sleeper


class TestThePass:
    def test_a_fleet_holding_nothing_is_read_once_and_said(self) -> None:
        sleeper = _sleeper()
        runner = FakeRun(_reading(_answer((), (), (), ()), API_BUSY))
        _test_hooks.run = runner
        assert runner_reap_dropped.reap_pass(_roster()) == [
            "lavender: examined 4 runner(s); none holds a job GitHub dropped"
        ]
        assert sleeper.slept == []
        assert len(runner.calls) == 3

    def test_a_worker_both_readings_name_is_ended_through_the_reaper(self) -> None:
        sleeper = _sleeper()
        reaped = (
            f"fleet-runner-reaper: {HELD_UNIT}: ending pid 41: Runner.Worker\n"
            f"fleet-runner-reaper: {HELD_UNIT}: ended Runner.Worker 41 and its tree\n"
        )
        runner = FakeRun([*_reading(WITH_HELD), *_reading(WITH_HELD), ok(reaped)])
        _test_hooks.run = runner
        assert runner_reap_dropped.reap_pass(_roster()) == [
            "lavender: 1 Runner.Worker(s) read as holding a job GitHub dropped, read again in "
            f"120 s: {NAMED}",
            f"lavender: ending {NAMED}",
            f"lavender: fleet-runner-reaper: {HELD_UNIT}: ending pid 41: Runner.Worker",
            f"lavender: fleet-runner-reaper: {HELD_UNIT}: ended Runner.Worker 41 and its tree",
        ]
        assert sleeper.slept == [runner_reap_dropped.CONFIRM_SECONDS]
        assert tuple(runner.calls[-1][-8:]) == (
            "lavender",
            "sudo",
            "-n",
            REAPER_PATH,
            "--dropped",
            HELD_UNIT,
            "41",
            str(HELD["start"]),
        )

    @pytest.mark.parametrize(
        "again",
        [
            _answer((), (), (), ()),
            _answer((HostWorker(pid=41, start=9, age=6420),), (), (), ()),
        ],
        ids=["ended", "pid-reused"],
    )
    def test_a_worker_the_second_reading_does_not_name_is_left(self, again: str) -> None:
        _sleeper()
        runner = FakeRun([*_reading(WITH_HELD), *_reading(again)])
        _test_hooks.run = runner
        lines = runner_reap_dropped.reap_pass(_roster())
        assert lines[1:] == ["lavender: 1 Runner.Worker(s) no longer read as dropped, left"]
        assert len(runner.calls) == 6

    def test_a_worker_github_reads_busy_again_is_left(self) -> None:
        """A listener's reconnect: offline for the first reading, back with
        its job for the second."""
        _sleeper()
        _test_hooks.run = FakeRun([*_reading(WITH_HELD), *_reading(WITH_HELD, API_BUSY)])
        assert runner_reap_dropped.reap_pass(_roster())[1:] == [
            "lavender: 1 Runner.Worker(s) no longer read as dropped, left"
        ]

    def test_a_host_that_does_not_answer_either_reading_is_a_line(self) -> None:
        _sleeper()
        _test_hooks.run = FakeRun([ok(API_BUSY), ok(X_BUSY), failed(255, "No route to host")])
        lines = runner_reap_dropped.reap_pass(_roster())
        assert len(lines) == 1
        assert lines[0].startswith("lavender: the dropped-job reading gave nothing: ssh to ")
        assert lines[0].endswith("No route to host")
        _test_hooks.run = FakeRun(
            [*_reading(WITH_HELD), ok(API_ONE_OFFLINE), ok(X_BUSY), failed(255, "Connection reset")]
        )
        lines = runner_reap_dropped.reap_pass(_roster())
        assert len(lines) == 2
        assert lines[1].endswith("Connection reset")

    def test_a_reaper_that_could_not_be_run_is_a_line(self) -> None:
        _sleeper()
        _test_hooks.run = FakeRun(
            [
                *_reading(WITH_HELD),
                *_reading(WITH_HELD),
                failed(1, "FLEET_RUNNER_REAPER_REFUSED: pid 41 is no Runner.Worker"),
            ]
        )
        lines = runner_reap_dropped.reap_pass(_roster())
        assert lines[1] == f"lavender: ending {NAMED}"
        assert lines[2].startswith(
            f"lavender: the reaper did not end it: running `sudo -n {REAPER_PATH} --dropped "
        )
        assert lines[2].endswith("FLEET_RUNNER_REAPER_REFUSED: pid 41 is no Runner.Worker")


#: A commit the fake ref resolves to.
COMMIT = "c" * 40


def _blob(text: str) -> str:
    """``git cat-file --batch``'s answer for one blob.

    Args:
        text: Its content.

    Returns:
        The header, the content and the newline after it.
    """
    return f"{'b' * 40} blob {len(text.encode())}\n{text}\n"


class TestTheRolledRoster:
    def test_the_roster_is_read_at_the_rolled_commit(self, tmp_path: pathlib.Path) -> None:
        roster = RunnerSpec(hosts=[_host()])
        runner = FakeRun([ok(f"{COMMIT}\n"), ok(_blob(dump_json_str(encode_runner_spec(roster))))])
        _test_hooks.run = runner
        assert runner_reap_dropped.read_rolled_roster(tmp_path) == roster
        assert runner.calls[1] == ("git", "-C", str(tmp_path), "cat-file", "--batch")
        assert runner.stdin[1] == f"{COMMIT}:tools/fleet/runners.json\n".encode()

    def test_an_unrolled_checkout_is_a_refusal(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([failed(128, "fatal: Needed a single revision")])
        assert runner_reap_dropped.read_rolled_roster(tmp_path) == (
            f"{rolled.REF_UNRESOLVED_CODE}: {rolled.ROLLED_REF} in {tmp_path} did not resolve to "
            "a commit (exit 128, stdout '', stderr 'fatal: Needed a single revision'); "
            "nothing ran"
        )

    def test_a_failed_read_is_a_refusal(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(f"{COMMIT}\n"), failed(128, "fatal: bad object")])
        assert runner_reap_dropped.read_rolled_roster(tmp_path) == (
            f"{runner_reap_dropped.ROSTER_READ_FAILED_CODE}: git cat-file --batch {COMMIT} "
            "exited 128: fatal: bad object"
        )

    def test_a_roster_missing_at_the_commit_is_a_refusal(self, tmp_path: pathlib.Path) -> None:
        missing = f"{COMMIT}:tools/fleet/runners.json missing"
        _test_hooks.run = FakeRun([ok(f"{COMMIT}\n"), ok(f"{missing}\n")])
        assert runner_reap_dropped.read_rolled_roster(tmp_path) == (
            f"{declared.BATCH_UNREADABLE_CODE}: git cat-file --batch answered {missing!r} for "
            "tools/fleet/runners.json, not a blob header"
        )

    def test_the_hub_pass_logs_each_line_of_the_rolled_roster_s_pass(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        roster = dump_json_str(encode_runner_spec(_roster()))
        _test_hooks.run = FakeRun(
            [ok(f"{COMMIT}\n"), ok(_blob(roster)), *_reading(_answer((), (), (), ()), API_BUSY)]
        )
        with caplog.at_level("INFO"):
            hub_passes.reap_pass(tmp_path)
        assert [record.getMessage() for record in caplog.records] == [
            "lavender: examined 4 runner(s); none holds a job GitHub dropped"
        ]

    def test_a_roster_that_does_not_decode_is_raised(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(f"{COMMIT}\n"), ok(_blob('{"hosts": []}'))])
        with pytest.raises(AppError) as fault:
            runner_reap_dropped.read_rolled_roster(tmp_path)
        assert fault.value.code is FleetErrorCode.RUNNER_SPEC_UNREADABLE
        assert f"{COMMIT}:tools/fleet/runners.json" in fault.value.message
