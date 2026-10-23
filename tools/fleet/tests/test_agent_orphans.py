"""A tick gives back each held job whose run is gone, before it claims (MCPs task 05dd6588).

On diphtheria on 2026-10-21 the hub runner's ship restarted it 22 s into job
e2f2a964's netboot arm; the arm's unit stopped with it, and the row read
running under a lease an hour long while every later tick found the lane
empty. Here the claim pass meets that row: the queue holds a job under this
runner's label whose run id names a unit the manager reads as stopped, so
the job is reported lost before the claim, which then takes it back; a run
still live, one on another machine, and one whose id names no unit are left
alone. The systemctl answers are the shapes measured there that day, which
``test_tick_run_hub.py`` reads from the hub's own manager.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, dump_json_str

from fleet.cli import agent
from fleet.core import _test_hooks, tick_run
from tests._agent_cli import HUB_HOST, agent_argv, bind_hub_tick, tick_run_id
from tests._hub_tick import UNROLLED_TICK, unrolled_calls
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, queue_job
from tests.conftest import FakeRun, failed, ok

#: The unit the earlier tick's run was, which the restart stopped.
STOPPED_UNIT = "session-audit-job-df261b2fda99482cb3d130e721072c91.service"

#: That run's id, as the earlier tick reported its job started.
STOPPED_RUN_ID = f"{STOPPED_UNIT} on {HUB_HOST}"

#: The stopped unit as the manager reads it once collected.
GONE = ok("ActiveState=inactive\nControlGroup=\n")

#: What the claim pass reports for it.
LOST_DETAIL = (
    f"run {STOPPED_RUN_ID} is gone: systemctl --user show read ActiveState=inactive "
    "ControlGroup=, so the make build-bases it ran ended without closing the job"
)


@pytest.fixture(name="hub_tick", autouse=True)
def _hub_tick() -> None:
    """The queue's variables, and the machine and unit the tick runs in."""
    bind_hub_tick()


def held(**overrides: JSONValue) -> str:
    """The queue's answer listing one job this runner holds, a running rebuild.

    Args:
        **overrides: Fields to vary on the row.

    Returns:
        The rendered ``dispatch_list`` answer.
    """
    fields: dict[str, JSONValue] = {
        "status": "running",
        "command": "build-bases",
        "project": "MCPs",
        "sha": None,
        "claimedBy": f"fleet-hub-{HUB_HOST}",
        "node": HUB_HOST,
        "runId": STOPPED_RUN_ID,
    }
    fields.update(overrides)
    return dump_json_str({"jobs": [queue_job(**fields)]})


def lost_answer(status: str) -> str:
    """The queue's answer to the lost report.

    Args:
        status: What the queue left the job: ``queued``, or ``failed``.

    Returns:
        The rendered ``dispatch_report`` answer.
    """
    return dump_json_str(
        {"job": queue_job(status=status, command="build-bases", project="MCPs", sha=None)}
    )


def tick(repo: pathlib.Path, config_path: pathlib.Path, argv: list[str] | None = None) -> int:
    """Run one hub tick.

    Args:
        repo: The monorepo root.
        config_path: The workspace document.
        argv: The arguments, the hub tick's when None.

    Returns:
        Its exit code.
    """
    return agent.main(argv if argv is not None else agent_argv(config_path, repo))


class TestAStoppedRunIsGivenBack:
    def test_it_is_reported_lost_before_the_claim_naming_the_stopped_run(
        self, config_path: pathlib.Path, repo: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run = FakeRun([GONE, *UNROLLED_TICK])
        _test_hooks.run = run
        endpoint = FakeQueue([held(), lost_answer("queued"), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert tick(repo, config_path) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        assert endpoint.arguments[0]["claimedBy"] == f"fleet-hub-{HUB_HOST}"
        reported = endpoint.arguments[1]
        assert reported["action"] == "lost"
        assert reported["jobId"] == DEFAULT_JOB_ID
        assert reported["detail"] == LOST_DETAIL
        assert reported["agent"] == f"fleet-hub-{HUB_HOST}"
        assert run.calls == [
            (*tick_run.SYSTEMCTL, *tick_run.SHOWN, STOPPED_UNIT),
            *unrolled_calls(repo),
        ]
        assert (
            f"reported lost: {LOST_DETAIL}; the queue left it {DEFAULT_JOB_ID} queued "
            "make build-bases MCPs @any node"
        ) in [record.getMessage() for record in caplog.records]

    def test_the_claim_that_follows_takes_the_job_back_as_this_ticks_run(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        _test_hooks.run = FakeRun([GONE, ok("Base images rebuilt.\n"), *UNROLLED_TICK])
        claimed = queue_job(status="claimed", command="build-bases", project="MCPs", sha=None)
        endpoint = FakeQueue(
            [
                held(),
                lost_answer("queued"),
                dump_json_str({"claimed": claimed}),
                dump_json_str({"job": queue_job(status="running", node=HUB_HOST)}),
                dump_json_str({"job": queue_job(status="passed", node=HUB_HOST)}),
            ]
        )
        _test_hooks.http_post = endpoint

        argv = [*agent_argv(config_path, repo), agent.MCPS_ROOT_FLAG, str(mcps)]
        assert tick(repo, config_path, argv) == 0

        assert endpoint.tools == [
            "dispatch_list",
            "dispatch_report",
            "dispatch_claim",
            "dispatch_report",
            "dispatch_report",
        ]
        assert endpoint.arguments[1]["action"] == "lost"
        started = endpoint.arguments[3]
        assert started["action"] == "start"
        assert started["runId"] == tick_run_id()
        assert endpoint.arguments[4]["status"] == "passed"

    def test_a_job_the_queue_had_requeued_before_comes_back_failed(
        self, config_path: pathlib.Path, repo: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The second loss is the queue's to judge (MCPs reportDispatchRunLost):
        the tick reports it the same way and logs what the queue made of it."""
        _test_hooks.run = FakeRun([GONE, *UNROLLED_TICK])
        endpoint = FakeQueue([held(), lost_answer("failed"), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert tick(repo, config_path) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        assert (
            f"reported lost: {LOST_DETAIL}; the queue left it {DEFAULT_JOB_ID} failed "
            "make build-bases MCPs @any node"
        ) in [record.getMessage() for record in caplog.records]


class TestARunNotProvenGoneIsLeftToItsLease:
    def test_a_live_run_is_another_tick_working_and_is_not_reported(
        self, config_path: pathlib.Path, repo: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        live = ok(f"ActiveState=active\nControlGroup=/user.slice/app.slice/{STOPPED_UNIT}\n")
        _test_hooks.run = FakeRun([live, *UNROLLED_TICK])
        endpoint = FakeQueue([held(), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert tick(repo, config_path) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        assert (
            f"job {DEFAULT_JOB_ID} runs as {STOPPED_UNIT}, which reads ActiveState=active "
            f"ControlGroup=/user.slice/app.slice/{STOPPED_UNIT}: live"
        ) in [record.getMessage() for record in caplog.records]

    @pytest.mark.parametrize(
        ("overrides", "logged"),
        [
            (
                {"runId": f"{STOPPED_UNIT} on demure"},
                f"job {DEFAULT_JOB_ID} runs as {STOPPED_UNIT}, on demure rather than this "
                f"machine, {HUB_HOST}, so it is left to its lease",
            ),
            (
                {"runId": f"bases-{DEFAULT_JOB_ID}"},
                f"job {DEFAULT_JOB_ID} is running under this label with run id "
                f"'bases-{DEFAULT_JOB_ID}', which names no unit, so it is left to its lease",
            ),
            (
                {"status": "claimed", "runId": "", "node": None},
                f"job {DEFAULT_JOB_ID} is claimed under this label with run id '', which names "
                "no unit, so it is left to its lease",
            ),
        ],
    )
    def test_a_run_this_machine_cannot_ask_about_is_not_reported(
        self,
        config_path: pathlib.Path,
        repo: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
        overrides: dict[str, JSONValue],
        logged: str,
    ) -> None:
        run = FakeRun([*UNROLLED_TICK])
        _test_hooks.run = run
        endpoint = FakeQueue([held(**overrides), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert tick(repo, config_path) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        assert run.calls == [*unrolled_calls(repo)]
        assert logged in [record.getMessage() for record in caplog.records]

    def test_a_manager_that_cannot_be_asked_stops_the_tick_before_its_claim(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun([failed(1, "Failed to connect to bus: No medium found")])
        endpoint = FakeQueue([held()])
        _test_hooks.http_post = endpoint

        with pytest.raises(AppError, match=f"{tick_run.UNREADABLE_CODE}: "):
            tick(repo, config_path)

        assert endpoint.tools == ["dispatch_list"]


class TestATickInNoUnitClaimsNothing:
    def test_it_is_refused_before_the_queue_is_asked(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        def in_a_login_scope(path: pathlib.Path) -> str:
            if path == tick_run.PROC_SELF_CGROUP:
                return "0::/user.slice/user-1000.slice/session-3.scope\n"
            return _test_hooks._default_read_text(path)

        _test_hooks.read_text = in_a_login_scope
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(AppError, match=f"{tick_run.UNBOUND_CODE}: "):
            tick(repo, config_path)

        assert endpoint.tools == []
