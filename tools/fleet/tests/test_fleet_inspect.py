"""One queued inspection (MCPs board task 3e9c99af): what a job asks, the
command it becomes, and how the command's run closes the job.

The command runs through the package's real run hook twice here, against a
real process standing in for the release's CLI: once answering, once held
past its deadline, so the deadline and the answer are read off a process
and not off a scripted result.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Final

import pytest
from platform_core.json_utils import JSONValue

from fleet.contracts.dispatch import ClosingStatus, DispatchJob, decode_job
from fleet.core import _test_hooks, fleet_inspect
from fleet.core.release_cli import ReleaseCliOutcome, release_cli_argv
from fleet.core.sealed_release import SealedRelease
from tests._queue_fakes import queue_job
from tests.conftest import FakeRun, failed, ok, timed_out

RELEASE: Final = SealedRelease(
    release="5d0e6b1c2a3f4e5d6c7b8a9f0e1d2c3b",
    revision="c3f1e2d4b5a6978869a7b6c5d4e3f2a1b0c9d8e7",
    root=pathlib.Path("/home/corvis/.local/share/corvis/releases/fleet-audit/5d0e/MCPs"),
)

PLACE: Final = fleet_inspect.InspectPlace(
    release=RELEASE["root"],
    nodejs="/usr/bin/node",
    workspace=pathlib.Path("/home/corvis/PROJECTS/MCPs"),
)

TARGET: Final = fleet_inspect.InspectTarget(node="pendragon", probe="services")

#: The line the CLI prints for an answered probe, as fleet-mcp encodes it.
ANSWER: Final = (
    '{"node":"pendragon","hostname":"pendragon","probe":"services",'
    '"startedAt":"2026-11-17T06:00:00.000Z","finishedAt":"2026-11-17T06:00:01.200Z",'
    '"exitStatus":0,"data":{"systemState":"running"},"error":null}'
)

OPENING: Final = (
    "fleet-inspect services of pendragon from sealed release "
    f"{RELEASE['release']} (revision {RELEASE['revision']})"
)


def job(**overrides: JSONValue) -> DispatchJob:
    """A claimed inspect job, decoded from the real wire shape.

    Args:
        **overrides: Fields to vary.

    Returns:
        The job: pendragon's services, no commit (MCPs mig 987).
    """
    fields: dict[str, JSONValue] = {
        "status": "claimed",
        "command": "inspect",
        "project": "probes/services",
        "requestedNode": "pendragon",
        "sha": None,
    }
    fields.update(overrides)
    return decode_job(queue_job(**fields), answer="the test's claimed row")


class TestWhatAJobAsks:
    def test_the_machine_is_the_requested_node_and_the_probe_the_project_s_id(self) -> None:
        assert fleet_inspect.inspect_target(job()) == TARGET

    @pytest.mark.parametrize(
        ("overrides", "named"),
        [
            ({"requestedNode": None}, "None and 'probes/services'"),
            ({"project": "models/services"}, "'pendragon' and 'models/services'"),
        ],
    )
    def test_a_job_without_a_machine_or_a_probe_is_refused_naming_both(
        self, overrides: dict[str, JSONValue], named: str
    ) -> None:
        assert fleet_inspect.inspect_target(job(**overrides)) == (
            "INSPECT_JOB_MALFORMED: an inspect job names its machine as the requested node "
            f"and its probe as probes/<id>; this one names {named}, so nothing ran"
        )


class TestTheCommand:
    def test_the_release_s_own_cli_is_entered_with_its_root_and_handed_two_names(
        self,
    ) -> None:
        assert fleet_inspect.inspect_argv(PLACE, RELEASE, TARGET) == release_cli_argv(
            "/usr/bin/node",
            RELEASE["root"] / "fleet-mcp/dist/src/inspect-cli.js",
            (str(RELEASE["root"]),),
            "--workspace",
            "/home/corvis/PROJECTS/MCPs",
            "--node",
            "pendragon",
            "--probe",
            "services",
        )
        assert fleet_inspect.KIND == "fleet-audit"

    def test_a_run_is_bounded_and_adds_nothing_to_its_environment(self) -> None:
        runner = FakeRun([ok(f"{ANSWER}\n")])
        _test_hooks.run = runner
        argv = fleet_inspect.inspect_argv(PLACE, RELEASE, TARGET)

        assert fleet_inspect.run_inspection(
            argv, timeout_seconds=fleet_inspect.INSPECT_TIMEOUT_SECONDS
        ) == ok(f"{ANSWER}\n")

        assert runner.calls == [argv]
        assert runner.timeouts == [fleet_inspect.INSPECT_TIMEOUT_SECONDS]
        assert fleet_inspect.INSPECT_TIMEOUT_SECONDS == 180
        assert runner.set_env == [()]


class TestTheAnswer:
    def test_one_json_object_on_an_exit_0_run_is_the_inspection_passed_on_unread(
        self,
    ) -> None:
        assert fleet_inspect.answered_inspection(ok(f"{ANSWER}\n")) == {
            "node": "pendragon",
            "hostname": "pendragon",
            "probe": "services",
            "startedAt": "2026-11-17T06:00:00.000Z",
            "finishedAt": "2026-11-17T06:00:01.200Z",
            "exitStatus": 0,
            "data": {"systemState": "running"},
            "error": None,
        }

    @pytest.mark.parametrize(
        "result",
        [
            ok(f"{ANSWER}\n{ANSWER}\n"),
            ok(""),
            ok("not json\n"),
            ok('["an", "array"]\n'),
            failed(1, f"{ANSWER}\n"),
        ],
    )
    def test_any_other_run_answers_nothing(self, result: _test_hooks.CommandResult) -> None:
        assert fleet_inspect.answered_inspection(result) is None

    def test_a_failure_s_stdout_is_never_read_as_an_answer(self) -> None:
        result = _test_hooks.CommandResult(
            returncode=1, stdout=f"{ANSWER}\n", stderr="", timed_out=False
        )

        assert fleet_inspect.answered_inspection(result) is None


class TestTheClose:
    def test_an_exit_0_run_without_one_object_is_refused_naming_its_lines(self) -> None:
        assert fleet_inspect.inspect_outcome(ok("one\ntwo\n"), RELEASE, TARGET) == (
            ReleaseCliOutcome(
                status=ClosingStatus.REFUSED,
                exit_code=None,
                detail=(
                    f"INSPECT_ANSWER_MALFORMED: {OPENING} exited 0 and printed 2 line(s) where "
                    "it prints one JSON object, so this runner reports no inspection"
                ),
            )
        )

    def test_the_cli_s_refusal_is_the_job_s_in_its_own_words(self) -> None:
        refusal = "INSPECT_NODE_CLIENT: s23 is a client, which the fleet never reaches over ssh"

        assert fleet_inspect.inspect_outcome(failed(2, f"{refusal}\n"), RELEASE, TARGET) == (
            ReleaseCliOutcome(
                status=ClosingStatus.REFUSED, exit_code=None, detail=f"{OPENING} refused: {refusal}"
            )
        )
        assert fleet_inspect.CLI_EXIT_REFUSED == 2

    def test_a_run_that_never_answered_fails_with_its_output_s_tail(self) -> None:
        unanswered = "INSPECT_RUN_UNANSWERED: pendragon: copy exited 255: Connection refused"

        assert fleet_inspect.inspect_outcome(failed(1, f"{unanswered}\n"), RELEASE, TARGET) == (
            ReleaseCliOutcome(
                status=ClosingStatus.FAILED,
                exit_code=1,
                detail=f"{OPENING} exited 1: {unanswered}",
            )
        )

    def test_a_run_ended_at_its_deadline_fails(self) -> None:
        result = timed_out(fleet_inspect.INSPECT_TIMEOUT_SECONDS)
        outcome = fleet_inspect.inspect_outcome(result, RELEASE, TARGET)

        assert outcome["status"] is ClosingStatus.FAILED
        assert outcome["exit_code"] == result["returncode"]
        assert outcome["detail"].startswith(f"{OPENING} exited {result['returncode']}: ")


class TestARealProcess:
    def test_a_process_printing_one_object_is_answered(self) -> None:
        script = f"import sys; sys.stdout.write({ANSWER!r} + '\\n')"

        result = fleet_inspect.run_inspection(
            (sys.executable, "-c", script), timeout_seconds=fleet_inspect.INSPECT_TIMEOUT_SECONDS
        )

        assert result["returncode"] == 0
        assert fleet_inspect.answered_inspection(result) == {
            "node": "pendragon",
            "hostname": "pendragon",
            "probe": "services",
            "startedAt": "2026-11-17T06:00:00.000Z",
            "finishedAt": "2026-11-17T06:00:01.200Z",
            "exitStatus": 0,
            "data": {"systemState": "running"},
            "error": None,
        }

    def test_a_process_held_past_its_deadline_is_ended_and_the_job_fails(self) -> None:
        result = fleet_inspect.run_inspection(
            (sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"),
            timeout_seconds=1,
        )

        assert result["timed_out"] is True
        assert fleet_inspect.answered_inspection(result) is None
        outcome = fleet_inspect.inspect_outcome(result, RELEASE, TARGET)
        assert outcome["status"] is ClosingStatus.FAILED
        assert outcome["exit_code"] == result["returncode"]
