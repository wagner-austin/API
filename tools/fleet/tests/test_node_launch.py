"""A claim goes on while the job it claimed launches (MCPs board task 8993c306).

On 2026-10-05 lavender-wsl launched another session's tools/hpc3 job from
08:14:44Z to 08:15:45Z, and row 4fb00b3c, submitted at 08:14:52.8Z, was
claimed 64.7 s after it was submitted, because the fill pass launched each
job before it claimed again. The first case binds the real launch pool and
holds the claimed job's launch at its first command: the fill pass claims,
probes again and ends while the launch is still held, the gate charging the
claimed job's grant to the node and leaving its project out, and once the
launch is let go its run is launched, reported started and handed to the
watch. The others show a launch that raises is raised by the next fill pass
and by the serve's end, a serve's own error is not replaced by a launch's,
and the collect pass leaves a claimed job to the launch that carries it.
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str
from platform_core.mcp_client import McpHttpResponse

from fleet.cli import _config, node_agent, node_collect, node_watch
from fleet.cli.node_launch import Launcher, Launching
from fleet.cli.node_prepare import Admitted
from fleet.contracts.dispatch import decode_job
from fleet.contracts.node import LiveLoad
from fleet.contracts.workspace import require_project
from fleet.core import _test_hooks, queue, records, staging
from fleet.core.queue_transport import answering
from tests._holder_fakes import RecordingHolder
from tests._node_agent_fixtures import (
    PROBED,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    launch_steps,
    prebuilt_export,
)
from tests._queue_fakes import (
    DEFAULT_JOB_ID,
    DEFAULT_SHA,
    FakeQueue,
    ToolRefusal,
    Unanswered,
    queue_job,
)
from tests._thread_fakes import await_event
from tests.conftest import DEMO_RUN_ID, FakeRun, failed

__all__ = ["_credentials_in_env", "_sourced_config"]

#: This node's runner.
LAVENDER = "fleet-node-lavender"

#: The prefix of a launch thread's name (:class:`fleet.cli.node_launch.Launcher`).
LAUNCH_THREAD = "fleet-launch-"

#: What the launcher holds with no launch under way.
NOTHING_LAUNCHING = Launching(
    load=LiveLoad(runs=0, workers=0, ram_gb=0.0), projects=frozenset(), jobs=frozenset()
)


def _on_a_launch_thread() -> bool:
    """Whether the caller runs on one of the launcher's threads.

    Returns:
        True on a launch thread.
    """
    return threading.current_thread().name.startswith(LAUNCH_THREAD)


class ThreadRoutedRun:
    """The node and the hub's git, answering the claim and the launch from scripts of their own.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`. The launch's first
    command waits for the case's go-ahead, so the launch is under way for as
    long as the case holds it.

    Attributes:
        claims: The claim's answers, the probes.
        launches: The launch's answers.
    """

    claims: FakeRun
    launches: FakeRun

    def __init__(
        self,
        *,
        claims: FakeRun,
        launches: FakeRun,
        release: threading.Event,
    ) -> None:
        """Bind the scripts and the go-ahead.

        Args:
            claims: The claim's answers.
            launches: The launch's answers.
            release: Set when the launch may go on.
        """
        self.claims = claims
        self.launches = launches
        self._release = release

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Answer from the caller's thread's script.

        Args:
            argv: The command.
            timeout_seconds: The deadline the caller chose.
            stdin_bytes: Its standard input, or None.
            unset_env: The variables the caller withheld from the child.
            set_env: The variables the caller set in the child.

        Returns:
            The next scripted result for that thread.
        """
        script = self.launches if _on_a_launch_thread() else self.claims
        if script is self.launches and not script.calls:
            await_event(self._release, what="the case's go-ahead to launch")
        return script(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )


class ThreadRoutedQueue:
    """The queue, answering the claim and the launch from scripts of their own.

    Satisfies :class:`~platform_core.mcp_client.McpPostProtocol`.
    """

    def __init__(self, *, claims: FakeQueue, launches: FakeQueue) -> None:
        """Bind the scripts.

        Args:
            claims: The claim's answers.
            launches: The launch's answers.
        """
        self._claims = claims
        self._launches = launches

    def __call__(
        self, url: str, *, headers: dict[str, str], body: bytes, timeout_seconds: int
    ) -> McpHttpResponse:
        """Answer from the caller's thread's script.

        Args:
            url: Absolute URL posted to.
            headers: Every request header.
            body: The encoded JSON-RPC body.
            timeout_seconds: The caller's timeout.

        Returns:
            The next scripted answer for that thread.
        """
        script = self._launches if _on_a_launch_thread() else self._claims
        return script(url, headers=headers, body=body, timeout_seconds=timeout_seconds)


def _launcher(
    config_path: pathlib.Path, held: list[frozenset[str]]
) -> tuple[_config.LoadedWorkspace, Launcher]:
    """Lavender's launcher, handing launched runs to a list.

    Args:
        config_path: The workspace document.
        held: Receives each run handed to the watch.

    Returns:
        The workspace and the launcher.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    agent, session_id = node_agent.node_identity("lavender", elevated=False)
    identity = queue.identity_arguments(agent, session_id, str(loaded.directory))
    launcher = Launcher(
        loaded,
        queue.load_credentials(),
        identity,
        alias="lavender",
        node=loaded.workspace["nodes"]["lavender"],
        agent=agent,
        hold=held.append,
    )
    return loaded, launcher


def _fill(loaded: _config.LoadedWorkspace, launcher: Launcher) -> tuple[str, ...]:
    """Run lavender's fill pass.

    Args:
        loaded: The workspace.
        launcher: Its launcher.

    Returns:
        The jobs it claimed and handed to a launch.
    """
    agent, session_id = node_agent.node_identity("lavender", elevated=False)
    return node_agent.fill_pass(
        loaded,
        queue.load_credentials(),
        queue.identity_arguments(agent, session_id, str(loaded.directory)),
        alias="lavender",
        node=loaded.workspace["nodes"]["lavender"],
        elevated=False,
        launcher=launcher,
    )


class TestAClaimWhileItsJobLaunches:
    def test_goes_on_charging_the_node_and_the_run_is_launched_once_let_go(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        digest = staging.digest(prebuilt_export(sourced_config))
        _test_hooks.executor = _test_hooks._default_executor
        release = threading.Event()
        _test_hooks.run = ThreadRoutedRun(
            claims=FakeRun([*PROBED, *PROBED]),
            launches=FakeRun(launch_steps(digest, commit_present=True)),
            release=release,
        )
        _test_hooks.http_post = ThreadRoutedQueue(
            claims=FakeQueue(
                [dump_json_str({"claimed": queue_job(status="claimed", taskId=VERDICT_TASK)})]
            ),
            launches=FakeQueue([dump_json_str({"job": queue_job(status="running")})]),
        )
        held: list[frozenset[str]] = []
        loaded, launcher = _launcher(sourced_config, held)

        with caplog.at_level("INFO"), launcher:
            claimed = _fill(loaded, launcher)
            during = launcher.launching()
            release.set()

        assert claimed == (DEFAULT_JOB_ID,)
        row = records.read_ledger(loaded.ledger)[-1]
        assert row["run_id"] == DEMO_RUN_ID
        assert during == Launching(
            load=LiveLoad(
                runs=1,
                workers=row["workers"],
                ram_gb=row["workers"]
                * require_project(loaded.workspace, "libs/demo")["worker_ram_gb"],
            ),
            projects=frozenset({"libs/demo"}),
            jobs=frozenset({DEFAULT_JOB_ID}),
        )
        assert (
            "lavender leaves out what it is launching now, whose lease is not yet taken: libs/demo"
        ) in [record.getMessage() for record in caplog.records]
        assert held == [frozenset({DEMO_RUN_ID})]
        assert launcher.launching() == NOTHING_LAUNCHING


def _admitted(loaded: _config.LoadedWorkspace) -> Admitted:
    """The demo project's plan, granted one worker.

    Args:
        loaded: The workspace.

    Returns:
        The admission.
    """
    return Admitted(plan=require_project(loaded.workspace, "libs/demo"), workers=1)


class TestALaunchThatRaises:
    def test_is_raised_by_the_next_fill_pass_and_by_the_serves_end(
        self, sourced_config: pathlib.Path
    ) -> None:
        """Each launch's mirror cannot be made and the queue refuses its
        refusal, so both launches raise; the first is the one raised."""
        _test_hooks.run = FakeRun([failed(1, "no disk"), failed(1, "no disk")])
        _test_hooks.http_post = FakeQueue(
            [ToolRefusal("the queue is down"), ToolRefusal("the queue is down again")]
        )
        loaded, launcher = _launcher(sourced_config, [])
        first = decode_job(queue_job(status="claimed"), answer="dispatch_claim")
        second = decode_job(
            queue_job(id="bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb", status="claimed"),
            answer="dispatch_claim",
        )

        launcher.start(first, admitted=_admitted(loaded), sha=DEFAULT_SHA)
        launcher.start(second, admitted=_admitted(loaded), sha=DEFAULT_SHA)

        with pytest.raises(AppError, match=r"the queue is down$"):
            _fill(loaded, launcher)
        with pytest.raises(AppError, match=r"the queue is down$"), launcher:
            assert launcher.launching() == NOTHING_LAUNCHING

    def test_does_not_replace_the_serves_own_error(self, sourced_config: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([failed(1, "no disk")])
        _test_hooks.http_post = FakeQueue([ToolRefusal("the queue is down")])
        loaded, launcher = _launcher(sourced_config, [])
        job = decode_job(queue_job(status="claimed"), answer="dispatch_claim")

        with pytest.raises(LookupError, match="the serve's own"), launcher:
            launcher.start(job, admitted=_admitted(loaded), sha=DEFAULT_SHA)
            raise LookupError("the serve's own")


class TestALaunchWhoseStartReportTheQueueDidNotAnswer:
    def test_leaves_the_run_launched_and_unheld_for_the_collect_pass_and_says_so_once(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Both the start report and the read that would find a cancel go
        unanswered, as every call did while a deploy recreated mcp-fleet
        on 2026-10-05; the run stays launched for
        :func:`fleet.cli.node_collect.reconcile_claim` to adopt, and the
        launcher raises nothing (MCPs board task 8993c306)."""
        digest = staging.digest(prebuilt_export(sourced_config))
        _test_hooks.run = FakeRun(launch_steps(digest, commit_present=True))
        endpoint = FakeQueue([Unanswered(), Unanswered()])
        _test_hooks.http_post = answering(endpoint)
        held: list[frozenset[str]] = []
        loaded, launcher = _launcher(sourced_config, held)
        job = decode_job(queue_job(status="claimed"), answer="dispatch_claim")

        with caplog.at_level("INFO"), launcher:
            launch = launcher.start(job, admitted=_admitted(loaded), sha=DEFAULT_SHA)
        unreported = [launcher.take_unreported(), launcher.take_unreported()]

        assert launch.result() == DEMO_RUN_ID
        assert endpoint.tools == ["dispatch_report", "dispatch_get"]
        assert held == []
        assert unreported == [True, False]
        assert node_watch.live_among(loaded, frozenset({DEMO_RUN_ID})) == {DEMO_RUN_ID}
        assert any(
            record.getMessage().startswith(
                f"{DEFAULT_JOB_ID}: the queue did not answer the start report of "
                f"{DEMO_RUN_ID}; the next collect pass adopts it: "
            )
            for record in caplog.records
        )
        assert launcher.launching() == NOTHING_LAUNCHING


class TestTheCollectPass:
    def test_leaves_a_claimed_job_to_the_launch_that_carries_it(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        credentials = queue.load_credentials()
        claimed = queue_job(status="claimed", claimedBy=LAVENDER)
        endpoint = FakeQueue([dump_json_str({"jobs": [claimed]})])
        _test_hooks.http_post = endpoint
        _test_hooks.run = FakeRun([])
        holder = RecordingHolder()

        with caplog.at_level("INFO"):
            node_collect.collect_pass(
                loaded,
                credentials,
                credentials,
                {},
                agent=LAVENDER,
                alias="lavender",
                holder=holder,
                launching=frozenset({DEFAULT_JOB_ID}),
            )

        assert endpoint.tools == ["dispatch_list"]
        assert holder.held == [frozenset()]
        assert holder.owed == []
        assert caplog.records[-1].getMessage().endswith(": its launch is under way")
