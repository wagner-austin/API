"""Running a project's host target on its node, for real (MCPs board task 57791f39, criterion A1).

Each case drives ``python -m fleet.cli.host_target``'s own ``main`` over a
real registry, real git and real ``sh``: the remote is a real repository
(git's ``url.<path>.insteadOf`` points the registry's https remote at it, so
the hub's real fetch reads it), the hub's mirror and bundle are real, and
every ssh the stage, the run and the retire send is carried out on this
machine by :func:`tests._on_this_machine.on_this_machine`, so the node's
scripts are the ones a node runs. The remote's Makefile has a target that
passes, one that fails and one that outlives its deadline.

Every expected value is read from outside the code under test: the commit
from the test's own ``git rev-parse``, the lines from the Makefile the case
wrote, the statuses from the Makefile's ``exit`` and GNU ``timeout``'s
documented 124.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
import time
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError

from fleet.cli import _config
from fleet.cli import host_target as host_target_cli
from fleet.contracts.feed import FeedKind
from fleet.contracts.host_target import HostTarget
from fleet.contracts.retire_record import RetireState
from fleet.core import _test_hooks, host_target, leases, names, records, retire_owed, run_lease
from tests._git_fixtures import git
from tests._on_this_machine import on_this_machine
from tests._registry import write_registry
from tests._roster_fixtures import machine_document, machines_of
from tests.conftest import DEMO_NOW, FakeRun, FakeTempRoot, failed, pin_clock, workspace_document

#: The project the cases run, and the remote its registry line declares.
PROJECT = "games/desk"
REMOTE = "https://github.com/example/desk.git"

#: The identity the lease names.
AGENT = "opus-mcps-1127-cd54"
SESSION = "cd544177-8b37-4879-86bd-fc61e1fb0ae3"

#: The run every case's lease names: the project, its node and the pinned
#: second (run_lease.run_id_for), worked by hand.
RUN_ID = f"games-desk-lavender-{DEMO_NOW}"

#: The remote's Makefile at its second commit. ``pass`` prints the workers
#: it was given and a line on standard error; ``fail`` exits 3; ``slow``
#: sleeps past any deadline a case gives it.
MAKEFILE = (
    "pass:\n"
    "\t@printf 'built with %s workers\\n' \"$$CORVIS_TEST_MAX_WORKERS\"\n"
    "\t@cat version.txt\n"
    "\t@echo 'a warning' >&2\n"
    "fail:\n"
    "\t@echo 'about to fail'\n"
    "\t@exit 3\n"
    "slow:\n"
    "\t@echo started\n"
    "\t@sleep 300\n"
    "\t@echo finished\n"
)


def _remote(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A real repository with two commits on its default branch.

    Args:
        tmp_path: The case's directory.

    Returns:
        The repository, and its default branch's head, read by the test's
        own git.
    """
    repo = tmp_path / "remote"
    (repo / "desk").mkdir(parents=True)
    git(repo, "init", "--quiet", "-b", "main")
    (repo / "desk" / "Makefile").write_text(MAKEFILE, encoding="utf-8")
    (repo / "desk" / "version.txt").write_text("version one\n", encoding="utf-8")
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "-m", "one")
    (repo / "desk" / "version.txt").write_text("version two\n", encoding="utf-8")
    git(repo, "commit", "--quiet", "--all", "-m", "two")
    return repo, git(repo, "rev-parse", "HEAD")


def _registry(
    tmp_path: pathlib.Path, *, host_target_entry: JSONObject | None, enabled: bool = True
) -> pathlib.Path:
    """Write a registry whose one project runs on lavender, staged under the case's directory.

    Args:
        tmp_path: The case's directory.
        host_target_entry: The project's ``host_target``, or None to leave
            the key out.
        enabled: Whether lavender is enabled.

    Returns:
        The registry directory, which ``--config`` names.
    """
    document = workspace_document()
    lavender = machine_document(enabled=enabled)
    machines_of(document)["lavender"] = lavender
    dispatch = lavender["dispatch"]
    assert isinstance(dispatch, dict)
    dispatch["stage_root"] = (tmp_path / "stage").as_posix()
    project: JSONObject = {
        "worker_ram_gb": 1.1,
        "minimum_workers": 1,
        "expected_minutes": 5,
        "required_tags": [],
        "source": {"remote": REMOTE, "path": "desk", "install": [], "companions": []},
    }
    if host_target_entry is not None:
        project["host_target"] = host_target_entry
    document["projects"] = {PROJECT: project}
    return write_registry(tmp_path / "registry", document)


def _entry(target: str) -> JSONObject:
    """A host target on lavender with three workers.

    Args:
        target: The make target.

    Returns:
        Its registry object.
    """
    return {"target": target, "node": "lavender", "workers": 3, "deadline_seconds": 60}


def _argv(registry: pathlib.Path) -> list[str]:
    """The command line every case runs.

    Args:
        registry: The registry directory.

    Returns:
        The flags.
    """
    return [
        _config.CONFIG_FLAG,
        str(registry),
        "--project",
        PROJECT,
        "--agent",
        AGENT,
        "--session",
        SESSION,
    ]


#: The make variables this suite inherits from its own ``make check``. A node
#: runs its script in ssh's fresh login, which carries none of them; inherited,
#: ``MAKELEVEL`` makes the target's make print ``make[1]: Entering directory``
#: lines, measured with GNU make on 2026-11-27.
MAKE_VARIABLES = ("MAKELEVEL", "MAKEFLAGS", "MFLAGS")


class HubAndNode:
    """The hub's git against a real repository, and each ssh carried out here as a node's login.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`. The hub's git
    is given ``url.<repository>.insteadOf <REMOTE>`` through git's own
    ``GIT_CONFIG_*`` variables, so its real fetch of the registry's https
    remote reads the case's repository; every ssh and scp runs here without
    :data:`MAKE_VARIABLES`, or, for a node that does not answer, ends as
    ssh ends when it cannot connect.

    Attributes:
        repo: The repository the remote stands for.
        reachable: Whether the node answers.
    """

    repo: pathlib.Path
    reachable: bool

    def __init__(self, repo: pathlib.Path, *, reachable: bool) -> None:
        """Route to one repository and one node.

        Args:
            repo: The repository the remote stands for.
            reachable: Whether the node answers.
        """
        self.repo = repo
        self.reachable = reachable

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Run one command as the hub or the node would.

        Args:
            argv: The command.
            timeout_seconds: Its deadline.
            stdin_bytes: Its standard input.
            unset_env: Variables it withholds.
            set_env: Variables it sets.

        Returns:
            What it did here.
        """
        if argv[0] == "git":
            redirect = (
                ("GIT_CONFIG_COUNT", "1"),
                ("GIT_CONFIG_KEY_0", f"url.{self.repo.as_posix()}.insteadOf"),
                ("GIT_CONFIG_VALUE_0", REMOTE),
            )
            return _test_hooks._default_run(
                argv,
                timeout_seconds=timeout_seconds,
                stdin_bytes=stdin_bytes,
                unset_env=unset_env,
                set_env=(*set_env, *redirect),
            )
        if not self.reachable:
            # ssh's own status and words for a refused connection.
            return failed(255, "ssh: connect to host lavender port 22: Connection refused")
        return on_this_machine(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=(*unset_env, *MAKE_VARIABLES),
            set_env=set_env,
        )


@pytest.fixture(name="hub")
def _hub(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Point the registry's remote at a real repository and carry ssh out here.

    Args:
        tmp_path: The case's directory.

    Returns:
        The repository and its head.
    """
    repo, head = _remote(tmp_path)
    pin_clock(DEMO_NOW)
    _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
    _test_hooks.run = HubAndNode(repo, reachable=True)
    return repo, head


def _kinds(registry: pathlib.Path) -> list[FeedKind]:
    """The feed's kinds for the case's run, in order.

    Args:
        registry: The registry directory, beside whose ``runs`` the feed is.

    Returns:
        Each event's kind.
    """
    feed = registry.parent / "runs" / "feed.jsonl"
    return [event["kind"] for event in records.read_feed(feed) if event["run_id"] == RUN_ID]


class TestARealRun:
    def test_a_passing_target_prints_its_lines_with_the_head_and_exits_zero(
        self,
        tmp_path: pathlib.Path,
        hub: tuple[pathlib.Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, head = hub
        registry = _registry(tmp_path, host_target_entry=_entry("pass"))

        status = host_target_cli.main(_argv(registry))

        printed = capsys.readouterr().out.splitlines()
        assert status == 0
        # Standard error is merged; version two is the head's, not the first commit's.
        assert printed[:3] == [
            f"{head} built with 3 workers",
            f"{head} version two",
            f"{head} a warning",
        ]
        assert printed[3] == (
            f"host target pass of {PROJECT} at {head} on lavender, run {RUN_ID}: "
            "make pass exited 0; retire ended"
        )
        assert len(printed) == 4
        assert _kinds(registry) == [FeedKind.LEASED, FeedKind.STAGED, FeedKind.PASSED]

    def test_a_passing_run_gives_its_lease_back_and_retires_its_tree_keeping_the_transcript(
        self, tmp_path: pathlib.Path, hub: tuple[pathlib.Path, str]
    ) -> None:
        registry = _registry(tmp_path, host_target_entry=_entry("pass"))
        stage = (tmp_path / "stage").as_posix()
        runs = registry.parent / "runs"

        host_target_cli.main(_argv(registry))

        assert leases.read_leases(runs / "leases.json") == ()
        assert not pathlib.Path(names.run_enclosure(stage, RUN_ID)).exists()
        kept = pathlib.Path(names.retained_log_path(stage, RUN_ID)).read_text(encoding="utf-8")
        assert kept == "built with 3 workers\nversion two\na warning\n"
        states = [
            record["state"] for record in records.read_retires(runs / retire_owed.RETIRES_FILE)
        ]
        assert states == [RetireState.OWED, RetireState.RETIRED]

    def test_a_failing_target_exits_with_make_s_status_and_is_retired(
        self,
        tmp_path: pathlib.Path,
        hub: tuple[pathlib.Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, head = hub
        registry = _registry(tmp_path, host_target_entry=_entry("fail"))

        status = host_target_cli.main(_argv(registry))

        printed = capsys.readouterr().out.splitlines()
        # The recipe's exit 3 ends make with its own status 2 and its error
        # line, as GNU make printed them for this Makefile on 2026-11-27.
        assert status == 2
        assert printed == [
            f"{head} about to fail",
            f"{head} make: *** [Makefile:7: fail] Error 3",
            f"host target fail of {PROJECT} at {head} on lavender, run {RUN_ID}: "
            "make fail exited 2; retire ended",
        ]
        assert _kinds(registry) == [FeedKind.LEASED, FeedKind.STAGED, FeedKind.FAILED]
        stage = (tmp_path / "stage").as_posix()
        assert not pathlib.Path(names.run_enclosure(stage, RUN_ID)).exists()

    def test_running_as_a_module_runs_the_target(
        self, tmp_path: pathlib.Path, hub: tuple[pathlib.Path, str]
    ) -> None:
        """The half that goes missing without an ``if __name__`` block.

        The passing target, so the exit 0 is make's and never the boundary's
        refusal status 2.
        """
        registry = _registry(tmp_path, host_target_entry=_entry("pass"))
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.host_target", None)
        sys.argv = ["host_target", *_argv(registry)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.host_target", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.host_target"] = saved_module

        assert raised.value.code == 0
        assert _kinds(registry) == [FeedKind.LEASED, FeedKind.STAGED, FeedKind.PASSED]


class TestTheDeadline:
    def test_a_target_outliving_its_deadline_is_stopped_with_timeout_s_status(
        self, tmp_path: pathlib.Path
    ) -> None:
        stage = (tmp_path / "stage").as_posix()
        target = names.dispatch_directory(stage, RUN_ID)
        (pathlib.Path(target) / "desk").mkdir(parents=True)
        (pathlib.Path(target) / "desk" / "Makefile").write_text(MAKEFILE, encoding="utf-8")
        _test_hooks.run = HubAndNode(tmp_path / "remote", reachable=True)
        node = _config.load_workspace(
            {_config.CONFIG_FLAG: str(_registry(tmp_path, host_target_entry=_entry("slow")))}
        ).workspace["nodes"]["lavender"]
        declared = HostTarget(target="slow", node="lavender", workers=1, deadline_seconds=1)

        began = time.monotonic()
        ran = host_target.run_on_node(
            node, run_id=RUN_ID, target=target, path="desk", host_target=declared
        )
        took = time.monotonic() - began

        # GNU timeout's 124 for a command it stopped with TERM, and make's
        # line for its interrupted recipe, as both printed on 2026-11-27;
        # the recipe's last line never ran.
        assert ran == host_target.HostTargetRun(
            lines=("started", "make: *** [Makefile:10: slow] Terminated"), status=124
        )
        assert took < host_target.KILL_AFTER_SECONDS
        assert host_target.describe(ran, declared) == (
            "stopped at its deadline of 1 s (timeout exited 124)"
        )

    def test_a_target_that_outlives_term_is_described_by_kill_s_status(self) -> None:
        declared = HostTarget(target="slow", node="lavender", workers=1, deadline_seconds=60)
        ran = host_target.HostTargetRun(lines=(), status=137)

        assert host_target.describe(ran, declared) == (
            "stopped at its deadline of 60 s (timeout exited 137)"
        )


class TestAFailureAfterTheLease:
    def test_an_unreachable_node_gives_the_lease_back_and_leaves_the_retire_owed(
        self, tmp_path: pathlib.Path, hub: tuple[pathlib.Path, str]
    ) -> None:
        registry = _registry(tmp_path, host_target_entry=_entry("pass"))
        runs = registry.parent / "runs"
        _test_hooks.run = HubAndNode(hub[0], reachable=False)

        with pytest.raises(AppError) as raised:
            host_target_cli.main(_argv(registry))

        assert raised.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert leases.read_leases(runs / "leases.json") == ()
        assert _kinds(registry) == [FeedKind.LEASED, FeedKind.REFUSED]
        assert retire_owed.owed_on(runs / retire_owed.RETIRES_FILE, alias="lavender") == (
            RUN_ID,
        )

    def test_a_lease_another_run_holds_refuses_before_anything_is_staged(
        self, tmp_path: pathlib.Path, hub: tuple[pathlib.Path, str]
    ) -> None:
        registry = _registry(tmp_path, host_target_entry=_entry("pass"))
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(registry)})
        plan = loaded.workspace["projects"][PROJECT]
        leases.acquire(
            loaded.leases,
            run_lease.open_lease(
                node="lavender",
                project=PROJECT,
                run_id="games-desk-lavender-1756999000",
                agent="opus-other-1127",
                session_id="11111111-aaaa-4aaa-8aaa-111111111111",
                plan=plan,
                window_seconds=run_lease.check_window(plan),
                node_local=(),
                now_unix=DEMO_NOW - 60,
            ),
            now_unix=DEMO_NOW - 60,
        )
        _test_hooks.run = HubAndNode(hub[0], reachable=False)

        with pytest.raises(AppError) as raised:
            host_target_cli.main(_argv(registry))

        assert raised.value.code is FleetErrorCode.LEASE_HELD
        assert not (tmp_path / "stage").exists()


class TestARefusalBeforeAnythingRuns:
    def test_a_project_declaring_no_host_target_is_refused_by_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        registry = _registry(tmp_path, host_target_entry=None)
        nothing = FakeRun([])
        _test_hooks.run = nothing

        with pytest.raises(AppError) as raised:
            host_target_cli.main(_argv(registry))

        assert raised.value.code is FleetErrorCode.HOST_TARGET_UNDECLARED
        assert f"project {PROJECT!r} declares no host_target" in raised.value.message
        assert nothing.calls == []

    def test_a_node_the_registry_holds_disabled_is_asked_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        registry = _registry(tmp_path, host_target_entry=_entry("pass"), enabled=False)
        nothing = FakeRun([])
        _test_hooks.run = nothing

        with pytest.raises(AppError) as raised:
            host_target_cli.main(_argv(registry))

        assert raised.value.code is FleetErrorCode.NODE_DISABLED
        assert raised.value.message.startswith("lavender is declared disabled")
        assert nothing.calls == []

    def test_a_node_the_registry_does_not_hold_refuses_the_registry_by_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        registry = _registry(tmp_path, host_target_entry={**_entry("pass"), "node": "loki"})
        nothing = FakeRun([])
        _test_hooks.run = nothing

        with pytest.raises(JSONTypeError, match="runs its host target on node 'loki'"):
            host_target_cli.main(_argv(registry))

        assert nothing.calls == []


class TestTheNodeAnswer:
    def test_an_answer_without_its_status_line_is_refused_naming_the_line(self) -> None:
        with pytest.raises(AppError) as raised:
            host_target.read_run("lavender", "built\nKilled\n")

        assert raised.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "the last line was 'Killed'" in raised.value.message

    def test_an_empty_answer_is_refused(self) -> None:
        with pytest.raises(AppError, match="the last line was ''"):
            host_target.read_run("lavender", "")

    def test_the_status_line_is_read_off_and_the_rest_kept(self) -> None:
        ran = host_target.read_run("lavender", f"one\ntwo\n{host_target.STATUS_WORD} 0\n")

        assert ran == host_target.HostTargetRun(lines=("one", "two"), status=0)
