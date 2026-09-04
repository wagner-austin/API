"""The two commands that start and stop a run, end to end.

THE TEST THAT MATTERS MOST is
:meth:`TestRun.test_a_second_dispatch_into_one_project_is_refused_before_anything_is_copied`.
It is the 2026-09-04 incident as a regression: two dispatches into one
project's environment, where the second must be turned away BEFORE it stages,
because staging into a tree another run holds is the corruption itself.

Only the ssh boundary is faked; the records are real files on a real path, so
the ledger, feed and lease interactions are the ones production performs.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.cli import _config, cancel, run
from fleet.contracts.ledger import decode_ledger_entry
from fleet.contracts.project import ProjectConfig
from fleet.core import _test_hooks, dispatch, leases, records, staging
from tests.conftest import (
    DEMO_NOW,
    DEMO_PROJECT,
    DEMO_RUN_ID,
    PROBE_OK,
    FakeClock,
    FakeRun,
    dispatch_replies,
    failed,
    ok,
    prebuilt_archive,
)


def _plan() -> ProjectConfig:
    """The demo project's declaration.

    Returns:
        The project.
    """
    return ProjectConfig(
        worker_ram_gb=1.1, minimum_workers=2, expected_minutes=5, exclusive_resources=()
    )


class TestRunIdentity:
    def test_a_run_id_names_its_project_and_time(self) -> None:
        assert dispatch.run_id_for(DEMO_PROJECT, started_unix=DEMO_NOW) == DEMO_RUN_ID

    def test_the_lease_is_sized_at_twice_the_estimate(self) -> None:
        lease = dispatch.open_lease(
            node="lavender",
            project=DEMO_PROJECT,
            run_id=DEMO_RUN_ID,
            agent="opus-fleet-0904",
            session_id="s",
            plan=_plan(),
            now_unix=DEMO_NOW,
        )

        assert lease["expires_unix"] - lease["acquired_unix"] == 600


class TestRun:
    def test_the_archive_is_built_where_the_records_live(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Not beside the workspace document. An archive is run output, and
        one left where a tree is staged FROM is carried by the next dispatch:
        measured 2026-09-04, a 20.7 MB archive beside fleet.json made the
        following dispatch 42.5 MB."""
        # The archive is placed where the workspace says archives go. tar is
        # faked, so if run.py used any OTHER directory the dispatch would fail
        # reading its own archive back -- which is what makes exit 0 here an
        # assertion about the path and not merely about the file existing.
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})

        code = run.main(
            [
                _config.CONFIG_FLAG,
                str(config_path),
                run.PROJECT_FLAG,
                DEMO_PROJECT,
                run.AGENT_FLAG,
                "opus-fleet-0904",
                run.SESSION_FLAG,
                "s",
                run.ROOT_FLAG,
                str(repo),
            ]
        )

        assert code == 0
        assert loaded.archives != loaded.directory
        assert (loaded.archives / f"{DEMO_RUN_ID}.tgz").is_file()
        assert not (loaded.directory / f"{DEMO_RUN_ID}.tgz").exists()

    def test_a_dispatch_leases_stages_launches_and_records(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))

        code = run.main(
            [
                _config.CONFIG_FLAG,
                str(config_path),
                run.PROJECT_FLAG,
                DEMO_PROJECT,
                run.AGENT_FLAG,
                "opus-fleet-0904",
                run.SESSION_FLAG,
                "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
                run.ROOT_FLAG,
                str(repo),
            ]
        )

        assert code == 0
        rows = records.read_ledger(loaded.ledger)
        assert [row["outcome"] for row in rows] == ["running"]
        assert rows[0]["run_id"] == DEMO_RUN_ID
        assert rows[0]["agent"] == "opus-fleet-0904"
        assert [event["kind"] for event in records.read_feed(loaded.feed)] == [
            "leased",
            "staged",
            "started",
        ]
        held = leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW)
        assert held == {
            "node": "lavender",
            "project": DEMO_PROJECT,
            "run_id": DEMO_RUN_ID,
            "agent": "opus-fleet-0904",
            "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
            "acquired_unix": DEMO_NOW,
            "expires_unix": DEMO_NOW + 600,
            "resources": (),
        }

    def test_a_second_dispatch_into_one_project_is_refused_before_anything_is_copied(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """THE 2026-09-04 INCIDENT, AS A REGRESSION TEST.

        Two sessions in one project's environment: the second's `poetry sync`
        deleted the first's interpreter mid-run. Here the second dispatch is
        refused at the lease, BEFORE it stages -- because staging into a tree
        another run holds is the corruption, not a step towards it.
        """
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        argv = [
            _config.CONFIG_FLAG,
            str(config_path),
            run.PROJECT_FLAG,
            DEMO_PROJECT,
            run.AGENT_FLAG,
            "opus-fleet-0904",
            run.SESSION_FLAG,
            "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
            run.ROOT_FLAG,
            str(repo),
        ]
        run.main(argv)

        # A fresh runner that would ANSWER a probe but has nothing scripted
        # for a stage: reaching one is the failure this test is about.
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])
        with pytest.raises(AppError) as excinfo:
            run.main(argv)

        assert excinfo.value.code is FleetErrorCode.LEASE_HELD
        assert "opus-fleet-0904" in excinfo.value.message
        assert len(records.read_ledger(loaded.ledger)) == 1

    def test_a_node_that_refuses_stops_the_dispatch(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("free_ram_gb=4.5\nfree_disk_gb=860.0\n")])

        with pytest.raises(AppError) as excinfo:
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.AGENT_FLAG,
                    "opus-fleet-0904",
                    run.SESSION_FLAG,
                    "s",
                    run.ROOT_FLAG,
                    str(repo),
                ]
            )

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED

    def test_a_named_node_is_used_directly(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))

        assert (
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.NODE_FLAG,
                    "lavender",
                    run.AGENT_FLAG,
                    "opus-fleet-0904",
                    run.SESSION_FLAG,
                    "s",
                    run.ROOT_FLAG,
                    str(repo),
                ]
            )
            == 0
        )
        assert len(records.read_ledger(loaded.ledger)) == 1

    def test_the_agent_flag_is_required(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """A default label would be one label shared by every session."""
        with pytest.raises(ValueError, match="--agent"):
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.ROOT_FLAG,
                    str(repo),
                ]
            )


class TestCancel:
    def _dispatch(self, config_path: pathlib.Path, repo: pathlib.Path) -> None:
        """Start one run so there is something to cancel.

        Args:
            config_path: The workspace document.
            repo: The monorepo root.
        """
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        run.main(
            [
                _config.CONFIG_FLAG,
                str(config_path),
                run.PROJECT_FLAG,
                DEMO_PROJECT,
                run.AGENT_FLAG,
                "opus-fleet-0904",
                run.SESSION_FLAG,
                "s",
                run.ROOT_FLAG,
                str(repo),
            ]
        )

    def test_it_closes_the_row_emits_and_releases_the_lease(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        self._dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok("stopped")])

        assert (
            cancel.main([_config.CONFIG_FLAG, str(config_path), cancel.RUN_FLAG, DEMO_RUN_ID]) == 0
        )

        rows = records.read_ledger(loaded.ledger)
        assert [row["outcome"] for row in rows] == ["running", "cancelled"]
        assert records.read_feed(loaded.feed)[-1]["kind"] == "cancelled"
        assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) is None

    def test_an_unknown_run_is_refused(self, config_path: pathlib.Path) -> None:
        with pytest.raises(AppError) as excinfo:
            cancel.main([_config.CONFIG_FLAG, str(config_path), cancel.RUN_FLAG, "nope"])

        assert excinfo.value.code is FleetErrorCode.RUN_UNKNOWN

    def test_a_finished_run_cannot_be_cancelled_twice(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The LAST row wins, because a closing row supersedes its running one."""
        self._dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok("stopped")])
        cancel.main([_config.CONFIG_FLAG, str(config_path), cancel.RUN_FLAG, DEMO_RUN_ID])

        with pytest.raises(AppError) as excinfo:
            cancel.main([_config.CONFIG_FLAG, str(config_path), cancel.RUN_FLAG, DEMO_RUN_ID])

        assert excinfo.value.code is FleetErrorCode.RUN_UNKNOWN
        assert "already ended" in excinfo.value.message

    def test_a_wedged_run_whose_lease_lapsed_is_still_cancellable(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """THE CASE THE ONLY KILL TOOL MUST NOT REFUSE.

        A run whose lease expired while it was still going IS the wedge, and
        refusing to stop it because the lease is gone would leave nothing able
        to.
        """
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        self._dispatch(config_path, repo)
        later = FakeClock(DEMO_NOW + 100_000)
        _test_hooks.now = later
        _test_hooks.run = FakeRun([ok(""), ok("stopped")])

        assert (
            cancel.main([_config.CONFIG_FLAG, str(config_path), cancel.RUN_FLAG, DEMO_RUN_ID]) == 0
        )
        assert records.read_ledger(loaded.ledger)[-1]["outcome"] == "cancelled"


class TestFinish:
    def test_it_records_emits_and_releases(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        lease = dispatch.open_lease(
            node="lavender",
            project=DEMO_PROJECT,
            run_id=DEMO_RUN_ID,
            agent="opus-fleet-0904",
            session_id="s",
            plan=_plan(),
            now_unix=DEMO_NOW,
        )
        leases.acquire(loaded.leases, lease, now_unix=DEMO_NOW)
        row = dispatch.started_row(lease=lease, host="lavender", workers=6, detail="")
        records.append_ledger(loaded.ledger, row)

        closing = dispatch.finish(
            loaded.leases,
            loaded.ledger,
            loaded.feed,
            row=row,
            outcome="passed",
            exit_code=0,
            detail="156 passed",
        )

        assert closing["outcome"] == "passed"
        assert closing["exit_code"] == 0
        assert records.read_feed(loaded.feed)[-1]["kind"] == "passed"
        assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) is None
        assert [row["outcome"] for row in records.read_ledger(loaded.ledger)] == [
            "running",
            "passed",
        ]

    def test_a_closing_row_keeps_the_identity_of_its_running_one(self) -> None:
        row = decode_ledger_entry(
            {
                "run_id": DEMO_RUN_ID,
                "node": "lavender",
                "host": "lavender",
                "project": DEMO_PROJECT,
                "agent": "opus-fleet-0904",
                "session_id": "s",
                "started_unix": DEMO_NOW,
                "ended_unix": DEMO_NOW,
                "outcome": "running",
                "exit_code": -1,
                "workers": 6,
                "detail": "",
            }
        )

        closing = dispatch.closed_row(
            row, outcome="failed", exit_code=2, ended_unix=DEMO_NOW + 60, detail="make check failed"
        )

        assert closing["run_id"] == row["run_id"]
        assert closing["started_unix"] == row["started_unix"]
        assert closing["workers"] == row["workers"]
        assert closing["ended_unix"] == DEMO_NOW + 60


class TestInvocationForms:
    def test_run_entrypoint_exits_zero(self, config_path: pathlib.Path, repo: pathlib.Path) -> None:
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        saved = sys.argv
        sys.argv = [
            "fleet-run",
            _config.CONFIG_FLAG,
            str(config_path),
            run.PROJECT_FLAG,
            DEMO_PROJECT,
            run.AGENT_FLAG,
            "opus-fleet-0904",
            run.SESSION_FLAG,
            "s",
            run.ROOT_FLAG,
            str(repo),
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                run.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_cancel_entrypoint_exits_zero(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        TestCancel()._dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok("stopped")])
        saved = sys.argv
        sys.argv = [
            "fleet-cancel",
            _config.CONFIG_FLAG,
            str(config_path),
            cancel.RUN_FLAG,
            DEMO_RUN_ID,
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                cancel.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_an_unreachable_node_fails_the_dispatch(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun([failed(255, "timed out")])

        with pytest.raises(AppError) as excinfo:
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.AGENT_FLAG,
                    "opus-fleet-0904",
                    run.SESSION_FLAG,
                    "s",
                    run.ROOT_FLAG,
                    str(repo),
                ]
            )

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE

    def test_running_run_as_a_module_actually_dispatches(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The half that silently goes missing without an `if __name__` block.

        A module under `cli/` that defines `entrypoint` and carries no such
        block imports, runs nothing and exits 0 -- which for this command
        reads as a successful dispatch that never happened.
        """
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.run", None)
        sys.argv = [
            "x",
            _config.CONFIG_FLAG,
            str(config_path),
            run.PROJECT_FLAG,
            DEMO_PROJECT,
            run.AGENT_FLAG,
            "opus-fleet-0904",
            run.SESSION_FLAG,
            "s",
            run.ROOT_FLAG,
            str(repo),
        ]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.run", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.run"] = saved_module

        assert raised.value.code == 0

    def test_running_cancel_as_a_module_actually_cancels(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Same hazard, worse consequence: a cancellation that never happened
        leaves a suite running that somebody believes they stopped."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        TestCancel()._dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok("stopped")])
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.cancel", None)
        sys.argv = ["x", _config.CONFIG_FLAG, str(config_path), cancel.RUN_FLAG, DEMO_RUN_ID]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.cancel", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.cancel"] = saved_module

        assert raised.value.code == 0
        assert records.read_ledger(loaded.ledger)[-1]["outcome"] == "cancelled"
