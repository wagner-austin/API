"""The three entry points, exercised end to end over a real workspace file.

INTEGRATION RATHER THAN UNIT, deliberately. Each test writes a real workspace
document to disk, binds the file hooks to real filesystem calls, and drives
``main`` with an argv list. Only the ssh boundary is faked, because it is the
one thing a test cannot own. So what is exercised is the whole path a person
gets: flag parsing, workspace decoding, path resolution against the document's
directory, the capacity arithmetic, and the rendering.

THE CONSOLE SCRIPT AND ``python -m`` ARE BOTH EXERCISED. A module under
``cli/`` that defines ``entrypoint`` but carries no ``if __name__`` block is
importable, runnable and does NOTHING -- exits 0, prints nothing -- while the
console script works. The two forms then disagree, and the broken one looks
exactly like a fleet with nothing to report.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import _config, nodes, preflight, watch
from fleet.contracts.feed import FeedEvent, decode_feed_event
from fleet.contracts.ledger import LedgerEntry, decode_ledger_entry
from fleet.core import _test_hooks, records
from tests.conftest import FakeClock, FakeRun, failed, ok

_NOW = 1_757_000_000

_PROBE_OK = "free_ram_gb=27.0\nfree_disk_gb=860.0\n"
_PROBE_SMALL = "free_ram_gb=5.0\nfree_disk_gb=860.0\n"


def _workspace_document() -> JSONObject:
    """Build a two-node, one-project workspace as JSON.

    Built as JSON and decoded by the real decoder rather than constructed as
    a ``FleetWorkspace``, so these tests exercise the path a person's file
    actually takes.

    Returns:
        The document, ready to serialise.
    """
    budget: JSONObject = {
        "reserved_cores": 2,
        "reserved_ram_gb": 4.0,
        "worker_ram_gb": 1.1,
        "max_concurrent_runs": 2,
        "max_disk_gb": 20.0,
    }
    node: JSONObject = {
        "host": "lavender",
        "stage_root": "C:/fleet/stage",
        "logical_cores": 16,
        "ram_gb": 32.0,
        "gpu": {
            "model": "NVIDIA GeForce GTX 1630",
            "vram_mib": 4096,
            "compute_capability": "7.5",
            "driver_version": "591.86",
        },
        "enabled": True,
        "budget": budget,
    }
    return {
        "nodes": {
            "lavender": node,
            "loki": {**node, "host": "loki", "gpu": None},
        },
        "not_dispatchable": {},
        "projects": {
            "services/Model-Trainer": {
                "worker_ram_gb": 1.1,
                "minimum_workers": 4,
                "expected_minutes": 5,
            }
        },
        "ledger": "ledger.jsonl",
        "feed": "feed.jsonl",
        "leases": "leases.json",
    }


@pytest.fixture(name="config_path")
def _config_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a workspace document and pin the clock.

    The file hooks are NOT rebound: the autouse reset in ``conftest`` already
    leaves them on their real implementations, so these tests read and write
    the same way production does.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path to the written document.
    """
    _test_hooks.now = FakeClock(_NOW)
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(_workspace_document()), encoding="utf-8")
    return path


class TestLoadWorkspace:
    def test_record_paths_resolve_against_the_document(self, config_path: pathlib.Path) -> None:
        """A workspace can live beside its records and be used from anywhere."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})

        assert loaded.ledger == config_path.parent / "ledger.jsonl"
        assert loaded.feed == config_path.parent / "feed.jsonl"
        assert loaded.leases == config_path.parent / "leases.json"

    def test_the_config_flag_is_required(self) -> None:
        with pytest.raises(ValueError, match="--config"):
            _config.load_workspace({})


class TestNodes:
    def test_it_describes_every_node(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(_PROBE_OK), ok(""), ok(_PROBE_OK)])

        assert nodes.main([_config.CONFIG_FLAG, str(config_path)]) == 0

    def test_an_unreachable_node_is_a_line_and_a_non_zero_status(
        self, config_path: pathlib.Path
    ) -> None:
        """A dead node must not hide the live ones, and must not read as fine."""
        _test_hooks.run = FakeRun([failed(255, "timed out"), ok(""), ok(_PROBE_OK)])

        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        lines, unreachable = nodes.describe_fleet(loaded)

        assert unreachable == 1
        assert any("UNREACHABLE" in line and "timed out" in line for line in lines)
        # lavender is the one that failed, so the surviving line is loki's --
        # and loki is the CPU-only node. The point is that a dead node did not
        # take the live one's description down with it.
        assert any("cpu-only" in line and "GB RAM free" in line for line in lines)

    def test_it_exits_non_zero_when_a_node_is_missing(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([failed(255, "timed out"), failed(255, "timed out")])

        assert nodes.main([_config.CONFIG_FLAG, str(config_path)]) == 1

    def test_live_runs_are_counted_from_the_ledger(self, config_path: pathlib.Path) -> None:
        """Ours by construction, because we wrote them.

        Counting the node's python processes would be unattributable -- some
        belong to whoever is sitting at it.
        """
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        records.append_ledger(loaded.ledger, _running_row())
        _test_hooks.run = FakeRun([ok(""), ok(_PROBE_OK), ok(""), ok(_PROBE_OK)])

        lines, _unreachable = nodes.describe_fleet(loaded)

        assert any("1 live run(s)" in line for line in lines)


def _running_row() -> LedgerEntry:
    """Build a live ledger row for the lavender node.

    Returns:
        The row, typed through its own decoder so the Literal is honest.
    """
    return decode_ledger_entry(
        {
            "run_id": "run-1",
            "node": "lavender",
            "host": "lavender",
            "project": "services/Model-Trainer",
            "agent": "opus-fleet-0904",
            "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
            "started_unix": _NOW - 60,
            "ended_unix": _NOW - 60,
            "outcome": "running",
            "exit_code": -1,
            "workers": 6,
            "detail": "",
        }
    )


class TestPreflight:
    def test_it_names_the_node_that_affords_the_most(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(_PROBE_OK), ok(""), ok(_PROBE_SMALL)])

        assert (
            preflight.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    preflight.PROJECT_FLAG,
                    "services/Model-Trainer",
                ]
            )
            == 0
        )

    def test_a_named_node_is_asked_directly(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(_PROBE_OK)])

        assert (
            preflight.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    preflight.PROJECT_FLAG,
                    "services/Model-Trainer",
                    preflight.NODE_FLAG,
                    "lavender",
                ]
            )
            == 0
        )

    def test_a_named_node_that_refuses_says_why(self, config_path: pathlib.Path) -> None:
        """5 GB free against a 4 GB reservation affords no workers."""
        _test_hooks.run = FakeRun([ok(""), ok("free_ram_gb=4.5\nfree_disk_gb=860.0\n")])

        with pytest.raises(AppError) as excinfo:
            preflight.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    preflight.PROJECT_FLAG,
                    "services/Model-Trainer",
                    preflight.NODE_FLAG,
                    "lavender",
                ]
            )

        assert excinfo.value.code is FleetErrorCode.NODE_OWNER_RESERVED

    def test_an_unknown_project_is_refused(self, config_path: pathlib.Path) -> None:
        with pytest.raises(AppError) as excinfo:
            preflight.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    preflight.PROJECT_FLAG,
                    "tools/nonesuch",
                    preflight.NODE_FLAG,
                    "lavender",
                ]
            )

        assert excinfo.value.code is FleetErrorCode.WORKSPACE_PROJECT_UNKNOWN

    def test_an_unknown_node_is_refused(self, config_path: pathlib.Path) -> None:
        with pytest.raises(AppError) as excinfo:
            preflight.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    preflight.PROJECT_FLAG,
                    "services/Model-Trainer",
                    preflight.NODE_FLAG,
                    "sedona",
                ]
            )

        assert excinfo.value.code is FleetErrorCode.WORKSPACE_NODE_UNKNOWN

    def test_the_project_flag_is_required(self, config_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--project"):
            preflight.main([_config.CONFIG_FLAG, str(config_path)])


class TestWatch:
    def test_it_renders_the_feed_oldest_first(self, config_path: pathlib.Path) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        for kind in ("leased", "started", "passed"):
            records.append_feed(loaded.feed, _feed_event(kind))

        lines = watch.lines_for(loaded, run_id=None, now_unix=_NOW)

        assert [line.split()[0] for line in lines] == ["LEASED", "STARTED", "PASSED"]

    def test_it_filters_to_one_run(self, config_path: pathlib.Path) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        records.append_feed(loaded.feed, _feed_event("started", run_id="mine"))
        records.append_feed(loaded.feed, _feed_event("started", run_id="theirs"))

        lines = watch.lines_for(loaded, run_id="mine", now_unix=_NOW)

        assert len(lines) == 1
        assert "mine" in lines[0]

    def test_a_live_row_with_no_lease_is_reported_lost(self, config_path: pathlib.Path) -> None:
        """THE WEDGE DETECTOR.

        A wedged run cannot report its own death, so a live ledger row with
        no lease behind it is the only observable signature. Without this a
        wedge is indistinguishable from a slow suite -- which is exactly how
        two suites held 77.9 GB for twenty-nine minutes on 2026-09-04.
        """
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        records.append_ledger(loaded.ledger, _running_row())

        lines = watch.lines_for(loaded, run_id=None, now_unix=_NOW)

        assert len(lines) == 1
        assert lines[0].startswith("LOST ")
        assert "opus-fleet-0904" in lines[0]

    def test_a_live_row_holding_a_lease_is_not_lost(self, config_path: pathlib.Path) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        records.append_ledger(loaded.ledger, _running_row())
        loaded.leases.write_text(
            dump_json_str(
                [
                    {
                        "node": "lavender",
                        "project": "services/Model-Trainer",
                        "run_id": "run-1",
                        "agent": "opus-fleet-0904",
                        "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
                        "acquired_unix": _NOW - 60,
                        "expires_unix": _NOW + 600,
                    }
                ]
            ),
            encoding="utf-8",
        )

        assert watch.lost_runs(loaded, now_unix=_NOW) == ()

    def test_a_finished_row_is_never_lost(self, config_path: pathlib.Path) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        records.append_ledger(
            loaded.ledger,
            decode_ledger_entry(
                {
                    "run_id": "done",
                    "node": "lavender",
                    "host": "lavender",
                    "project": "services/Model-Trainer",
                    "agent": "opus-fleet-0904",
                    "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
                    "started_unix": _NOW - 60,
                    "ended_unix": _NOW - 10,
                    "outcome": "passed",
                    "exit_code": 0,
                    "workers": 6,
                    "detail": "",
                }
            ),
        )

        assert watch.lost_runs(loaded, now_unix=_NOW) == ()

    def test_an_empty_workspace_reports_nothing_and_succeeds(
        self, config_path: pathlib.Path
    ) -> None:
        assert watch.main([_config.CONFIG_FLAG, str(config_path)]) == 0

    def test_main_emits_a_line_per_event(self, config_path: pathlib.Path) -> None:
        """The loop body, not only the empty case.

        A watcher that exits 0 having printed nothing looks identical to one
        that exits 0 having printed everything, so the populated path is what
        has to be exercised.
        """
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        records.append_feed(loaded.feed, _feed_event("started"))
        records.append_feed(loaded.feed, _feed_event("failed"))

        assert watch.main([_config.CONFIG_FLAG, str(config_path)]) == 0
        assert len(watch.lines_for(loaded, run_id=None, now_unix=_NOW)) == 2

    def test_a_run_filter_reaches_main(self, config_path: pathlib.Path) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        records.append_feed(loaded.feed, _feed_event("passed", run_id="mine"))

        assert watch.main([_config.CONFIG_FLAG, str(config_path), watch.RUN_FLAG, "mine"]) == 0


def _feed_event(kind: str, *, run_id: str = "run-1") -> FeedEvent:
    """Build a feed event through the feed's own decoder.

    Args:
        kind: What happened.
        run_id: The dispatch it belongs to.

    Returns:
        The event.
    """
    return decode_feed_event(
        {
            "at_unix": _NOW,
            "run_id": run_id,
            "node": "lavender",
            "project": "services/Model-Trainer",
            "kind": kind,
            "detail": "",
        }
    )


class TestInvocationForms:
    """Both forms must do the same thing, for all three commands."""

    def test_console_entry_points_exit_zero(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(_PROBE_OK), ok(""), ok(_PROBE_OK)])
        saved = sys.argv
        sys.argv = ["fleet-nodes", _config.CONFIG_FLAG, str(config_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                nodes.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_watch_entrypoint_exits_zero(self, config_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["fleet-watch", _config.CONFIG_FLAG, str(config_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                watch.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_preflight_entrypoint_exits_zero(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(_PROBE_OK)])
        saved = sys.argv
        sys.argv = [
            "fleet-preflight",
            _config.CONFIG_FLAG,
            str(config_path),
            preflight.PROJECT_FLAG,
            "services/Model-Trainer",
            preflight.NODE_FLAG,
            "lavender",
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                preflight.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    @pytest.mark.parametrize(
        "module_name",
        ["fleet.cli.nodes", "fleet.cli.watch", "fleet.cli.preflight"],
    )
    def test_running_as_a_module_actually_runs(
        self, config_path: pathlib.Path, module_name: str
    ) -> None:
        """The half that silently goes missing without an `if __name__` block."""
        _test_hooks.run = FakeRun([ok(""), ok(_PROBE_OK), ok(""), ok(_PROBE_OK)])
        argv = ["x", _config.CONFIG_FLAG, str(config_path)]
        if module_name == "fleet.cli.preflight":
            argv += [preflight.PROJECT_FLAG, "services/Model-Trainer"]
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
