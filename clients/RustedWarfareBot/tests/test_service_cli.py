"""The submission and worker entry points, end to end against fakes.

The job file really parses, the queue really queues, the worker really
drains -- only the filesystem, the database and the engine are fakes.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import TracebackType

import pytest
from scripts.match_worker import main as worker_main
from scripts.submit_batch import main as submit_main

from rw_bot.harness.runner import SweepConfig
from rw_bot.harness.sweep import SweepJob
from rw_bot.service import _test_hooks as service_hooks
from tests.harness_fakes import FakeHost
from tests.service_fakes import FakeConnection

_JOBS = "sweeps/demo.txt"
_LINES = (
    "alpha|12345|doctrines/flame-nocover.doctrine|400",
    "alpha|777|doctrines/flame-nocover.doctrine|400",
)


class _Service:
    """A shared fake database plus a scripted engine, for one test."""

    def __init__(self) -> None:
        self.conn = FakeConnection()
        self.played: list[int] = []
        # Captured at construction, where inference gives the hooks their
        # own precise types; __exit__ restores exactly these.
        self._saved = (
            service_hooks.connect,
            service_hooks.prepare_tree,
            service_hooks.prepare_clone,
            service_hooks.play_job,
            service_hooks.read_card,
            service_hooks.sleep,
        )

    def connect(self, dsn: str) -> FakeConnection:
        self.conn.closed = False
        return self.conn

    def prepare_tree(self, config: SweepConfig) -> None:
        return None

    def prepare_clone(self, index: int, config: SweepConfig) -> str:
        return f".game-w{index}"

    def play_job(self, job: SweepJob, game_dir: str, config: SweepConfig) -> bool:
        self.played.append(job["seed"])
        return True

    def read_card(self, path: str) -> str:
        return f"### {Path(path).stem}\nverdict        won (won)"

    def __enter__(self) -> _Service:
        service_hooks.connect = self.connect
        service_hooks.prepare_tree = self.prepare_tree
        service_hooks.prepare_clone = self.prepare_clone
        service_hooks.play_job = self.play_job
        service_hooks.read_card = self.read_card
        service_hooks.sleep = lambda seconds: None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        (
            service_hooks.connect,
            service_hooks.prepare_tree,
            service_hooks.prepare_clone,
            service_hooks.play_job,
            service_hooks.read_card,
            service_hooks.sleep,
        ) = self._saved


def test_a_bad_submit_argument_count_prints_usage() -> None:
    with FakeHost() as host:
        assert submit_main(["only", "two"]) == 2
        assert any(line.startswith("usage: submit_batch") for line in host.printed)


def test_a_bad_worker_argument_count_prints_usage() -> None:
    with FakeHost() as host:
        assert worker_main(["only"]) == 2
        assert any(line.startswith("usage: match_worker") for line in host.printed)


def test_a_submitted_batch_is_played_by_a_worker() -> None:
    """The whole phase-zero loop: submit queues, a worker drains, rows file."""
    with FakeHost() as host, _Service() as service:
        host.write_text_lines(Path(_JOBS), _LINES)
        code = submit_main(
            ["dsn://q", _JOBS, "demo", "75", "maps/skirmish/[p2]duel_lake.tmx", "2", "3", "10"]
        )
        assert code == 0
        assert any("2 of 2 matches newly queued" in line for line in host.printed)
        assert worker_main(["dsn://q", "w1", "1,2"]) == 0
        assert service.played == [12345, 777]
        assert all(row.state == "done" for row in service.conn.store.jobs)
        assert any("played 2 match(es)" in line for line in host.printed)


def test_the_stored_match_line_reaches_the_submission_log() -> None:
    """The submitter states the match exactly as sweeps state theirs."""
    with FakeHost() as host, _Service():
        host.write_text_lines(Path(_JOBS), _LINES)
        submit_main(["dsn://q", _JOBS, "demo", "75", "maps/skirmish/[p2]duel_lake.tmx", "2"])
        assert any(line.startswith("[submit] ") and "difficulty 2" in line for line in host.printed)


def test_the_module_guards_run_main() -> None:
    """Both entry points are runnable as modules, like every script here."""
    with FakeHost() as host, _Service():
        argv = sys.argv
        for module in ("scripts.submit_batch", "scripts.match_worker"):
            already_imported = sys.modules.pop(module)
            sys.argv = [module.rsplit(".", 1)[1]]
            try:
                with pytest.raises(SystemExit) as caught:
                    runpy.run_module(module, run_name="__main__")
                assert caught.value.code == 2
            finally:
                sys.argv = argv
                sys.modules[module] = already_imported
        assert len(host.printed) == 2


def test_a_batch_without_a_match_submits_without_a_match_line() -> None:
    """Sandbox batches state no match, in the queue as in sweeps."""
    with FakeHost() as host, _Service() as service:
        host.write_text_lines(Path(_JOBS), _LINES)
        assert submit_main(["dsn://q", _JOBS, "demo"]) == 0
        assert not any("difficulty" in line for line in host.printed)
        assert len(service.conn.store.jobs) == 2
