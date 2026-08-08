"""The worker loop, driven against the in-memory queue and a fake engine.

What is tested: a worker drains the queue through the harness seams in
claim order, files outcomes, leases and releases clones, stops on its
budget, and sleeps exactly once before deciding an empty queue is drained.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.harness.match import decode_match_config
from rw_bot.harness.runner import SweepConfig, decode_sweep_config
from rw_bot.harness.sweep import SweepJob, parse_job_line
from rw_bot.service import _test_hooks
from rw_bot.service.queue import bootstrap, submit
from rw_bot.service.worker import run_worker
from tests.service_fakes import FakeConnection

_CONFIG = decode_sweep_config(
    {
        "out_dir": "runs/sweeps/demo",
        "workers": 1,
        "lockstep": 75,
        "clone_prefix": ".game-w",
        "source_game_dir": ".game",
        "tree": "runs/sweeps/demo/.tree",
        "pin_delta": 3,
        "fast_forward": 10,
    },
    decode_match_config(
        {"map_path": "maps/skirmish/[p2]duel_lake.tmx", "opponents": 1, "difficulty": 2}
    ),
)

_JOBS = (
    parse_job_line("alpha|12345|doctrines/flame-nocover.doctrine|400"),
    parse_job_line("alpha|777|doctrines/flame-nocover.doctrine|400"),
)


class _Rig:
    """One worker run's seams, recorded: the connection, the engine, sleep."""

    def __init__(self, outcomes: dict[int, bool]) -> None:
        self.conn = FakeConnection()
        bootstrap(self.conn)
        submit(self.conn, "demo", _CONFIG, _JOBS)
        self.outcomes = outcomes
        self.trees: list[str] = []
        self.clones: list[int] = []
        self.played: list[tuple[int, str]] = []
        self.cards_read: list[str] = []
        self.slept: list[float] = []

    def connect(self, dsn: str) -> FakeConnection:
        return self.conn

    def prepare_tree(self, config: SweepConfig) -> None:
        self.trees.append(config["tree"])

    def prepare_clone(self, index: int, config: SweepConfig) -> str:
        self.clones.append(index)
        return f"{config['clone_prefix']}{index}"

    def play_job(self, job: SweepJob, game_dir: str, config: SweepConfig) -> bool:
        self.played.append((job["seed"], game_dir))
        return self.outcomes[job["seed"]]

    def read_card(self, path: str) -> str:
        self.cards_read.append(path)
        return f"scripted card from {path}\nverdict        won (won)"

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def _with_rig(outcomes: dict[int, bool], max_jobs: int) -> tuple[_Rig, int]:
    rig = _Rig(outcomes)
    saved = (
        _test_hooks.connect,
        _test_hooks.prepare_tree,
        _test_hooks.prepare_clone,
        _test_hooks.play_job,
        _test_hooks.read_card,
        _test_hooks.sleep,
    )
    _test_hooks.connect = rig.connect
    _test_hooks.prepare_tree = rig.prepare_tree
    _test_hooks.prepare_clone = rig.prepare_clone
    _test_hooks.play_job = rig.play_job
    _test_hooks.read_card = rig.read_card
    _test_hooks.sleep = rig.sleep
    try:
        played = run_worker("dsn://demo", "w1", (1, 2), max_jobs)
    finally:
        (
            _test_hooks.connect,
            _test_hooks.prepare_tree,
            _test_hooks.prepare_clone,
            _test_hooks.play_job,
            _test_hooks.read_card,
            _test_hooks.sleep,
        ) = saved
    return rig, played


def test_the_worker_drains_the_queue_in_claim_order() -> None:
    rig, played = _with_rig({12345: True, 777: True}, 0)
    assert played == 2
    assert [seed for seed, _ in rig.played] == [12345, 777]
    assert all(row.state == "done" for row in rig.conn.store.jobs)
    assert rig.conn.store.leases == {}
    assert rig.conn.closed is True


def test_each_match_plays_in_its_leased_clone() -> None:
    """The dir handed to play_job is the one the lease named."""
    rig, _ = _with_rig({12345: True, 777: True}, 0)
    assert rig.played[0][1] == ".game-w1"
    assert rig.clones[0] == 1


def test_the_tree_is_frozen_before_every_match() -> None:
    """prepare_tree is idempotent, and calling it per claim is what makes a
    worker joining an old batch inherit the batch's frozen code."""
    rig, _ = _with_rig({12345: True, 777: True}, 0)
    assert rig.trees == ["runs/sweeps/demo/.tree", "runs/sweeps/demo/.tree"]


def test_a_failed_match_files_as_failed_and_the_worker_continues() -> None:
    rig, played = _with_rig({12345: False, 777: True}, 0)
    assert played == 2
    states = [row.state for row in rig.conn.store.jobs]
    assert states == ["failed", "done"]


def test_a_finished_match_mirrors_its_filed_card_onto_the_row() -> None:
    """The card is read from where play_job filed it and lands in the queue."""
    rig, _ = _with_rig({12345: True, 777: True}, 0)
    path = str(Path("runs/sweeps/demo") / "alpha-s12345.txt")
    assert rig.cards_read[0] == path
    assert rig.conn.store.jobs[0].card == f"scripted card from {path}\nverdict        won (won)"


def test_a_failed_match_mirrors_no_card() -> None:
    """A failed job filed only a .partial; the mirror must not invent one."""
    rig, _ = _with_rig({12345: False, 777: True}, 0)
    assert rig.cards_read == [str(Path("runs/sweeps/demo") / "alpha-s777.txt")]
    assert rig.conn.store.jobs[0].card == ""


def test_the_budget_stops_the_worker_before_the_queue_empties() -> None:
    rig, played = _with_rig({12345: True, 777: True}, 1)
    assert played == 1
    assert rig.conn.store.jobs[1].state == "queued"


def test_an_empty_queue_is_polled_once_before_the_worker_leaves() -> None:
    """One sleep, one recheck: a queue empty twice in a row is drained."""
    rig, played = _with_rig({12345: True, 777: True}, 0)
    assert rig.slept == [15.0]
    assert played == 2
