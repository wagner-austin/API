"""A host's claims under way and the turn its runners claim in (MCPs board task a85ef09e).

Every case runs against real files in a temporary records directory. The
turn's lock is the exclusive creation of a file, so the cases that need it
held, given back or left behind make the filesystem do it: a second thread
holding the turn, a lock aged past the stale bound, a lock removed between
a runner's create and its look, and, through :mod:`tests._obstruct`, a claim
the platform refuses to remove for a moment, the way Windows refuses a file
another process has open.
"""

from __future__ import annotations

import os
import pathlib
import threading
import time
from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from fleet.contracts.host_claim import HostClaim, decode_host_claim, encode_host_claim
from fleet.contracts.node import LiveLoad
from fleet.core import _test_hooks, host_claims
from tests._obstruct import refused_removal
from tests._thread_fakes import await_event

#: The node every case claims on.
HOST = "serendipity"

#: Its two runners.
ORDINARY = "fleet-node-serendipity"
ELEVATED = "fleet-node-serendipity-elevated"


def _claim(job_id: str, *, until_unix: int, workers: int = 2) -> HostClaim:
    """A claim of MCPs/sms-gateway by the ordinary runner.

    Args:
        job_id: The queue job.
        until_unix: When it stops counting.
        workers: Its grant.

    Returns:
        The claim.
    """
    return HostClaim(
        job_id=job_id,
        runner=ORDINARY,
        project="MCPs/sms-gateway",
        workers=workers,
        ram_gb=workers * 1.1,
        until_unix=until_unix,
    )


class FakeClock:
    """A clock that a sleep moves, running the case's step at each sleep.

    Satisfies :class:`~fleet.core._test_hooks.NowProtocol` through
    :meth:`now` and :class:`~fleet.core._test_hooks.SleepProtocol` through
    :meth:`sleep`.

    Attributes:
        at: The current instant.
        slept: Every sleep asked for, in order.
    """

    at: int
    slept: list[int]

    def __init__(self, *, at: int, on_sleep: list[Callable[[], None]] | None = None) -> None:
        """Start the clock.

        Args:
            at: The first instant.
            on_sleep: Run in turn, one per sleep, so another thread or the
                filesystem can act between a runner's looks at the lock.
        """
        self.at = at
        self.slept = []
        self._on_sleep = on_sleep if on_sleep is not None else []

    def now(self) -> int:
        """Read the clock.

        Returns:
            The current instant.
        """
        return self.at

    def sleep(self, seconds: int) -> None:
        """Move the clock and run the next step.

        Args:
            seconds: How long the caller waits.
        """
        self.slept.append(seconds)
        self.at += seconds
        if self._on_sleep:
            self._on_sleep.pop(0)()


class TestTheClaimContract:
    def test_a_claim_survives_encoding(self) -> None:
        claim = _claim("25d07c7c", until_unix=1791453435)

        assert decode_host_claim(load_json_str(dump_json_str(encode_host_claim(claim)))) == claim

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="host claim must be a JSON object, got list"):
            decode_host_claim([1])

    def test_a_claim_of_no_workers_is_refused(self) -> None:
        encoded = encode_host_claim(_claim("25d07c7c", until_unix=1791453435))
        encoded["workers"] = 0

        with pytest.raises(JSONTypeError, match="host claim grants 0 workers"):
            decode_host_claim(encoded)


class TestTheTurn:
    def test_is_a_lock_naming_its_runner_and_is_given_back(self, tmp_path: pathlib.Path) -> None:
        lock = host_claims.lock_path(tmp_path, alias=HOST)

        with host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY):
            held = lock.read_text(encoding="utf-8")

        assert held.startswith(f"{ORDINARY} pid {os.getpid()} turn ")
        assert not lock.exists()

    def test_is_given_back_when_the_block_raises(self, tmp_path: pathlib.Path) -> None:
        with (
            pytest.raises(RuntimeError, match="the gate raised"),
            host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY),
        ):
            raise RuntimeError("the gate raised")

        assert not host_claims.lock_path(tmp_path, alias=HOST).exists()

    def test_waits_for_the_other_runners_turn_and_then_takes_it(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The elevated runner asks while the ordinary one claims, and claims after it."""
        holding, leave, left = threading.Event(), threading.Event(), threading.Event()

        def let_the_ordinary_runner_leave() -> None:
            leave.set()
            await_event(left, what="the ordinary runner leaving its turn")

        clock = FakeClock(at=1791449835, on_sleep=[let_the_ordinary_runner_leave])
        order: list[str] = []

        def ordinary_turn() -> None:
            with host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY):
                order.append(ORDINARY)
                holding.set()
                await_event(leave, what="the elevated runner's first look at the lock")
            left.set()

        other = threading.Thread(target=ordinary_turn)
        other.start()
        await_event(holding, what="the ordinary runner's turn")
        _test_hooks.now = clock.now
        _test_hooks.sleep = clock.sleep
        with (
            caplog.at_level("INFO"),
            host_claims.claim_turn(tmp_path, alias=HOST, runner=ELEVATED),
        ):
            order.append(ELEVATED)
        other.join()

        assert order == [ORDINARY, ELEVATED]
        assert clock.slept == [host_claims.TURN_POLL_SECONDS]
        assert (
            f"{ELEVATED} took its claim turn after waiting 1 s for its host's other runner"
        ) in [record.getMessage() for record in caplog.records]

    def test_takes_away_a_turn_whose_holder_died_inside_it(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        lock = host_claims.lock_path(tmp_path, alias=HOST)
        lock.write_text(f"{ORDINARY} pid 4242 turn dead", encoding="utf-8")
        died = time.time() - host_claims.TURN_STALE_SECONDS - 5
        os.utime(lock, (died, died))

        with (
            caplog.at_level("INFO"),
            host_claims.claim_turn(tmp_path, alias=HOST, runner=ELEVATED),
        ):
            held = lock.read_text(encoding="utf-8")

        assert held.startswith(f"{ELEVATED} pid ")
        assert not lock.exists()
        taken = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith(f"{ELEVATED} took away a claim turn held for")
        ]
        assert len(taken) == 1
        assert taken[0].endswith(
            f"past the {host_claims.TURN_STALE_SECONDS} s a live runner's turn can last, so its "
            "holder died inside it"
        )

    def test_leaves_a_lock_another_turn_took_after_this_one_outlived_its_bound(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        lock = host_claims.lock_path(tmp_path, alias=HOST)

        with (
            caplog.at_level("INFO"),
            host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY),
        ):
            lock.write_text(f"{ELEVATED} pid 4243 turn later", encoding="utf-8")

        assert lock.read_text(encoding="utf-8") == f"{ELEVATED} pid 4243 turn later"
        assert (
            f"a claim turn outlived the {host_claims.TURN_STALE_SECONDS} s bound and was taken "
            f"away; the lock now reads '{ELEVATED} pid 4243 turn later' and is left to its holder"
        ) in [record.getMessage() for record in caplog.records]

    def test_leaves_nothing_when_its_lock_was_taken_away_and_given_back(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        lock = host_claims.lock_path(tmp_path, alias=HOST)

        with (
            caplog.at_level("INFO"),
            host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY),
        ):
            lock.unlink()

        assert not lock.exists()
        messages = [record.getMessage() for record in caplog.records]
        assert any("and was taken away; the lock now reads None" in line for line in messages)

    def test_looks_again_when_the_lock_is_given_back_between_its_create_and_its_look(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The other runner leaves its turn after this one's create failed and before its stat."""
        lock = host_claims.lock_path(tmp_path, alias=HOST)
        lock.write_text(f"{ELEVATED} pid 4243 turn leaving", encoding="utf-8")
        clock = FakeClock(at=1791449835)
        looks: list[int] = []

        def now_as_the_other_runner_leaves() -> int:
            if len(looks) == 1:
                lock.unlink()
            looks.append(clock.at)
            return clock.now()

        _test_hooks.now = now_as_the_other_runner_leaves
        _test_hooks.sleep = clock.sleep
        with host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY):
            held = lock.read_text(encoding="utf-8")

        assert held.startswith(f"{ORDINARY} pid ")
        assert clock.slept == [host_claims.TURN_POLL_SECONDS]


class TestTheClaims:
    def test_a_host_that_never_claimed_holds_none(self, tmp_path: pathlib.Path) -> None:
        assert host_claims.live_claims(tmp_path, alias=HOST) == ()
        assert host_claims.host_load(()) == LiveLoad(runs=0, workers=0, ram_gb=0.0)

    def test_every_runners_claims_are_read_and_summed(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.now = FakeClock(at=1791449835).now
        first = _claim("25d07c7c", until_unix=1791453435)
        second = HostClaim(
            job_id="f65df8f5",
            runner=ELEVATED,
            project="MCPs/execution-elevated",
            workers=2,
            ram_gb=1.0,
            until_unix=1791453505,
        )
        host_claims.record(tmp_path, alias=HOST, claim=second)
        host_claims.record(tmp_path, alias=HOST, claim=first)

        live = host_claims.live_claims(tmp_path, alias=HOST)

        assert live == (first, second)
        assert host_claims.host_load(live) == LiveLoad(runs=2, workers=4, ram_gb=2 * 1.1 + 1.0)
        assert host_claims.live_claims(tmp_path, alias="sedona") == ()

    def test_a_lapsed_claim_is_removed_by_the_turn_that_finds_it(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _test_hooks.now = FakeClock(at=1791453435).now
        host_claims.record(tmp_path, alias=HOST, claim=_claim("25d07c7c", until_unix=1791453435))
        kept = _claim("c858b46b", until_unix=1791453436)
        host_claims.record(tmp_path, alias=HOST, claim=kept)

        with caplog.at_level("INFO"):
            live = host_claims.live_claims(tmp_path, alias=HOST)

        assert live == (kept,)
        directory = host_claims.claims_directory(tmp_path, alias=HOST)
        assert sorted(path.name for path in directory.iterdir()) == ["c858b46b.json"]
        assert (
            f"{HOST}: {ORDINARY}'s claim of 25d07c7c lapsed at 1791453435 unlaunched; removing it"
        ) in [record.getMessage() for record in caplog.records]

    def test_a_claim_removed_while_the_turn_reads_it_is_not_counted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Its launch ended between the listing and the read, so its run is on the ledger."""
        _test_hooks.now = FakeClock(at=1791449835).now
        host_claims.record(tmp_path, alias=HOST, claim=_claim("25d07c7c", until_unix=1791453435))
        kept = _claim("f65df8f5", until_unix=1791453435)
        host_claims.record(tmp_path, alias=HOST, claim=kept)
        removed = host_claims.claims_directory(tmp_path, alias=HOST) / "25d07c7c.json"
        real_read = _test_hooks.read_text

        def read_after_its_launch_removed_it(path: pathlib.Path) -> str:
            if path == removed:
                host_claims.discharge(tmp_path, alias=HOST, job_id="25d07c7c")
            return real_read(path)

        _test_hooks.read_text = read_after_its_launch_removed_it

        assert host_claims.live_claims(tmp_path, alias=HOST) == (kept,)

    def test_a_claim_file_that_is_not_a_claim_is_refused(self, tmp_path: pathlib.Path) -> None:
        directory = host_claims.claims_directory(tmp_path, alias=HOST)
        directory.mkdir(parents=True)
        (directory / "25d07c7c.json").write_text("[]", encoding="utf-8")

        with pytest.raises(JSONTypeError, match="host claim must be a JSON object, got list"):
            host_claims.live_claims(tmp_path, alias=HOST)

    def test_a_discharge_removes_only_its_own_claim(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.now = FakeClock(at=1791449835).now
        host_claims.record(tmp_path, alias=HOST, claim=_claim("25d07c7c", until_unix=1791453435))
        kept = _claim("f65df8f5", until_unix=1791453435)
        host_claims.record(tmp_path, alias=HOST, claim=kept)

        host_claims.discharge(tmp_path, alias=HOST, job_id="25d07c7c")
        host_claims.discharge(tmp_path, alias=HOST, job_id="00000000")

        assert host_claims.live_claims(tmp_path, alias=HOST) == (kept,)

    def test_a_discharge_the_platform_refuses_is_asked_again(self, tmp_path: pathlib.Path) -> None:
        """Windows refuses to remove a file the host's other runner has open to read."""
        host_claims.record(tmp_path, alias=HOST, claim=_claim("25d07c7c", until_unix=1791453435))
        claim = host_claims.claims_directory(tmp_path, alias=HOST) / "25d07c7c.json"
        with refused_removal(claim) as release:
            clock = FakeClock(at=1791449835, on_sleep=[release])
            _test_hooks.now = clock.now
            _test_hooks.sleep = clock.sleep
            host_claims.discharge(tmp_path, alias=HOST, job_id="25d07c7c")

        assert not claim.exists()
        assert clock.slept == [host_claims.TURN_POLL_SECONDS]

    def test_a_discharge_refused_past_the_turns_bound_raises(self, tmp_path: pathlib.Path) -> None:
        host_claims.record(tmp_path, alias=HOST, claim=_claim("25d07c7c", until_unix=1791453435))
        claim = host_claims.claims_directory(tmp_path, alias=HOST) / "25d07c7c.json"
        with refused_removal(claim):
            clock = FakeClock(at=1791449835)
            _test_hooks.now = clock.now
            _test_hooks.sleep = clock.sleep
            with pytest.raises(PermissionError):
                host_claims.discharge(tmp_path, alias=HOST, job_id="25d07c7c")

        assert len(clock.slept) == host_claims.TURN_STALE_SECONDS // host_claims.TURN_POLL_SECONDS
        assert claim.exists()
