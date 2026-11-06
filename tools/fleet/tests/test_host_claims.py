"""A host's claims under way and the turn its processes claim in (MCPs board task a85ef09e).

Every case runs against real files in a temporary records directory. The
turn is an exclusive ``flock`` on a lock file (:mod:`fleet.core.file_turn`),
so the cases that need it held make a second thread hold it, and ask
whether it is held with a ``flock`` of their own.
"""

from __future__ import annotations

import pathlib
import re
import threading
import time

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from fleet.contracts.host_claim import HostClaim, decode_host_claim, encode_host_claim
from fleet.contracts.node import LiveLoad
from fleet.core import _test_hooks, host_claims
from tests._held_turns import is_held
from tests._thread_fakes import await_event

#: The node every case claims on.
HOST = "serendipity"

#: Its runner, and a second process claiming for the host, labelled apart
#: so a case can tell the two turns apart.
ORDINARY = "fleet-node-serendipity"
SECOND = "fleet-node-serendipity-second"


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
        cores=workers,
        ram_gb=workers * 1.1,
        until_unix=until_unix,
    )


class FakeClock:
    """A clock standing at one instant.

    Satisfies :class:`~fleet.core._test_hooks.NowProtocol` through :meth:`now`.

    Attributes:
        at: The current instant.
    """

    at: int

    def __init__(self, *, at: int) -> None:
        """Start the clock.

        Args:
            at: The instant.
        """
        self.at = at

    def now(self) -> int:
        """Read the clock.

        Returns:
            The current instant.
        """
        return self.at


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

    def test_a_claim_of_no_cores_is_refused(self) -> None:
        """Its workers would hold cores its host was never charged (MCPs board task 51e3eac5)."""
        encoded = encode_host_claim(_claim("25d07c7c", until_unix=1791453435))
        encoded["cores"] = 0

        with pytest.raises(JSONTypeError, match="host claim grants 2 workers on 0 cores"):
            decode_host_claim(encoded)


class TestTheTurn:
    def test_is_a_flock_on_a_lock_file_that_stays(self, tmp_path: pathlib.Path) -> None:
        lock = host_claims.lock_path(tmp_path, alias=HOST)

        with host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY):
            held = is_held(lock)

        assert {"lock": lock.name, "held inside": held, "held after": is_held(lock)} == {
            "lock": f"{HOST}.turn",
            "held inside": True,
            "held after": False,
        }

    def test_is_given_back_when_the_block_raises(self, tmp_path: pathlib.Path) -> None:
        with (
            pytest.raises(RuntimeError, match="the gate raised"),
            host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY),
        ):
            raise RuntimeError("the gate raised")

        assert not is_held(host_claims.lock_path(tmp_path, alias=HOST))

    def test_waits_for_the_other_process_s_turn_and_then_takes_it(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The second process asks while the runner claims, and claims after it."""
        holding, leave = threading.Event(), threading.Event()
        order: list[str] = []

        def ordinary_turn() -> None:
            with host_claims.claim_turn(tmp_path, alias=HOST, runner=ORDINARY):
                order.append(ORDINARY)
                holding.set()
                await_event(leave, what="the second process's ask")
                time.sleep(0.1)

        other = threading.Thread(target=ordinary_turn)
        other.start()
        await_event(holding, what="the ordinary runner's turn")
        leave.set()
        with (
            caplog.at_level("INFO"),
            host_claims.claim_turn(tmp_path, alias=HOST, runner=SECOND),
        ):
            order.append(SECOND)
        other.join()

        assert order == [ORDINARY, SECOND]
        said = [record.getMessage() for record in caplog.records]
        assert len(said) == 1
        assert re.fullmatch(
            rf"{SECOND} took its claim turn after waiting \d+ ms for another process claiming "
            "for its host",
            said[0],
        ), said


class TestTheClaims:
    def test_a_host_that_never_claimed_holds_none(self, tmp_path: pathlib.Path) -> None:
        assert host_claims.live_claims(tmp_path, alias=HOST) == ()
        assert host_claims.host_load(()) == LiveLoad(runs=0, workers=0, cores=0, ram_gb=0.0)

    def test_every_runners_claims_are_read_and_summed(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.now = FakeClock(at=1791449835).now
        first = _claim("25d07c7c", until_unix=1791453435)
        second = HostClaim(
            job_id="f65df8f5",
            runner=SECOND,
            project="MCPs/packages/wiki-check",
            workers=2,
            cores=2,
            ram_gb=1.0,
            until_unix=1791453505,
        )
        host_claims.record(tmp_path, alias=HOST, claim=second)
        host_claims.record(tmp_path, alias=HOST, claim=first)

        live = host_claims.live_claims(tmp_path, alias=HOST)

        assert live == (first, second)
        assert host_claims.host_load(live) == LiveLoad(
            runs=2, workers=4, cores=4, ram_gb=2 * 1.1 + 1.0
        )
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
