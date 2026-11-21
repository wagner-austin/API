"""A run a process still uses is kept whole, and its retire's bytes are recorded.

MCPs board task 1526ece1. The review that reopened it asked for a real
process holding a real stage directory: these cases start one (``sleep``
with its working directory inside the run's enclosure), and run the real
retire pass (:func:`fleet.core.retire_owed.retire_owed`) with the real retire
script, its ssh carried out on this machine by the real subprocess run, so
the script's ``/proc`` scan is what decides. A held lease sends nothing; a
lapsed lease with the process alive keeps the directory and the run owed;
the process ended, the next pass removes it and records the bytes it held,
counted here by summing every entry's apparent size, as ``du -sb`` does.
The scripted-answer cases beside them pin what each answer makes the record
say, and that an answer missing is a fault, never a retire.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from collections.abc import Generator, Sequence

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.freed_record import FreedRecord
from fleet.contracts.lease import Lease
from fleet.contracts.node import NodeConfig
from fleet.contracts.retire_record import RetireState
from fleet.core import _test_hooks, leases, names, records, retire_owed
from fleet.core.ssh_options import SSH_OPTIONS
from tests._retire_answers import RETIRED_BYTES, RETIRED_OUTPUT
from tests._toolchain_fixtures import node
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, ok, pin_clock, retire_replies

#: A second run on the same node.
OTHER_RUN_ID = "libs-demo-lavender-1757000100"

#: What the node prints when a process keeps the demo run whole.
HOLDER = f"/proc/4242/cwd /home/corvis/fleet/stage/{DEMO_RUN_ID}/{DEMO_RUN_ID}"
IN_USE_OUTPUT = f"in-use {HOLDER}\n"


def _on_this_machine(
    argv: Sequence[str],
    *,
    timeout_seconds: int,
    stdin_bytes: bytes | None = None,
    unset_env: Sequence[str] = (),
    set_env: Sequence[tuple[str, str]] = (),
) -> _test_hooks.CommandResult:
    """Carry out an ssh to the node here: its remote words, run by ``sh``.

    Everything after the destination is what ssh hands the far side's shell,
    so ``sh -c`` of those words joined is what the node runs.
    """
    words = list(argv)
    prefix = ["ssh", *SSH_OPTIONS]
    assert words[: len(prefix)] == prefix
    remote = words[len(prefix) + 1 :]
    return _test_hooks._default_run(
        ("sh", "-c", " ".join(remote)),
        timeout_seconds=timeout_seconds,
        stdin_bytes=stdin_bytes,
        unset_env=unset_env,
        set_env=set_env,
    )


def _apparent_bytes(root: pathlib.Path) -> int:
    """Every entry's apparent size under ``root``, itself included: ``du -sb``'s count.

    Args:
        root: The directory.

    Returns:
        The sum of ``st_size`` over it and everything beneath it.
    """
    total = root.lstat().st_size
    for parent, directories, files in os.walk(root):
        for name in (*directories, *files):
            total += (pathlib.Path(parent) / name).lstat().st_size
    return total


def _lease(run_id: str, *, expires: int) -> Lease:
    """A lease one run holds on the demo node."""
    return Lease(
        node="lavender",
        project="libs/demo",
        run_id=run_id,
        agent="opus-fleet-0904",
        session_id="acc774c0-3bc3-4cce-9dda-c7a12fb99519",
        acquired_unix=DEMO_NOW,
        expires_unix=expires,
        resources=(),
    )


def _pass(path: pathlib.Path, runs: pathlib.Path, at: NodeConfig) -> None:
    """Send the node's owed retires, the leases and ledger in ``runs``."""
    retire_owed.retire_owed(
        path,
        at,
        alias="lavender",
        leases_path=runs / "leases.json",
        ledger_path=runs / "ledger.jsonl",
    )


@pytest.fixture(name="holder")
def _holder(tmp_path: pathlib.Path) -> Generator[subprocess.Popen[bytes], None, None]:
    """A real process whose working directory is inside the demo run's export.

    Yields:
        The process, killed and reaped at the end if a case has not.
    """
    target = pathlib.Path(names.dispatch_directory((tmp_path / "stage").as_posix(), DEMO_RUN_ID))
    (target / "build").mkdir(parents=True)
    (target / "build" / "out.bin").write_bytes(b"x" * 70_000)
    (target / names.RESULT_NAME).write_text("0\n", encoding="utf-8")
    process = subprocess.Popen(["sleep", "300"], cwd=target / "build")
    yield process
    process.kill()
    process.wait(timeout=10)


class TestARealProcessKeepsItsRun:
    def test_kept_while_live_or_used_then_removed_and_its_bytes_recorded(
        self, tmp_path: pathlib.Path, holder: subprocess.Popen[bytes]
    ) -> None:
        stage = tmp_path / "stage"
        runs = tmp_path / "runs"
        path = runs / retire_owed.RETIRES_FILE
        at = node()
        at["stage_root"] = stage.as_posix()
        enclosure = pathlib.Path(names.run_enclosure(stage.as_posix(), DEMO_RUN_ID))
        clock = pin_clock(DEMO_NOW)
        leases.acquire(
            runs / "leases.json", _lease(DEMO_RUN_ID, expires=DEMO_NOW + 600), now_unix=DEMO_NOW
        )
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = _on_this_machine

        # The run is live: its lease is held, so nothing is sent.
        _pass(path, runs, at)
        assert (enclosure / DEMO_RUN_ID / "build" / "out.bin").is_file()

        # The lease lapsed, but the process is still in the run's tree: the
        # node keeps it whole and the run stays owed, nothing freed.
        clock.seconds = DEMO_NOW + 601
        _pass(path, runs, at)
        assert (enclosure / DEMO_RUN_ID / "build" / "out.bin").is_file()
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)
        assert records.read_freed(retire_owed.freed_path(path)) == ()

        # The process ended: the next pass removes the run and records what
        # its enclosure held.
        held = _apparent_bytes(enclosure)
        holder.kill()
        holder.wait(timeout=10)
        _pass(path, runs, at)
        assert not enclosure.exists()
        assert retire_owed.owed_on(path, alias="lavender") == ()
        assert records.read_freed(retire_owed.freed_path(path)) == (
            FreedRecord(
                run_id=DEMO_RUN_ID, node="lavender", at_unix=DEMO_NOW + 601, freed_bytes=held
            ),
        )
        assert held > 70_000


class TestEachAnswer:
    def test_a_retire_done_records_the_bytes_the_node_said_it_freed(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "runs" / retire_owed.RETIRES_FILE
        pin_clock(DEMO_NOW)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun(retire_replies()[:2])

        outcome = retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert outcome is retire_owed.RetireOutcome.ENDED
        assert records.read_freed(tmp_path / "runs" / "freed.jsonl") == (
            FreedRecord(
                run_id=DEMO_RUN_ID, node="lavender", at_unix=DEMO_NOW, freed_bytes=RETIRED_BYTES
            ),
        )

    def test_a_used_run_stays_owed_and_the_pass_goes_on_to_the_next(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "runs" / retire_owed.RETIRES_FILE
        pin_clock(DEMO_NOW)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        retire_owed.owe(path, alias="lavender", run_id=OTHER_RUN_ID)
        node_runs = FakeRun([ok(""), ok(IN_USE_OUTPUT), *retire_replies()])
        _test_hooks.run = node_runs

        _pass(path, tmp_path / "runs", node())

        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)
        assert [
            record["run_id"] for record in records.read_freed(retire_owed.freed_path(path))
        ] == [OTHER_RUN_ID]
        # Both retires were sent and run, then the generations' removal once.
        assert len(node_runs.calls) == 6

    def test_a_pass_whose_every_run_is_used_removes_no_generations(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "runs" / retire_owed.RETIRES_FILE
        pin_clock(DEMO_NOW)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        node_runs = FakeRun([ok(""), ok(IN_USE_OUTPUT)])
        _test_hooks.run = node_runs

        _pass(path, tmp_path / "runs", node())

        assert len(node_runs.calls) == 2
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)

    def test_a_stop_s_retire_of_a_used_run_raises_resource_held(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "runs" / retire_owed.RETIRES_FILE
        pin_clock(DEMO_NOW)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun([ok(""), ok(IN_USE_OUTPUT)])

        with pytest.raises(AppError) as raised:
            retire_owed.retire_or_raise(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert raised.value.code is FleetErrorCode.RESOURCE_HELD
        assert str(raised.value.message) == (
            f"retire {DEMO_RUN_ID} on lavender: kept whole and still owed, "
            f"a process uses it: {HOLDER}"
        )
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)

    def test_a_retire_that_prints_no_answer_is_recorded_failed(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A script that exited 0 without its last line did not run to its end."""
        path = tmp_path / "runs" / retire_owed.RETIRES_FILE
        pin_clock(DEMO_NOW)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun([ok(""), ok("moved\n")])

        with pytest.raises(AppError) as raised:
            retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        expected = (
            "lavender's retire ended without its answer: its last line was 'moved', "
            "not 'in-use <link> <target>' or 'freed <bytes>'"
        )
        assert raised.value.code is FleetErrorCode.DISPATCH_FAILED
        assert str(raised.value.message) == expected
        (*_, failed) = records.read_retires(path)
        assert (failed["state"], failed["detail"]) == (RetireState.FAILED, expected)
        assert records.read_freed(retire_owed.freed_path(path)) == ()

    def test_a_retire_the_stop_sends_records_its_bytes_too(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "runs" / retire_owed.RETIRES_FILE
        pin_clock(DEMO_NOW)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun([ok(""), ok(f"moved\n{RETIRED_OUTPUT}")])

        kept = retire_owed.retire_or_raise(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert kept == f"/home/corvis/fleet/stage/logs/{DEMO_RUN_ID}.log"
        (freed,) = records.read_freed(retire_owed.freed_path(path))
        assert freed["freed_bytes"] == RETIRED_BYTES
