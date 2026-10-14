"""The append turn of the ledger, the feed and the retire record (MCPs board task 8d96f89a).

Every node runner is its own process on the hub and all of them append to the
same three files. On 2026-10-14 two finish rows appended at once left the
ledger holding one row whole and fifteen bytes of the other, and every node
runner stopped on ``InvalidJsonError`` for 36 minutes. Each case runs against a
real record file and a real lock beside it: another process's turn is a lock
file this process did not create, and its append is what the case's sleep does
while this one waits.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable

import pytest
from platform_core.json_utils import InvalidJsonError, load_json_str

from fleet.contracts.feed import decode_feed_event
from fleet.contracts.ledger import LedgerEntry, decode_ledger_entry
from fleet.contracts.retire_record import RetireRecord, RetireState
from fleet.core import _test_hooks, records

_NOW = 1_791_971_310


def _row(run_id: str) -> LedgerEntry:
    """A finished ledger row of a run on lavender-wsl.

    Args:
        run_id: The dispatch.

    Returns:
        The row, typed through the ledger's own decoder.
    """
    return decode_ledger_entry(
        {
            "run_id": run_id,
            "node": "lavender-wsl",
            "host": "lavender-wsl",
            "project": "MCPs/packages/maketools",
            "agent": "opus-mcps-1014-34a8",
            "session_id": "34a80fbe-1a83-4cb4-b7ce-dd33f8d72215",
            "started_unix": _NOW - 120,
            "ended_unix": _NOW,
            "outcome": "passed",
            "exit_code": 0,
            "workers": 4,
            "detail": "",
        }
    )


#: One appender per record file, each appending a line naming ``run_id``.
APPENDERS: dict[str, Callable[[pathlib.Path, str], None]] = {
    "ledger": lambda path, run_id: records.append_ledger(path, _row(run_id)),
    "feed": lambda path, run_id: records.append_feed(
        path,
        decode_feed_event(
            {
                "at_unix": _NOW,
                "run_id": run_id,
                "node": "lavender-wsl",
                "project": "MCPs/packages/maketools",
                "kind": "started",
                "detail": "",
            }
        ),
    ),
    "retire record": lambda path, run_id: records.append_retire(
        path,
        RetireRecord(
            run_id=run_id, node="lavender-wsl", state=RetireState.OWED, at_unix=1, detail=""
        ),
    ),
}


def _run_ids(path: pathlib.Path) -> list[str]:
    """Every line's run id, each line parsed whole.

    Args:
        path: The record file.

    Returns:
        The run ids, in file order.
    """
    found: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = load_json_str(line)
        if not isinstance(value, dict):
            raise AssertionError(f"{path} holds a line that is not an object: {line!r}")
        run_id = value["run_id"]
        if not isinstance(run_id, str):
            raise AssertionError(f"{path} holds a line whose run_id is not a string: {line!r}")
        found.append(run_id)
    return found


@pytest.mark.parametrize("kind", sorted(APPENDERS))
def test_an_append_writes_inside_its_turn_and_leaves_no_lock(
    kind: str, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "record.jsonl"
    lock = records.append_lock_path(path)
    held_while_writing: list[bool] = []
    write = _test_hooks.append_text

    def append_text(path: pathlib.Path, line: str) -> None:
        held_while_writing.append(lock.exists())
        write(path, line)

    _test_hooks.append_text = append_text

    APPENDERS[kind](path, "run-1")

    assert {"held": held_while_writing, "lock left": lock.exists(), "rows": _run_ids(path)} == {
        "held": [True],
        "lock left": False,
        "rows": ["run-1"],
    }


@pytest.mark.parametrize("kind", sorted(APPENDERS))
def test_an_append_waits_out_another_process_appending_then_follows_its_whole_line(
    kind: str, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "record.jsonl"
    lock = records.append_lock_path(path)
    lock.write_text("fleet process 4242 pid 4242 turn other", encoding="utf-8")
    slept: list[int] = []

    def sleep(seconds: int) -> None:
        # The other process ends its turn, and the next append is made in a
        # turn of its own while this one is still waiting to look again.
        slept.append(seconds)
        lock.unlink()
        APPENDERS[kind](path, "run-other")

    created = int(os.stat(lock).st_mtime)
    _test_hooks.now = lambda: created
    _test_hooks.sleep = sleep

    APPENDERS[kind](path, "run-mine")

    assert {"slept": slept, "rows": _run_ids(path), "lock left": lock.exists()} == {
        "slept": [records.APPEND_TURN["poll_seconds"]],
        "rows": ["run-other", "run-mine"],
        "lock left": False,
    }


def test_a_lock_left_by_a_process_that_died_mid_append_is_taken_away(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    lock = records.append_lock_path(path)
    lock.write_text("fleet process 4242 pid 4242 turn dead", encoding="utf-8")
    created = int(os.stat(lock).st_mtime)
    _test_hooks.now = lambda: created + records.APPEND_TURN["stale_seconds"]
    slept: list[int] = []

    def sleep(seconds: int) -> None:
        slept.append(seconds)

    _test_hooks.sleep = sleep

    records.append_ledger(path, _row("run-after"))

    assert {"slept": slept, "rows": _run_ids(path), "lock left": lock.exists()} == {
        "slept": [],
        "rows": ["run-after"],
        "lock left": False,
    }


#: One appender process: it says it is ready, waits for the go file, then
#: appends its rows to the ledger through records.append_ledger. Run as a
#: script by the interpreter running the tests, so it imports this package.
_APPENDER = """
import pathlib, sys, time
from fleet.contracts.ledger import decode_ledger_entry
from fleet.core import records

record, go, name, rows = sys.argv[1:5]
pathlib.Path(record + "." + name + ".ready").write_text("ready", encoding="utf-8")
while not pathlib.Path(go).exists():
    time.sleep(0.005)
for index in range(int(rows)):
    records.append_ledger(pathlib.Path(record), decode_ledger_entry({
        "run_id": name + "-" + str(index), "node": "lavender-wsl", "host": "lavender-wsl",
        "project": "MCPs/packages/maketools", "agent": "opus-mcps-1014-34a8",
        "session_id": "34a80fbe-1a83-4cb4-b7ce-dd33f8d72215", "started_unix": 1,
        "ended_unix": 2, "outcome": "passed", "exit_code": 0, "workers": 4,
        "detail": "x" * 300,
    }))
"""

#: How many appender processes run at once, and how many rows each appends.
#: Measured on austinpc on 2026-10-14 with the bare append and no turn, eight
#: processes of 200 rows wrote 1170, 1200 and 1152 of their 1600 lines, 0, 9
#: and 8 of them torn; through the turn eight of ten wrote all 80, none torn,
#: in 7.0 s. This case is sized to stay a few seconds long.
_PROCESSES = 4
_ROWS = 5

#: How long the case waits for every appender to start and to finish.
_PROCESS_TIMEOUT_SECONDS = 120


def test_appender_processes_running_at_once_leave_every_row_whole(tmp_path: pathlib.Path) -> None:
    child = tmp_path / "appender.py"
    child.write_text(_APPENDER, encoding="utf-8")
    record = tmp_path / "ledger.jsonl"
    go = tmp_path / "go"
    names = [f"p{n}" for n in range(_PROCESSES)]
    started = [
        subprocess.Popen([sys.executable, str(child), str(record), str(go), name, str(_ROWS)])
        for name in names
    ]
    deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
    while not all(record.with_name(f"{record.name}.{name}.ready").exists() for name in names):
        if time.monotonic() > deadline:
            raise AssertionError(f"not every appender started within {_PROCESS_TIMEOUT_SECONDS} s")
        time.sleep(0.01)
    go.write_text("go", encoding="utf-8")

    exits = [process.wait(timeout=_PROCESS_TIMEOUT_SECONDS) for process in started]

    assert {"exits": exits, "rows": sorted(_run_ids(record))} == {
        "exits": [0] * _PROCESSES,
        "rows": sorted(f"{name}-{index}" for name in names for index in range(_ROWS)),
    }


def test_a_torn_line_is_refused_naming_its_file_and_line(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ledger.jsonl"
    records.append_ledger(path, _row("run-whole"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('91971182.log"}\n')

    with pytest.raises(InvalidJsonError) as refused:
        records.read_ledger(path)

    assert {"message": str(refused.value), "cause": type(refused.value.__cause__)} == {
        "message": f"{path} line 2 is not JSON: '91971182.log\"}}'",
        "cause": InvalidJsonError,
    }


def test_the_lock_sits_beside_its_record() -> None:
    record = pathlib.Path("runs") / "ledger.jsonl"

    assert records.append_lock_path(record) == pathlib.Path("runs") / "ledger.jsonl.lock"
