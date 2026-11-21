"""What retires freed, recorded per run and totalled per node (MCPs board task 1526ece1).

:mod:`fleet.contracts.freed_record` round-trips every line it writes and
refuses what it never writes; :mod:`fleet.core.records` keeps the lines in a
real file under ``tmp_path``; :mod:`fleet.core.freed_report` totals them per
node over a window, the sums worked by hand here; and
:mod:`fleet.cli.freed_report` posts that line on a task under the hub
runner's identity, the board answered by :class:`tests._queue_fakes.FakeQueue`.
"""

from __future__ import annotations

import itertools
import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str, load_json_str

from fleet.cli import _config, freed_report
from fleet.contracts.freed_record import FreedRecord, decode_freed_record, encode_freed_record
from fleet.core import _test_hooks, records, retire_owed
from fleet.core.freed_report import NodeFreed, node_totals, report_line
from tests._queue_fakes import FakeEnv, FakeQueue, queue_env

#: 2026-11-21T22:00:00Z, the window's start in these cases.
SINCE = 1_795_298_400

#: The task the report is posted on.
TASK = "1526ece1-497d-4e67-8b48-4aba8f789c40"


def _line(run_id: str, node: str, at_unix: int, freed_bytes: int) -> FreedRecord:
    """One freed line."""
    return FreedRecord(run_id=run_id, node=node, at_unix=at_unix, freed_bytes=freed_bytes)


class TestTheContract:
    @pytest.mark.parametrize(
        "record",
        [
            _line(run, node, at, freed)
            for run, node, at, freed in itertools.product(
                ("MCPs-packages-db-loki-1795300810", "r"),
                ("loki", "lavender-wsl"),
                (0, SINCE, 2**40),
                (0, 1, 1_234_567_890_123),
            )
        ],
    )
    def test_every_line_written_reads_back_as_it_was(self, record: FreedRecord) -> None:
        text = dump_json_str(encode_freed_record(record))
        assert decode_freed_record(load_json_str(text)) == record

    @pytest.mark.parametrize(
        ("value", "refusal"),
        [
            ([], "freed record must be a JSON object, got list"),
            (
                {"run_id": "r", "node": "loki", "at_unix": 1, "freed_bytes": -1},
                "freed record's freed_bytes must not be negative, got -1",
            ),
            ({"run_id": "r", "node": "loki", "at_unix": 1}, "freed_bytes"),
            ({"run_id": "r", "node": "loki", "freed_bytes": 1}, "at_unix"),
            ({"run_id": "r", "at_unix": 1, "freed_bytes": 1}, "node"),
            ({"node": "loki", "at_unix": 1, "freed_bytes": 1}, "run_id"),
            ({"run_id": "r", "node": "loki", "at_unix": 1, "freed_bytes": "9"}, "freed_bytes"),
        ],
    )
    def test_a_line_it_never_writes_is_refused_by_name(
        self, value: JSONValue, refusal: str
    ) -> None:
        with pytest.raises(JSONTypeError, match=refusal):
            decode_freed_record(value)

    def test_the_file_keeps_every_line_in_order_and_an_absent_one_reads_empty(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "runs" / retire_owed.FREED_FILE
        assert records.read_freed(path) == ()
        first, second = _line("a", "loki", SINCE, 10), _line("b", "sedona", SINCE + 1, 20)

        records.append_freed(path, first)
        records.append_freed(path, second)

        assert records.read_freed(path) == (first, second)
        assert retire_owed.freed_path(tmp_path / "runs" / retire_owed.RETIRES_FILE) == path


class TestTheTotals:
    def test_each_node_s_lines_in_the_window_are_summed_and_every_node_is_named(self) -> None:
        lines = (
            _line("old", "loki", SINCE - 1, 999),
            _line("a", "loki", SINCE, 1_000),
            _line("b", "sedona", SINCE + 60, 250),
            _line("c", "loki", SINCE + 120, 24),
            _line("d", "retired-node", SINCE + 180, 5),
        )

        totals = node_totals(lines, since_unix=SINCE, nodes=("sedona", "loki", "lavender"))

        # loki: 1000 + 24, its line before the window left out; lavender
        # retired nothing; a node outside the workspace is named after them.
        assert totals == (
            NodeFreed(node="sedona", runs=1, freed_bytes=250),
            NodeFreed(node="loki", runs=2, freed_bytes=1_024),
            NodeFreed(node="lavender", runs=0, freed_bytes=0),
            NodeFreed(node="retired-node", runs=1, freed_bytes=5),
        )

    def test_the_line_names_the_window_each_node_and_the_total(self) -> None:
        totals = (
            NodeFreed(node="sedona", runs=1, freed_bytes=250),
            NodeFreed(node="loki", runs=2, freed_bytes=1_024),
        )

        assert report_line(totals, since_unix=SINCE, source="/r/freed.jsonl") == (
            "STAGE FREED since 2026-11-21T22:00:00Z: sedona 1 run(s) 250 bytes; "
            "loki 2 run(s) 1024 bytes; total 3 run(s) 1274 bytes, from /r/freed.jsonl"
        )


def _hub_hostname() -> str:
    """The hub's hostname, as the hook reports it."""
    return "diphtheria"


class TestTheCommand:
    def test_posts_the_totals_on_the_task_as_the_hub_runner(
        self, config_path: pathlib.Path
    ) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        source = retire_owed.freed_path(loaded.retires)
        records.append_freed(source, _line("a", "lavender", SINCE + 5, 4_096))
        records.append_freed(source, _line("old", "lavender", SINCE - 5, 7))
        endpoint = FakeQueue(["posted"])
        _test_hooks.http_post = endpoint
        _test_hooks.env = queue_env()
        _test_hooks.hostname = _hub_hostname

        code = freed_report.main(
            [
                _config.CONFIG_FLAG,
                str(config_path),
                freed_report.SINCE_FLAG,
                str(SINCE),
                freed_report.TASK_FLAG,
                TASK,
            ]
        )

        assert code == 0
        assert endpoint.tools == ["task_post"]
        assert endpoint.arguments == [
            {
                "agent": "fleet-hub-diphtheria",
                "sessionId": "a8f33e7c-eaa8-5715-a18d-f4f2e5786d7d",
                "cwd": str(pathlib.Path.cwd()),
                "taskId": TASK,
                "kind": "note",
                "body": "STAGE FREED since 2026-11-21T22:00:00Z: lavender 1 run(s) 4096 bytes; "
                f"total 1 run(s) 4096 bytes, from {source}",
            }
        ]

    @pytest.mark.parametrize("since", ["yesterday", "-5", "1.5"])
    def test_a_since_that_is_not_whole_seconds_is_refused_before_anything_is_sent(
        self, config_path: pathlib.Path, since: str
    ) -> None:
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(ValueError, match="--since must be whole seconds since the epoch"):
            freed_report.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    freed_report.SINCE_FLAG,
                    since,
                    freed_report.TASK_FLAG,
                    TASK,
                ]
            )

        assert endpoint.tools == []

    def test_missing_credentials_are_refused_by_name_and_nothing_is_sent(
        self, config_path: pathlib.Path
    ) -> None:
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint
        _test_hooks.env = FakeEnv({})

        with pytest.raises(AppError) as raised:
            freed_report.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    freed_report.SINCE_FLAG,
                    str(SINCE),
                    freed_report.TASK_FLAG,
                    TASK,
                ]
            )

        assert raised.value.code is FleetErrorCode.QUEUE_CREDENTIALS_MISSING
        assert endpoint.tools == []

    def test_running_it_as_a_module_actually_posts(self, config_path: pathlib.Path) -> None:
        """Without its ``if __name__`` block the module would post nothing
        and exit 0, a report the task never received."""
        endpoint = FakeQueue(["posted"])
        _test_hooks.http_post = endpoint
        _test_hooks.env = queue_env()
        _test_hooks.hostname = _hub_hostname
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.freed_report", None)
        sys.argv = [
            "x",
            _config.CONFIG_FLAG,
            str(config_path),
            freed_report.SINCE_FLAG,
            str(SINCE),
            freed_report.TASK_FLAG,
            TASK,
        ]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.freed_report", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.freed_report"] = saved_module

        assert raised.value.code == 0
        assert endpoint.tools == ["task_post"]

    def test_run_under_any_other_name_it_posts_nothing(self) -> None:
        """Only ``__main__`` starts it: a name sorting before it, ``__a__``,
        runs the module's body and nothing else: no ``SystemExit`` from a
        parse of pytest's own argv, and nothing posted."""
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint
        saved_module = sys.modules.pop("fleet.cli.freed_report", None)
        try:
            runpy.run_module("fleet.cli.freed_report", run_name="__a__", alter_sys=False)
        finally:
            if saved_module is not None:
                sys.modules["fleet.cli.freed_report"] = saved_module

        assert endpoint.tools == []
