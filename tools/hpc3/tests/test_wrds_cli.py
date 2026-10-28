"""Exercise the agent-facing command surface through the real orchestration."""

from __future__ import annotations

import pathlib
import runpy

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.cli import wrds
from hpc3.contracts.wrds_run import encode_wrds_run, encode_wrds_runtime, run_identity
from hpc3.core import _test_hooks
from tests.conftest import FakeRun
from tests.test_wrds_results import actual_result
from tests.wrds_support import example
from tests.wrds_transport import NOW, CopyTransport, completed, prepare


def test_inspect_and_preflight(
    tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    run = example(tmp_path)
    prepare(fake_run, run, tmp_path)
    assert (
        wrds.main(
            [
                "inspect",
                "--host",
                "wrds",
                "--root",
                run["root"] + "/tools",
                "--python",
                run["runtime"]["python"],
            ]
        )
        == 0
    )
    assert load_json_str(emitted[-1]) == encode_wrds_runtime(run["runtime"])
    spec = tmp_path / "experiment.json"
    spec.write_text(dump_json_str(encode_wrds_run(run)))
    assert wrds.main(["preflight", "--spec", str(spec)]) == 0
    assert load_json_str(emitted[-1]) == {"identity": run_identity(run), "preflight": "passed"}


def test_submit_resume_status_cancel_and_trace(tmp_path: pathlib.Path, emitted: list[str]) -> None:
    cli_hooks.now_iso = lambda: NOW
    run = example(tmp_path)
    spec = tmp_path / "experiment.json"
    spec.write_text(dump_json_str(encode_wrds_run(run)))
    flags = ["--spec", str(spec), "--attempt", "one"]
    fake = FakeRun()
    prepare(fake, run, tmp_path)
    _test_hooks.run = fake
    assert wrds.main(["submit", *flags]) == 0
    assert narrow_json_to_dict(load_json_str(emitted[-1]))["job_id"] == "123"
    assert wrds.main(["resume", *flags]) == 0
    assert wrds.main(["status", *flags]) == 0
    assert load_json_str(emitted[-1]) == {"job_id": "123", "state": "hqw"}
    assert wrds.main(["cancel", *flags]) == 0
    assert load_json_str(emitted[-1]) == {"state": "cancellation-pending"}
    ledger = tmp_path / "wrds-ledger.jsonl"
    with ledger.open("a") as stream:
        stream.write('{"identity":"other","attempt":"one"}\n')
    emitted.clear()
    assert wrds.main(["trace", *flags]) == 0
    assert len(emitted) == 6
    assert all(
        narrow_json_to_dict(load_json_str(line))["identity"] == run_identity(run)
        for line in emitted
    )


def test_collect_only_reports_metadata(tmp_path: pathlib.Path, emitted: list[str]) -> None:
    run, result = actual_result(tmp_path)
    transport = CopyTransport(tmp_path / "outputs")
    completed(transport, run, result)
    spec = tmp_path / "experiment.json"
    spec.write_text(dump_json_str(encode_wrds_run(run)))
    output = tmp_path / "download"
    assert (
        wrds.main(["collect", "--spec", str(spec), "--attempt", "one", "--output", str(output)])
        == 0
    )
    assert load_json_str(emitted[-1]) == {"output": str(output), "identity": run_identity(run)}


def test_command_boundaries(argv: list[str]) -> None:
    with pytest.raises(ValueError, match="WRDS_ACTION"):
        wrds.main([])
    with pytest.raises(ValueError, match="WRDS_ACTION"):
        wrds.main(["unknown"])
    argv[:] = ["wrds"]
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hpc3.cli.wrds", run_name="__main__")
    assert caught.value.code != 0
