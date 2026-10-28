"""Execute real payloads and the shipped zipapp, including failure paths."""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys
from collections.abc import Generator

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict, require_dict
from platform_core.run_record import decode_run_record

from hpc3.contracts.wrds_run import WrdsFile, WrdsRuntime, execution_document
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.wrds_bundle import worker_bundle
from hpc3.wrds import _test_hooks as hooks
from hpc3.wrds.worker import check_inputs, check_runtime, checked_file, execute_run, main
from tests.wrds_support import PAYLOAD, example


def job_environment() -> dict[str, str]:
    """Provide the minimal scheduler environment for local worker tests.

    Returns:
        A numeric job id and a stale image variable the launcher must remove.
    """
    return {"JOB_ID": "123", "IMAGE_DIGEST": "must-not-leak"}


@pytest.fixture(autouse=True)
def worker_effects() -> Generator[None, None, None]:
    hooks.reset_hooks()
    hooks.environment = job_environment
    yield
    hooks.reset_hooks()


def test_real_worker_repeats_and_emits_shared_run_record(tmp_path: pathlib.Path) -> None:
    records = []
    for attempt in ("one", "two"):
        directory = tmp_path / attempt
        directory.mkdir()
        run = example(directory)
        assert main(["run", str(directory / "run.json"), attempt]) == 0
        result = narrow_json_to_dict(load_json_str((directory / "completion.json").read_text()))
        record = decode_run_record(require_dict(result, "record"))
        assert record["fingerprint"]["image_digest"] == ""
        assert record["experiment"] == run["project"]
        assert record["observations"][0]["name"] == "sum"
        assert (directory / "payload.err").read_bytes() == b""
        records.append(record)
    assert records[0]["payload_digest"] == records[1]["payload_digest"]
    assert records[0]["observations"] == records[1]["observations"]
    with pytest.raises(FileExistsError):
        execute_run(tmp_path / "one" / "run.json", "one")


def test_bundle_is_deterministic_and_runs_without_installed_hpc3(tmp_path: pathlib.Path) -> None:
    payload = worker_bundle()
    assert payload == worker_bundle()
    archive = tmp_path / "worker.pyz"
    archive.write_bytes(payload)
    reply = core_hooks._default_run([sys.executable, "-s", str(archive), "inspect"])
    assert reply["returncode"] == 0
    result = narrow_json_to_dict(load_json_str(reply["stdout"]))
    assert result["sha256"] == hooks._default_runtime()["sha256"]
    failed = core_hooks._default_run([sys.executable, "-s", str(archive), "invalid"])
    assert failed["returncode"] != 0
    assert "WRDS_WORKER_ARGS" in failed["stderr"]


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("raise SystemExit(17)\n", "WRDS_PAYLOAD_FAILED"),
        ("pass\n", "missing"),
        (PAYLOAD + "(root/'metrics.json').write_text('[]')\n", "WRDS_METRICS_EMPTY"),
        (
            PAYLOAD + '(root/\'metrics.json\').write_text(\'[{"name":"bad","value":NaN}]\')\n',
            "WRDS_METRICS_NONFINITE",
        ),
        (PAYLOAD + "pathlib.Path(__file__).write_text('changed')\n", "WRDS_INPUT_DRIFT"),
    ],
)
def test_real_payload_failures_never_attest_success(
    tmp_path: pathlib.Path, source: str, error: str
) -> None:
    example(tmp_path, source)
    if error == "missing":
        with pytest.raises(FileNotFoundError):
            execute_run(tmp_path / "run.json", "one")
    else:
        with pytest.raises(ValueError, match=error):
            execute_run(tmp_path / "run.json", "one")
    assert not (tmp_path / "completion.json").exists()


def test_real_process_timeout_and_nonzero_exit(tmp_path: pathlib.Path) -> None:
    code = hooks._default_execute([sys.executable, "-c", "raise SystemExit(7)"], tmp_path, {}, 5)
    assert code == 7
    with pytest.raises(subprocess.TimeoutExpired):
        hooks._default_execute(
            [sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, {}, 1
        )
    assert (tmp_path / "payload.out").read_bytes() == b""


def test_input_bytes_and_symlink_escape(tmp_path: pathlib.Path) -> None:
    run = example(tmp_path)
    data = tmp_path / "data"
    data.write_bytes(b"a declared input")
    run["inputs"] = [
        WrdsFile(path=str(data), name="table", sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    ]
    (tmp_path / "run.json").write_text(dump_json_str(execution_document(run)))
    execute_run(tmp_path / "run.json", "one")
    assert (tmp_path / "completion.json").exists()
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="WRDS_INPUT_DRIFT"):
        check_inputs(run, tmp_path)
    (tmp_path / "code" / "main.py").unlink()
    (tmp_path / "code" / "main.py").symlink_to(data)
    with pytest.raises(ValueError, match="code symlink"):
        check_inputs(run, tmp_path)


def test_output_symlink_escape(tmp_path: pathlib.Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("[]")
    source = (
        "import pathlib,os\nroot=pathlib.Path(os.environ['WRDS_OUTPUT_DIR'])\n"
        + f"(root/'metrics.json').symlink_to({str(outside)!r})\n"
    )
    example(tmp_path, source)
    with pytest.raises(ValueError, match="WRDS_OUTPUT_ESCAPE"):
        execute_run(tmp_path / "run.json", "one")
    assert not (tmp_path / "completion.json").exists()


def test_runtime_drift_and_bad_job_id(tmp_path: pathlib.Path) -> None:
    run = example(tmp_path)
    run["runtime"]["version"] = "wrong"
    with pytest.raises(ValueError, match="WRDS_RUNTIME_DRIFT"):
        check_runtime(run)

    def bad_job() -> dict[str, str]:
        return {"JOB_ID": "invalid"}

    hooks.environment = bad_job
    with pytest.raises(ValueError, match="WRDS_JOB_ID"):
        execute_run(tmp_path / "run.json", "one")


def test_runtime_recheck_refuses_changed_environment(tmp_path: pathlib.Path) -> None:
    run = example(tmp_path)
    calls = 0

    def changed_runtime() -> WrdsRuntime:
        nonlocal calls
        calls += 1
        value = hooks._default_runtime()
        if calls == 2:
            value["version"] = "changed"
        return value

    hooks.runtime = changed_runtime
    with pytest.raises(ValueError, match="WRDS_RUNTIME_DRIFT"):
        execute_run(tmp_path / "run.json", "one")
    assert calls == 2
    assert run["name"] == "seeded-sum"
    assert not (tmp_path / "completion.json").exists()


def test_io_effects_fail_without_overwriting(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "artifact"
    hooks._default_write(path, b"one")
    assert hooks._default_read(path) == b"one"
    assert (
        checked_file(path, hashlib.sha256(b"one").hexdigest()) == hashlib.sha256(b"one").hexdigest()
    )
    with pytest.raises(FileExistsError):
        hooks._default_write(path, b"two")
    with pytest.raises(FileNotFoundError):
        hooks._default_digest(tmp_path / "absent")
    directory = tmp_path / "new"
    hooks._default_new_directory(directory)
    with pytest.raises(FileExistsError):
        hooks._default_new_directory(directory)
    assert directory.is_dir()
    assert hooks._default_environment() == dict(os.environ)


def test_worker_argument_boundaries(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    assert main(["inspect"]) == 0
    assert '"python"' in capsys.readouterr().out
    with pytest.raises(ValueError, match="WRDS_WORKER_ARGS"):
        main([])


def test_runtime_refuses_broken_distribution_metadata(tmp_path: pathlib.Path) -> None:
    distribution = tmp_path / "broken_wrds_pin-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Metadata-Version: 2.1\nVersion: 1.0\n")
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ValueError, match="distribution has no Name"):
            hooks._default_runtime()
    finally:
        sys.path.remove(str(tmp_path))


def test_worker_uses_process_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    previous = sys.argv
    sys.argv = ["wrds-worker", "inspect"]
    try:
        assert main() == 0
        assert '"python"' in capsys.readouterr().out
    finally:
        sys.argv = previous
