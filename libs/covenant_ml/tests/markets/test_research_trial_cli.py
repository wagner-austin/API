"""Compute CLI uses real files and checks its own source and numerical startup."""

import runpy
import subprocess
import sys
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str, load_json_str, require_dict
from platform_core.run_record import decode_run_record, run_record_sidecar
from scripts import research_trial

from covenant_ml.markets.research_final import select_candidate
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._research_final_fixtures import completed_study, write_trial
from ._trial_fixtures import trial_document


@pytest.fixture()
def image_environment() -> Generator[None, None, None]:
    """Declare the synthetic runtime identity and restore the shared seam."""
    previous = config_test_hooks.get_env
    config_test_hooks.get_env = lambda key: "d" * 64 if key == "IMAGE_DIGEST" else None
    yield
    config_test_hooks.get_env = previous


def test_compute_cli_measures_its_entrypoint_source(
    tmp_path: Path, image_environment: None, capsys: pytest.CaptureFixture[str]
) -> None:
    document = trial_document(tmp_path)
    require_dict(document, "policy")["code_digest"] = sha256(
        Path(research_trial.__file__).read_bytes()
    ).hexdigest()
    request = tmp_path / "request.json"
    request.write_text(dump_json_str(document))
    output = tmp_path / "trial" / "result.json"
    assert research_trial.main(["--request", str(request), "--out", str(output)], stand_in_pin) == 0
    record = decode_run_record(load_json_str(run_record_sidecar(output).read_text()))
    assert record["label"] == "1"
    assert record["fingerprint"]["image_digest"] == "d" * 64
    assert capsys.readouterr().out.splitlines()[-2:] == [
        f"manifest -> {output}",
        f"run record -> {run_record_sidecar(output)}",
    ]


def test_final_mode_scores_the_selected_attempt_once(
    tmp_path: Path, image_environment: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code = sha256(Path(research_trial.__file__).read_bytes()).hexdigest()
    ledger, trial, manifest = completed_study(tmp_path, code)
    select_candidate(ledger, trial, manifest)
    request = write_trial(trial, tmp_path / "request.json")
    output = tmp_path / "score" / "result.json"
    final = ["--ledger", str(ledger), "--final-manifest", str(manifest)]
    call = ["--request", str(request), "--out", str(output), *final]
    assert research_trial.main(call, stand_in_pin) == 0
    record = decode_run_record(load_json_str(run_record_sidecar(output).read_text()))
    assert record["label"] == "final-1"
    assert capsys.readouterr().out.splitlines()[-1] == f"run record -> {run_record_sidecar(output)}"
    again = ["--request", str(request), "--out", str(tmp_path / "again" / "r.json"), *final]
    with pytest.raises(ValueError, match="RESEARCH-009"):
        research_trial.main(again, stand_in_pin)


@pytest.mark.parametrize("flag", ["--ledger", "--final-manifest"])
def test_final_mode_needs_both_its_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    call = ["--request", str(tmp_path / "unread.json"), "--out", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit) as stopped:
        research_trial.main([*call, flag, str(tmp_path / "x")], stand_in_pin)
    assert stopped.value.code == 2
    _, message = capsys.readouterr().err.splitlines()[-1].split(": error: ")
    assert message == "--ledger and --final-manifest are required together"


def test_entrypoint_refuses_a_late_numerical_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.research_trial", run_name="__main__")


def test_real_compute_process_reports_missing_request(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research_trial",
            "--request",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "result" / "out.json"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert result.stderr.splitlines()[-1] == (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        f"{str(tmp_path / 'missing.json')!r}"
    )
    assert not (tmp_path / "result").exists()
