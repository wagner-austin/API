"""One frozen realized-path study run, end to end on a real sealed synthetic extract."""

from __future__ import annotations

import runpy
import subprocess
import sys
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from platform_core.run_record import decode_run_record, run_record_sidecar
from platform_core.testing import sample_run_fingerprint
from scripts import asian_option_study

from covenant_ml.options.crsp_paths import decode_path_counts
from covenant_ml.options.runner import (
    ERR_INPUT_DIGEST,
    EXPERIMENT,
    AsianStudyRequest,
    decode_request,
    encode_request,
    execute_asian_study,
    request_digest,
)
from covenant_ml.options.types import AsianPutContract, decode_study_result
from tests.benchmarking._cli_fixtures import stand_in_pin

from ..markets._crsp_fixtures import write_sources
from ..markets._crsp_study_fixtures import study_protocol, study_rows
from ..markets._encrypted_fixtures import create_access, evaluator_keys, seal_extract

MONTHLY = AsianPutContract(
    spot=100.0, strike=100.0, years=0.08, exercise_dates=4, sessions_per_date=5
)


def _request(root: Path) -> AsianStudyRequest:
    """Seal the committed synthetic extract and pin a request to it."""
    source = root / "source"
    source.mkdir()
    path, _ = write_sources(source, study_rows(), study_protocol())
    encrypted = seal_extract(path, root / "encrypted", create_access(root), "train-validation")
    rates = root / "dgs1.csv"
    lines = [
        f"{year}-{month:02d}-01,{month / 4}" for year in range(2016, 2021) for month in range(1, 13)
    ]
    rates.write_text("observation_date,DGS1\n" + "\n".join(lines) + "\n")
    return AsianStudyRequest(
        bundle=str(encrypted),
        bundle_sha256=sha256(encrypted.read_bytes()).hexdigest(),
        rates=str(rates),
        rates_sha256=sha256(rates.read_bytes()).hexdigest(),
        rates_series="DGS1",
        first_decision="2017-01-01",
        last_decision="2019-12-31",
        contract=MONTHLY,
        training_rate=0.02,
        training_volatility=0.3,
        training_paths=2_000,
        training_seed=5,
        block_length=3,
        n_resamples=50,
        seed=7,
        confidence=0.9,
    )


@pytest.fixture()
def image_environment() -> Generator[None, None, None]:
    """Declare the synthetic runtime identity and restore the shared seam."""
    previous = config_test_hooks.get_env
    config_test_hooks.get_env = lambda key: "d" * 64 if key == "IMAGE_DIGEST" else None
    yield
    config_test_hooks.get_env = previous


def test_requests_roundtrip_and_are_identified_by_their_content(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert decode_request(encode_request(request)) == request
    changed = AsianStudyRequest(**{**request, "seed": 8})
    assert request_digest(request) != request_digest(changed)
    saved = encode_request(request)
    saved["rates_sha256"] = "ABC"
    with pytest.raises(ValueError, match="CAMPAIGN-001"):
        decode_request(saved)


def test_a_run_publishes_its_manifest_and_record(tmp_path: Path) -> None:
    request = _request(tmp_path)
    output = tmp_path / "run" / "result.json"
    fingerprint = sample_run_fingerprint()
    record_path = execute_asian_study(
        request, fingerprint, "c" * 64, output, keys=evaluator_keys(tmp_path), scratch=tmp_path
    )
    manifest = narrow_json_to_dict(load_json_str(output.read_text()))
    assert manifest["kind"] == "asian-option-realized-paths-v1"
    assert manifest["request_sha256"] == request_digest(request)
    assert manifest["code_sha256"] == "c" * 64
    counts = decode_path_counts(manifest["paths"])
    assert (counts["decisions"], counts["terminal_contracts"]) == (36, 1)
    assert counts["excluded"] == {}
    (encoded,) = require_list(manifest, "results")
    result = decode_study_result(encoded)
    assert result["contracts"] == counts["contracts"]
    assert (result["first_month"], result["last_month"]) == ("2017-01", "2019-12")
    assert require_dict(manifest, "dataset")["bundle_sha256"] == request["bundle_sha256"]
    record = decode_run_record(load_json_str(record_path.read_text()))
    assert record["experiment"] == EXPERIMENT
    assert record["label"] == request_digest(request)[:12]
    # The shared record keeps observations in name order.
    names = [observation["name"] for observation in record["observations"]]
    assert names == [
        "difference",
        "difference.ci_high",
        "difference.ci_low",
        "realized_hold_mean",
        "realized_policy_mean",
        "simulated_value",
    ]
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("research-plaintext-")]


def test_a_changed_rate_download_is_refused_before_any_data_is_read(tmp_path: Path) -> None:
    request = _request(tmp_path)
    Path(request["rates"]).write_text("observation_date,DGS1\n2016-01-01,9\n")
    output = tmp_path / "run" / "result.json"
    with pytest.raises(ValueError, match=ERR_INPUT_DIGEST):
        execute_asian_study(
            request,
            sample_run_fingerprint(),
            "c" * 64,
            output,
            keys=evaluator_keys(tmp_path),
            scratch=tmp_path,
        )
    assert not output.parent.exists()


def test_a_run_is_never_overwritten(tmp_path: Path) -> None:
    request = _request(tmp_path)
    output = tmp_path / "run" / "result.json"
    output.parent.mkdir()
    with pytest.raises(FileExistsError):
        execute_asian_study(
            request,
            sample_run_fingerprint(),
            "c" * 64,
            output,
            keys=evaluator_keys(tmp_path),
            scratch=tmp_path,
        )


def test_the_entrypoint_measures_its_source_and_runs_once(
    tmp_path: Path, image_environment: None, capsys: pytest.CaptureFixture[str]
) -> None:
    request = tmp_path / "request.json"
    document: JSONObject = encode_request(_request(tmp_path))
    request.write_text(dump_json_str(document))
    output = tmp_path / "run" / "result.json"
    argv = ["--request", str(request), "--out", str(output)]
    argv += ["--access", str(tmp_path / "access.json"), "--scratch", str(tmp_path)]
    assert asian_option_study.main(argv, stand_in_pin) == 0
    manifest = narrow_json_to_dict(load_json_str(output.read_text()))
    source = Path(asian_option_study.__file__).read_bytes()
    assert manifest["code_sha256"] == sha256(source).hexdigest()
    record = decode_run_record(load_json_str(run_record_sidecar(output).read_text()))
    assert record["fingerprint"]["image_digest"] == "d" * 64
    assert capsys.readouterr().out.splitlines() == [
        f"manifest -> {output}",
        f"run record -> {run_record_sidecar(output)}",
    ]


def test_the_entrypoint_refuses_a_late_numerical_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.asian_option_study", run_name="__main__")


def test_a_real_process_reports_a_missing_request(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.asian_option_study",
            "--request",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "run" / "out.json"),
            "--access",
            str(tmp_path / "access.json"),
            "--scratch",
            str(tmp_path),
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
    assert not (tmp_path / "run").exists()
