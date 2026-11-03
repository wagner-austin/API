"""Encrypted qualification runs: lineage before keys, train-validation only, no plaintext left."""

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
    require_int,
    require_list,
)
from platform_core.run_record import decode_run_record, run_record_sidecar
from platform_core.testing import sample_run_fingerprint
from scripts import crsp_qualification

from covenant_ml.markets.crsp_qualification_run import (
    decode_crsp_qualification_request,
    encode_crsp_qualification_request,
    execute_crsp_qualification,
)
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import study_protocol, study_rows
from ._encrypted_fixtures import create_access, evaluator_keys, seal_extract
from .test_crsp_qualification import open_suspension_rows

CODE = sha256(Path(crsp_qualification.__file__).read_bytes()).hexdigest()


def sealed_request(root: Path, rows: dict[str, list[JSONObject]]) -> Path:
    """Seal rows for train-validation, delete the final key and write the request."""
    source = root / "source"
    source.mkdir()
    manifest, _ = write_sources(source, rows, study_protocol())
    access = create_access(root)
    bundle = seal_extract(manifest, root / "encrypted", access, "train-validation")
    (root / "keys" / "final.agekey").unlink()
    request: JSONObject = {
        "design": study_protocol(),
        "manifest": str(bundle),
        "dataset_digest": sha256(bundle.read_bytes()).hexdigest(),
        "code_digest": CODE,
        "image_digest": "d" * 64,
        "label": "v5-qualification",
    }
    path = root / "request.json"
    path.write_text(dump_json_str(request))
    return path


@pytest.fixture()
def image_environment() -> Generator[None, None, None]:
    """Declare the synthetic runtime identity and restore the shared seam."""
    previous = config_test_hooks.get_env
    config_test_hooks.get_env = lambda key: "d" * 64 if key == "IMAGE_DIGEST" else None
    yield
    config_test_hooks.get_env = previous


def test_request_codec_round_trips_and_refuses_unknown_fields(tmp_path: Path) -> None:
    document = load_json_str(sealed_request(tmp_path, study_rows()).read_text())
    request = decode_crsp_qualification_request(narrow_json_to_dict(document))
    assert encode_crsp_qualification_request(request) == document
    narrow_json_to_dict(document)["extra"] = 1
    with pytest.raises(ValueError, match="CRSP_QUALIFICATION_REQUEST"):
        decode_crsp_qualification_request(narrow_json_to_dict(document))


def test_entrypoint_publishes_counts_without_the_final_key(
    tmp_path: Path, image_environment: None, capsys: pytest.CaptureFixture[str]
) -> None:
    request = sealed_request(tmp_path, study_rows())
    output = tmp_path / "result" / "qualification.json"
    argv = ["--request", str(request), "--out", str(output)]
    argv += ["--access", str(tmp_path / "access.json"), "--scratch", str(tmp_path)]
    assert crsp_qualification.main(argv, stand_in_pin) == 0
    payload = narrow_json_to_dict(load_json_str(output.read_text()))
    counts = require_dict(payload, "counts")
    assert (payload["kind"], counts["refused"], payload["results"]) == (
        "crsp-qualification-v1",
        0,
        [],
    )
    record = decode_run_record(load_json_str(run_record_sidecar(output).read_text()))
    assert record["label"] == "v5-qualification"
    assert record["experiment"] == "crsp-qualification-v5"
    observed = {item["name"]: item["value"] for item in record["observations"]}
    assert observed["qualification.refused"] == 0.0
    assert observed["qualification.paths"] == float(require_int(counts, "paths"))
    assert list(tmp_path.glob("research-plaintext-*")) == []
    assert capsys.readouterr().out.splitlines()[-2:] == [
        f"manifest -> {output}",
        f"run record -> {run_record_sidecar(output)}",
    ]


def test_real_process_publishes_an_open_suspension_refusal(tmp_path: Path) -> None:
    request = sealed_request(tmp_path, open_suspension_rows())
    output = tmp_path / "result" / "qualification.json"
    environment = dict(config_test_hooks.get_environment())
    environment["IMAGE_DIGEST"] = "d" * 64
    command = [sys.executable, "-m", "scripts.crsp_qualification", "--request", str(request)]
    command += ["--out", str(output), "--access", str(tmp_path / "access.json")]
    command += ["--scratch", str(tmp_path)]
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert (result.returncode, result.stderr) == (0, "")
    assert list(tmp_path.glob("research-plaintext-*")) == []
    assert not (tmp_path / "keys" / "final.agekey").exists()
    payload = narrow_json_to_dict(load_json_str(output.read_text()))
    assert require_dict(payload, "counts")["reasons"] == {"price gap tail": 1}
    (refused,) = [narrow_json_to_dict(item) for item in require_list(payload, "results")]
    assert (refused["period"], refused["reason"]) == ("validation", "price gap tail")
    # Published metadata carries no security identifier, date, price or return.
    assert set(refused) == {
        "ordinal",
        "decision",
        "period",
        "reason",
        "entry_sha256",
        "exit_sha256",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("code_digest", "e" * 64), ("image_digest", "e" * 64), ("bootstrap_seed", 8)],
)
def test_lineage_is_refused_before_any_result_is_written(
    tmp_path: Path, field: str, value: str | int
) -> None:
    path = sealed_request(tmp_path, study_rows())
    document = narrow_json_to_dict(load_json_str(path.read_text()))
    if field == "bootstrap_seed":
        require_dict(document, "design")[field] = value
    else:
        document[field] = value
    request = decode_crsp_qualification_request(document)
    output = tmp_path / "result" / "qualification.json"
    with pytest.raises(ValueError, match="CRSP_QUALIFICATION_LINEAGE"):
        execute_crsp_qualification(
            request,
            sample_run_fingerprint(image_digest="d" * 64),
            CODE,
            output,
            keys=evaluator_keys(tmp_path),
            scratch=tmp_path,
        )
    assert not output.parent.exists()
    assert list(tmp_path.glob("research-plaintext-*")) == []


def test_entrypoint_refuses_a_late_numerical_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.crsp_qualification", run_name="__main__")
