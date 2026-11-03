"""Real encrypted monthly workers distinguish closed suspension from an open tail."""

import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)
from platform_core.run_record import decode_run_record, run_record_sidecar
from scripts import research_trial

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import missing_price_rows, study_protocol
from ._encrypted_fixtures import create_access, seal_extract
from ._trial_fixtures import frozen_request


@pytest.mark.parametrize("closed", [True, False])
def test_real_worker_scores_closed_suspension_or_refuses_and_cleans_an_open_tail(
    tmp_path: Path, closed: bool
) -> None:
    rows = missing_price_rows(price_flag="SU", gap=9)
    if not closed:
        for row in rows["returns"]:
            if row["permno"] == 7 and "2019-03-11" <= require_str(row, "dlycaldt") <= "2019-04-01":
                row.update(
                    dlyret=None,
                    dlyprevdt=None,
                    dlyretmissflg="MP",
                    dlyretdurflg="MR",
                    dlyprc=None,
                    dlyprcflg="SU",
                )
    source = tmp_path / "source"
    source.mkdir()
    manifest, _ = write_sources(source, rows, study_protocol())
    access = create_access(tmp_path)
    bundle = seal_extract(manifest, tmp_path / "encrypted", access, "train-validation")
    (tmp_path / "keys" / "final.agekey").unlink()
    evaluation: JSONObject = {
        "design": study_protocol(),
        "stopping_months": 6,
        "fold_months": 12,
        "confidence": 0.9,
    }
    document = frozen_request(evaluation, bundle, ["momentum", "liquidity"])
    require_dict(document, "policy")["code_digest"] = sha256(
        Path(research_trial.__file__).read_bytes()
    ).hexdigest()
    request = tmp_path / "request.json"
    request.write_text(dump_json_str(document))
    output = tmp_path / "result" / "evaluation.json"
    environment = dict(config_test_hooks.get_environment())
    environment["IMAGE_DIGEST"] = "d" * 64
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research_trial",
            "--request",
            str(request),
            "--out",
            str(output),
            "--access",
            str(tmp_path / "access.json"),
            "--scratch",
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert list(tmp_path.glob("research-plaintext-*")) == []
    assert not (tmp_path / "keys" / "final.agekey").exists()
    if closed:
        assert (result.returncode, result.stderr) == (0, "")
        assert result.stdout.splitlines() == [
            f"manifest -> {output}",
            f"run record -> {run_record_sidecar(output)}",
        ]
        record = decode_run_record(load_json_str(run_record_sidecar(output).read_text()))
        assert record["label"] == "1"
        assert record["fingerprint"]["image_digest"] == "d" * 64
        payload = narrow_json_to_dict(load_json_str(output.read_text()))
        assert payload["kind"] == "research-monthly-validation-v2"
        (arm,) = [
            narrow_json_to_dict(item)
            for item in require_list(payload, "results")
            if narrow_json_to_dict(item)["arm"] == "equal_weight"
            and narrow_json_to_dict(item)["cost_bps"] == 10
        ]
        (month,) = [
            narrow_json_to_dict(item)
            for item in require_list(arm, "months")
            if narrow_json_to_dict(item)["decision"] == "2019-02-28"
        ]
        assert month["selected"] == [3, 4, 5, 6, 7, 8]
        assert month["unfilled"] == [8]
        assert len(require_list(month, "held")) == 5
    else:
        assert result.returncode == 1
        assert result.stdout == ""
        assert result.stderr.splitlines()[-1] == (
            "ValueError: [CRSP-011] held return for 7 ends with an unresolved price gap"
        )
        assert not output.exists()
        assert not run_record_sidecar(output).exists()
