"""Real validation evaluation, strict lineage and no reuse of an attempt output."""

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)
from platform_core.run_record import decode_run_record
from platform_core.testing import sample_run_fingerprint

from covenant_ml.markets.research_trial import execute_trial
from covenant_ml.markets.research_trial_records import (
    decode_research_evaluation,
    decode_research_trial,
    encode_research_evaluation,
    encode_research_trial,
)

from ._trial_fixtures import trial_document


def test_real_trainers_emit_the_admitted_shared_record(tmp_path: Path) -> None:
    document = trial_document(tmp_path)
    trial = decode_research_trial(document)
    assert encode_research_trial(trial) == document
    assert encode_research_evaluation(trial["evaluation"]) == document["evaluation"]
    output = tmp_path / "attempt-1" / "result.json"
    record_path = execute_trial(
        trial, sample_run_fingerprint(image_digest="d" * 64), "c" * 64, output
    )
    record = decode_run_record(load_json_str(record_path.read_text()))
    result = narrow_json_to_dict(load_json_str(output.read_text()))
    assert (record["experiment"], record["label"]) == (trial["policy"]["protocol_digest"], "1")
    assert len(record["observations"]) == 13
    assert result["partition"] == "validation"
    assert result["attempt"] == 1
    assert result["kind"] == "research-validation-diagnostic-v1"
    with pytest.raises(FileExistsError):
        execute_trial(trial, sample_run_fingerprint(image_digest="d" * 64), "c" * 64, output)
    assert decode_run_record(load_json_str(record_path.read_text())) == record


@pytest.mark.parametrize("field", ["protocol", "code", "image", "dataset"])
def test_worker_rechecks_frozen_identities_before_fitting(tmp_path: Path, field: str) -> None:
    document = trial_document(tmp_path)
    if field == "protocol":
        require_dict(document, "evaluation")["inner_train_fraction"] = 0.7
    if field == "dataset":
        require_dict(document, "policy")["dataset_digest"] = "e" * 64
    fingerprint = sample_run_fingerprint(image_digest=("e" if field == "image" else "d") * 64)
    with pytest.raises(ValueError, match="RESEARCH_TRIAL_LINEAGE"):
        execute_trial(
            decode_research_trial(document),
            fingerprint,
            ("e" if field == "code" else "c") * 64,
            tmp_path / "result" / "out.json",
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("partition", "holdout", "holdout_forbidden"),
        ("features", ["volume_60"], "supports only return_1"),
    ],
)
def test_worker_refuses_final_fold_and_unsupported_features(
    tmp_path: Path, field: str, value: JSONValue, message: str
) -> None:
    document = trial_document(tmp_path)
    require_dict(document, "proposal")[field] = value
    with pytest.raises(ValueError, match=message):
        execute_trial(
            decode_research_trial(document),
            sample_run_fingerprint(image_digest="d" * 64),
            "c" * 64,
            tmp_path / "result" / "out.json",
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference", "unknown"),
        ("inner_train_fraction", 1.0),
        (
            "resampling",
            {
                "unit": "row",
                "block_length": 1,
                "n_resamples": 20,
                "resample_seed": 7,
                "confidence": 0.95,
            },
        ),
        (
            "resampling",
            {
                "unit": "block",
                "block_length": 3,
                "n_resamples": 10001,
                "resample_seed": 7,
                "confidence": 0.95,
            },
        ),
    ],
)
def test_invalid_protocol_is_refused(tmp_path: Path, field: str, value: JSONValue) -> None:
    document = require_dict(trial_document(tmp_path), "evaluation")
    document[field] = value
    with pytest.raises(ValueError, match="RESEARCH_TRIAL_PROTOCOL"):
        decode_research_evaluation(document)


def test_exact_request_and_protocol_fields_are_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact request"):
        decode_research_trial({})
    with pytest.raises(ValueError, match="exact protocol"):
        decode_research_evaluation({})
    document: JSONObject = trial_document(tmp_path)
    document["manifest"] = "relative"
    with pytest.raises(ValueError, match="absolute manifest"):
        decode_research_trial(document)
