"""Real monthly validation evaluation, strict lineage and no reuse of an attempt output."""

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
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

from ._trial_fixtures import frozen_request, trial_document


def _run(document: JSONObject, output: Path) -> Path:
    """Execute a request under the fixture's pinned code and image identities."""
    return execute_trial(
        decode_research_trial(document),
        sample_run_fingerprint(image_digest="d" * 64),
        "c" * 64,
        output,
    )


def test_real_trainers_emit_the_admitted_shared_monthly_record(tmp_path: Path) -> None:
    document = trial_document(tmp_path)
    trial = decode_research_trial(document)
    assert encode_research_trial(trial) == document
    assert encode_research_evaluation(trial["evaluation"]) == document["evaluation"]
    output = tmp_path / "attempt-1" / "result.json"
    record_path = _run(document, output)
    record = decode_run_record(load_json_str(record_path.read_text()))
    result = narrow_json_to_dict(load_json_str(output.read_text()))
    assert (record["experiment"], record["label"]) == (trial["policy"]["protocol_digest"], "1")
    names = [observation["name"] for observation in record["observations"]]
    # Five arms at three frozen costs, two means each, and four paired intervals.
    assert len(names) == 5 * 3 * 2 + 4 * 4
    assert "mean_net_return.momentum.10bps" in names
    assert "equal_weight_vs_momentum.net_return.ci_low" in names
    assert result["kind"] == "research-monthly-validation-v1"
    assert result["partition"] == "validation"
    assert result["attempt"] == 1
    assert len(require_list(result, "folds")) == 2
    assert len(require_list(result, "results")) == 15
    first = narrow_json_to_dict(require_list(result, "results")[0])
    month = narrow_json_to_dict(require_list(first, "months")[0])
    assert (first["arm"], first["cost_bps"], month["decision"]) == ("momentum", 10, "2018-07-31")
    assert len(require_list(month, "held")) == 3
    assert require_dict(result, "source")["extracted_at"] == "2026-10-29T00:00:00+00:00"
    with pytest.raises(FileExistsError):
        _run(document, output)
    assert decode_run_record(load_json_str(record_path.read_text())) == record


@pytest.mark.parametrize("field", ["protocol", "code", "image", "dataset"])
def test_worker_rechecks_frozen_identities_before_fitting(tmp_path: Path, field: str) -> None:
    document = trial_document(tmp_path)
    if field == "protocol":
        require_dict(document, "evaluation")["stopping_months"] = 5
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


def test_staged_protocol_must_be_the_admitted_design(tmp_path: Path) -> None:
    staged = trial_document(tmp_path)
    evaluation = require_dict(staged, "evaluation")
    require_dict(evaluation, "design")["bootstrap_draws"] = 300
    document = frozen_request(evaluation, Path(str(staged["manifest"])), ["momentum"])
    with pytest.raises(ValueError, match="staged CRSP protocol differs"):
        _run(document, tmp_path / "result" / "out.json")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("partition", "holdout", "holdout_forbidden"),
        ("features", ["return_1"], "features_forbidden"),
    ],
)
def test_worker_refuses_final_fold_and_unadmitted_features(
    tmp_path: Path, field: str, value: JSONValue, message: str
) -> None:
    document = trial_document(tmp_path)
    require_dict(document, "proposal")[field] = value
    with pytest.raises(ValueError, match=message):
        _run(document, tmp_path / "result" / "out.json")
    assert not (tmp_path / "result").exists()


def test_admitted_feature_the_panel_cannot_construct_is_refused(tmp_path: Path) -> None:
    staged = trial_document(tmp_path)
    document = frozen_request(
        require_dict(staged, "evaluation"), Path(str(staged["manifest"])), ["volume_60"]
    )
    with pytest.raises(ValueError, match="CRSP-009"):
        _run(document, tmp_path / "result" / "out.json")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stopping_months", 0, r"MARKET-001\] stopping_months"),
        ("fold_months", 61, r"MARKET-001\] fold_months"),
        ("confidence", 1.0, "CVML-EVAL-004"),
    ],
)
def test_invalid_protocol_is_refused(
    tmp_path: Path, field: str, value: JSONValue, message: str
) -> None:
    document = require_dict(trial_document(tmp_path), "evaluation")
    document[field] = value
    with pytest.raises(ValueError, match=message):
        decode_research_evaluation(document)


def test_changed_frozen_design_is_refused(tmp_path: Path) -> None:
    document = require_dict(trial_document(tmp_path), "evaluation")
    require_dict(document, "design")["cost_bps"] = [10, 5]
    with pytest.raises(ValueError, match="CRSP-001"):
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
