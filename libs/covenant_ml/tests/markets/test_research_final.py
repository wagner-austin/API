"""Final scoring needs the journal's selection and consumes its one release first."""

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from platform_core.run_record import decode_run_record
from platform_core.testing import sample_run_fingerprint

from covenant_ml.markets.research_final import execute_final, select_candidate
from covenant_ml.markets.research_journal import admit_proposal, research_journal
from covenant_ml.markets.research_records import encode_research_proposal
from covenant_ml.markets.research_trial_records import (
    ResearchTrial,
    decode_research_trial,
    encode_research_trial,
)

from ._encrypted_fixtures import evaluator_keys
from ._research_final_fixtures import completed_study


def _final(path: Path, trial: ResearchTrial, manifest: Path, output: Path, code: str = "c") -> Path:
    """Run the evaluator's final score under the fixture's pinned image."""
    return execute_final(
        path,
        trial,
        manifest,
        sample_run_fingerprint(image_digest="d" * 64),
        code * 64,
        output,
        keys=evaluator_keys(path.parent / "validation"),
        scratch=path.parent,
    )


def test_selected_candidate_is_released_once_and_scored_on_the_final_period(
    tmp_path: Path,
) -> None:
    path, trial, manifest = completed_study(tmp_path)
    selected = select_candidate(path, trial, manifest)
    assert (selected["status"], selected["selection"]) == ("complete", "selected")
    assert selected["final_manifest_digest"] == sha256(manifest.read_bytes()).hexdigest()
    # Selection seals admission: a later proposal is recorded and refused.
    sealed = admit_proposal(
        path, trial["policy"], encode_research_proposal(trial["proposal"]), 5000
    )
    assert (sealed["status"], sealed["reason"]) == ("refused", "study_sealed")
    output = tmp_path / "final-1" / "result.json"
    record = decode_run_record(load_json_str(_final(path, trial, manifest, output).read_text()))
    assert record["experiment"] == trial["policy"]["protocol_digest"]
    assert record["label"] == "final-1"
    result = narrow_json_to_dict(load_json_str(output.read_text()))
    assert (result["kind"], result["partition"], result["attempt"]) == (
        "research-monthly-final-v1",
        "final",
        1,
    )
    assert require_dict(result, "source")["protocol"] == require_dict(
        narrow_json_to_dict(load_json_str((tmp_path / "final/manifest.json").read_text())),
        "protocol",
    )
    (fit,) = [narrow_json_to_dict(item) for item in require_list(result, "folds")]
    assert fit["first_scored"] == "2020-01-31"
    first = narrow_json_to_dict(require_list(result, "results")[0])
    months = [narrow_json_to_dict(item) for item in require_list(first, "months")]
    assert [month["decision"] for month in months] == [
        "2020-01-31",
        "2020-02-28",
        "2020-03-31",
        "2020-04-30",
        "2020-05-29",
        "2020-06-30",
    ]
    rows = research_journal(trial["policy"]).read(path)
    assert (rows[0]["status"], rows[0]["selection"]) == ("complete", "released")
    assert rows[0]["record"] == selected["record"]
    with pytest.raises(ValueError, match="RESEARCH-009"):
        _final(path, trial, manifest, tmp_path / "again" / "result.json")
    assert not (tmp_path / "again").exists()


def _changed(trial: ResearchTrial) -> ResearchTrial:
    """The same request with a different seed than the journal admitted."""
    document = encode_research_trial(trial)
    require_dict(document, "proposal")["seed"] = 23
    return decode_research_trial(document)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unselected", "RESEARCH-009"),
        ("manifest", "RESEARCH_TRIAL_LINEAGE"),
        ("code", "RESEARCH_TRIAL_LINEAGE"),
        ("request", "RESEARCH_EXECUTION_REQUEST"),
    ],
)
def test_final_refusals_before_release_keep_the_selection(
    tmp_path: Path, case: str, message: str
) -> None:
    path, trial, manifest = completed_study(tmp_path)
    if case != "unselected":
        select_candidate(path, trial, manifest)
    before = research_journal(trial["policy"]).read(path)
    staged = Path(trial["manifest"]) if case == "manifest" else manifest
    requested = _changed(trial) if case == "request" else trial
    with pytest.raises(ValueError, match=message):
        _final(path, requested, staged, tmp_path / "out" / "r.json", "e" if case == "code" else "c")
    assert research_journal(trial["policy"]).read(path) == before
    assert not (tmp_path / "out").exists()


def test_release_is_consumed_before_any_final_relation_is_read(tmp_path: Path) -> None:
    path, trial, manifest = completed_study(tmp_path)
    select_candidate(path, trial, manifest)
    (manifest.parent / "returns.csv.age").unlink()
    with pytest.raises(FileNotFoundError):
        _final(path, trial, manifest, tmp_path / "out" / "r.json")
    assert research_journal(trial["policy"]).read(path)[0]["selection"] == "released"
    with pytest.raises(ValueError, match="RESEARCH-009"):
        _final(path, trial, manifest, tmp_path / "out" / "r.json")
    assert not (tmp_path / "out").exists()


def test_final_key_is_acquired_only_after_irrevocable_release(tmp_path: Path) -> None:
    path, trial, manifest = completed_study(tmp_path)
    (tmp_path / "validation/keys/final.agekey").unlink()
    with pytest.raises(ValueError, match="RESEARCH-009"):
        _final(path, trial, manifest, tmp_path / "out/r.json")
    assert research_journal(trial["policy"]).read(path)[0]["selection"] == ""
    select_candidate(path, trial, manifest)
    with pytest.raises(FileNotFoundError):
        _final(path, trial, manifest, tmp_path / "out/r.json")
    assert research_journal(trial["policy"]).read(path)[0]["selection"] == "released"
    with pytest.raises(ValueError, match="RESEARCH-009"):
        _final(path, trial, manifest, tmp_path / "out/r.json")
    assert list(tmp_path.glob("research-plaintext-*")) == []


def test_a_partition_field_never_authorizes_final_scoring(tmp_path: Path) -> None:
    path, trial, manifest = completed_study(tmp_path)
    request: JSONObject = encode_research_proposal(trial["proposal"])
    request["partition"] = "final"
    request["seed"] = 23
    refused = admit_proposal(path, trial["policy"], request, 5000)
    assert (refused["number"], refused["status"]) == (2, "refused")
    assert refused["reason"] == "holdout_forbidden"
    document = encode_research_trial(trial)
    document["proposal"] = request
    document["number"] = 2
    asked = decode_research_trial(document)
    with pytest.raises(ValueError, match="RESEARCH-008"):
        select_candidate(path, asked, manifest)
    with pytest.raises(ValueError, match="RESEARCH-009"):
        _final(path, asked, manifest, tmp_path / "out" / "r.json")
    assert [row["selection"] for row in research_journal(trial["policy"]).read(path)] == ["", ""]
    assert not (tmp_path / "out").exists()


def test_selection_requires_settled_completed_validation(tmp_path: Path) -> None:
    path, trial, manifest = completed_study(tmp_path)
    request: JSONObject = encode_research_proposal(trial["proposal"])
    request["seed"] = 23
    admitted = admit_proposal(path, trial["policy"], request, 5000)
    assert admitted["status"] == "admitted"
    document = encode_research_trial(trial)
    document["proposal"] = request
    document["number"] = 2
    with pytest.raises(ValueError, match="RESEARCH-008"):
        select_candidate(path, decode_research_trial(document), manifest)
    # Attempt 2 is still active, so even the completed attempt cannot be selected yet.
    with pytest.raises(ValueError, match="RESEARCH-008"):
        select_candidate(path, trial, manifest)
