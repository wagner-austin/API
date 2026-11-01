"""Real trainer-to-journal-to-summary publication, process failure and provenance."""

import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from platform_core.run_record import decode_run_record
from platform_core.testing import sample_run_fingerprint
from scripts.research_proposals import main

from covenant_ml.markets.research_journal import (
    admit_proposal,
    encode_research_attempt,
    research_journal,
)
from covenant_ml.markets.research_records import encode_research_policy, encode_research_proposal
from covenant_ml.markets.research_summary import write_validation_summary
from covenant_ml.markets.research_trial_records import decode_research_trial
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._research_final_fixtures import completed_study, write_trial
from ._trial_fixtures import trial_document


def test_real_process_publishes_bound_summary_without_keys_or_extracts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, trial, final_manifest = completed_study(tmp_path)
    journal = research_journal(trial["policy"])
    rows = journal.read(ledger)
    before = ledger.read_bytes()
    # The summarizer must use the durable record alone; even synthetic keys go away.
    keys = sorted((tmp_path / "validation/keys").glob("*.agekey"))
    assert [key.name for key in keys] == ["final.agekey", "train-validation.agekey"]
    for key in keys:
        key.unlink()
    Path(trial["manifest"]).unlink()
    final_manifest.unlink()
    policy = tmp_path / "policy.json"
    policy.write_text(dump_json_str(encode_research_policy(trial["policy"])), encoding="utf-8")
    request = write_trial(trial, tmp_path / "trial.json")
    output = tmp_path / "summary/result.json"
    command = [
        sys.executable,
        "-m",
        "scripts.research_proposals",
        "--policy",
        str(policy),
        "--ledger",
        str(ledger),
        "summarize",
        "--trial",
        str(request),
        "--out",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    record_path = Path(completed.stdout.strip())
    record = decode_run_record(load_json_str(record_path.read_text()))
    document = narrow_json_to_dict(load_json_str(output.read_text()))
    assert record["label"] == "validation-seed-summary"
    assert record["experiment"] == sha256(dump_json_str(document).encode()).hexdigest()
    assert (
        record["payload_digest"]
        == sha256(dump_json_str(require_list(document, "results")).encode()).hexdigest()
    )
    assert (
        document["journal_snapshot_sha256"]
        == sha256(
            dump_json_str([encode_research_attempt(row) for row in rows]).encode()
        ).hexdigest()
    )
    assert document["attempts"] == [
        {"number": row["number"], "candidate": row["candidate"], "status": "complete"}
        for row in rows
    ]
    assert require_dict(document, "evaluation")["confidence"] == 0.9
    assert (
        document["policy_digest"]
        == sha256(dump_json_str(encode_research_policy(trial["policy"])).encode()).hexdigest()
    )
    reporter = Path(__file__).parents[2] / "src/covenant_ml/markets/research_summary.py"
    assert document["reporter_sha256"] == sha256(reporter.read_bytes()).hexdigest()
    results = [narrow_json_to_dict(item) for item in require_list(document, "results")]
    assert len(results) == 4
    assert all((item["n_seeds"], item["n_months"]) == (2, 16) for item in results)
    assert len(record["observations"]) == 20
    payload_before, record_before = output.read_bytes(), record_path.read_bytes()
    with pytest.raises(FileExistsError) as existing:
        output.parent.mkdir()
    refused = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert refused.returncode == 1
    assert refused.stderr.splitlines()[-1] == f"FileExistsError: {existing.value}"
    assert (output.read_bytes(), record_path.read_bytes()) == (payload_before, record_before)
    assert ledger.read_bytes() == before
    assert journal.read(ledger) == rows
    assert list(tmp_path.glob("research-plaintext-*")) == []
    assert (
        main(
            [
                "--policy",
                str(policy),
                "--ledger",
                str(ledger),
                "summarize",
                "--trial",
                str(request),
                "--out",
                str(tmp_path / "controller-summary/result.json"),
            ],
            stand_in_pin,
        )
        == 0
    )
    controller_record = decode_run_record(
        load_json_str(Path(capsys.readouterr().out.strip()).read_text())
    )
    assert controller_record["experiment"] == record["experiment"]
    assert controller_record["observations"] == record["observations"]
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("status", ["admitted", "submitting", "running"])
def test_unsettled_journal_is_refused_without_output(tmp_path: Path, status: str) -> None:
    trial = decode_research_trial(trial_document(tmp_path))
    ledger = tmp_path / "journal.sqlite"
    journal = research_journal(trial["policy"])
    journal.initialize(ledger)
    proposal = encode_research_proposal(trial["proposal"])
    row = admit_proposal(ledger, trial["policy"], proposal, 2000)
    assert row["status"] == "admitted"
    if status in {"submitting", "running"}:
        journal.transact(ledger, lambda _: {"action": "submit", "number": 1, "observed_ms": 2001})
    if status == "running":
        journal.transact(
            ledger,
            lambda _: {
                "action": "scheduled",
                "number": 1,
                "observed_ms": 2002,
                "job_id": "slurm:1",
            },
        )
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="RESEARCH_SUMMARY_UNSETTLED"):
        write_validation_summary(
            ledger, trial, tmp_path / "summary/result.json", sample_run_fingerprint()
        )
    assert not (tmp_path / "summary").exists()
    assert ledger.read_bytes() == before
