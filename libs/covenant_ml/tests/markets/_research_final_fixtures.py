"""A real completed validation attempt and a staged final-period extract."""

from pathlib import Path

from platform_core.event_journal import EventJournal
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)
from platform_core.testing import sample_run_fingerprint

from covenant_ml.markets.research_journal import (
    ResearchAttempt,
    admit_proposal,
    research_journal,
)
from covenant_ml.markets.research_trial import execute_trial
from covenant_ml.markets.research_trial_records import (
    ResearchTrial,
    decode_research_trial,
    encode_research_trial,
)

from ._crsp_study_fixtures import final_manifest
from ._trial_fixtures import trial_document


def _commit(
    journal: EventJournal[tuple[ResearchAttempt, ...]], path: Path, event: JSONObject
) -> None:
    """Persist one lifecycle event through the journal's real transaction."""
    journal.transact(path, lambda _: event)


def completed_study(root: Path, code_digest: str = "c" * 64) -> tuple[Path, ResearchTrial, Path]:
    """Admit, run and complete attempt 1 with its real monthly record; stage final data.

    Args:
        root: Empty directory for the extracts, journal and outputs.
        code_digest: Entrypoint digest the frozen policy pins.

    Returns:
        Journal path, the completed attempt's request and the final manifest.
    """
    (root / "validation").mkdir()
    (root / "final").mkdir()
    document = trial_document(root / "validation")
    require_dict(document, "policy")["code_digest"] = code_digest
    trial = decode_research_trial(document)
    path = root / "journal.sqlite"
    journal = research_journal(trial["policy"])
    journal.initialize(path)
    admit_proposal(path, trial["policy"], require_dict(document, "proposal"), 1000)
    record = execute_trial(
        trial,
        sample_run_fingerprint(image_digest="d" * 64),
        code_digest,
        root / "validation-1" / "result.json",
    )
    _commit(journal, path, {"action": "submit", "number": 1, "observed_ms": 1001})
    _commit(
        journal,
        path,
        {"action": "scheduled", "number": 1, "observed_ms": 1002, "job_id": "slurm:1"},
    )
    _commit(
        journal,
        path,
        {
            "action": "complete",
            "number": 1,
            "observed_ms": 1003,
            "job_id": "slurm:1",
            "dataset_digest": trial["policy"]["dataset_digest"],
            "code_digest": code_digest,
            "record": narrow_json_to_dict(load_json_str(record.read_text())),
        },
    )
    return path, trial, final_manifest(root / "final")


def write_trial(trial: ResearchTrial, path: Path) -> Path:
    """Persist the evaluator's frozen request for a CLI.

    Args:
        trial: Frozen request.
        path: Destination.

    Returns:
        The written path.
    """
    path.write_text(dump_json_str(encode_research_trial(trial)), encoding="utf-8")
    return path


__all__ = ["completed_study", "write_trial"]
