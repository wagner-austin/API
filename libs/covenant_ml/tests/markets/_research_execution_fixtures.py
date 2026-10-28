"""Frozen research execution fixtures with real journal and bundle persistence."""

from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, require_dict
from platform_core.run_record import Observation, encode_run_record, run_record
from platform_core.testing import sample_run_fingerprint

from covenant_ml.markets.research_cluster import ResearchCluster, decode_research_cluster
from covenant_ml.markets.research_journal import admit_proposal, research_journal
from covenant_ml.markets.research_trial_records import ResearchTrial, decode_research_trial
from covenant_ml.markets.validation import read_document

from ._trial_fixtures import trial_document


def setup(root: Path) -> tuple[Path, ResearchTrial, ResearchCluster, Path]:
    trial = decode_research_trial(trial_document(root))
    path = root / "journal.sqlite"
    research_journal(trial["policy"]).initialize(path)
    admit_proposal(path, trial["policy"], require_dict(trial_document(root), "proposal"), 0)
    cluster = decode_research_cluster(
        {
            "python": "/usr/bin/python3",
            "workspace": str(root / "hpc3.json"),
            "project": "cleargbm",
            "root": "/pub/research",
            "entrypoint": "/pub/entrypoint",
            "image": "/pub/runtime.sif",
            "report_task": "6d2d086c-5242-482b-ac81-8a61dd8e354d",
            "checksums": (
                f"{'d' * 64}  /pub/runtime.sif\n"
                f"{'c' * 64}  /pub/entrypoint/scripts/research_trial.py\n"
                f"{trial['policy']['dataset_digest']}  {trial['manifest']}\n"
            ),
        }
    )
    return path, trial, cluster, root / "attempt"


def receipt(directory: Path) -> JSONObject:
    recipe = read_document(directory / "run.json")
    return {
        "job_id": "123",
        "experiment": recipe["experiment"],
        "artifact": recipe["artifact"],
        "image_digest": "d" * 64,
    }


def observation(directory: Path, trial: ResearchTrial, state: str = "COMPLETED") -> JSONObject:
    accounting: JSONObject = {"job": {"state": state}, "exit_code": "0:0"}
    record = run_record(
        trial["policy"]["protocol_digest"],
        str(trial["number"]),
        sample_run_fingerprint(image_digest="d" * 64),
        (Observation(name="score", value=0.5),),
        "e" * 64,
    )
    return {
        "ledger": receipt(directory),
        "accounting": accounting,
        "measurement": {"accounting": accounting},
        "record": encode_run_record(record),
    }


class Scheduler:
    """Transport seam; its real subprocess failure twin is tested separately."""

    def __init__(self, trial: ResearchTrial) -> None:
        self.trial = trial
        self.calls: list[tuple[str, ...]] = []
        self.fail = ""
        self.state = "COMPLETED"
        self.change: JSONObject = {}

    def __call__(self, command: tuple[str, ...], directory: Path, log: Path) -> int:
        self.calls.append(command)
        module = command[2]
        if module.endswith(self.fail) and self.fail:
            log.write_text("transport failed")
            return 7
        if module.endswith("stage"):
            log.write_text("staged")
            return 0
        document = (
            observation(directory, self.trial, self.state)
            if module.endswith("execution_inspect")
            else receipt(directory)
        )
        document.update(self.change)
        log.write_text(dump_json_str(document))
        return 0
