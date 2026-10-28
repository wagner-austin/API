"""The maintained proposal entrypoint dispatches submission and reconciliation."""

from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from scripts.research_proposals import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.research_cluster import encode_research_cluster
from covenant_ml.markets.research_journal import research_journal
from covenant_ml.markets.research_records import encode_research_policy
from covenant_ml.markets.research_trial_records import encode_research_trial
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._research_execution_fixtures import Scheduler, setup


@pytest.fixture(autouse=True)
def restore_runner() -> Generator[None, None, None]:
    previous = _test_hooks.campaign_runner
    yield
    _test_hooks.campaign_runner = previous


def test_commands_use_the_same_journal_and_policy(tmp_path: Path) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    policy_path, trial_path, cluster_path = (
        tmp_path / name for name in ("policy.json", "trial.json", "cluster.json")
    )
    policy_path.write_text(dump_json_str(encode_research_policy(trial["policy"])))
    trial_path.write_text(dump_json_str(encode_research_trial(trial)))
    cluster_path.write_text(dump_json_str(encode_research_cluster(cluster)))
    scheduler = Scheduler(trial)
    _test_hooks.campaign_runner = scheduler
    base = ["--policy", str(policy_path), "--ledger", str(path)]
    execution = [
        "--trial",
        str(trial_path),
        "--cluster",
        str(cluster_path),
        "--directory",
        str(directory),
    ]
    assert main([*base, "submit", *execution], stand_in_pin) == 0
    assert main([*base, "reconcile", *execution], stand_in_pin) == 0
    assert research_journal(trial["policy"]).read(path)[0]["status"] == "complete"
    changed = encode_research_policy(trial["policy"])
    changed["code_digest"] = "e" * 64
    policy_path.write_text(dump_json_str(changed))
    with pytest.raises(ValueError, match="trial policy differs"):
        main([*base, "submit", *execution], stand_in_pin)
    assert len(scheduler.calls) == 3
