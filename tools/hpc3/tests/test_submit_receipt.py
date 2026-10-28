"""Machine-readable submission uses the same preflight and persisted ledger."""

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, load_json_str

from hpc3.cli import submit
from hpc3.contracts.ledger import encode_ledger_entry
from hpc3.core import ledger
from tests.conftest import FakeRun, cluster, write_workspace
from tests.healthy_cluster import script_healthy_cluster


def test_json_submission_is_exactly_the_persisted_entry(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    config = write_workspace(tmp_path / "hpc3.json")
    recipe = tmp_path / "run.json"
    recipe.write_text(
        dump_json_str(
            {
                "project": "abl",
                "name": "arm-b-42",
                "command": "python train.py",
                "artifact": None,
                "experiment": {"arm": "B", "seed": "42"},
            }
        )
    )
    script_healthy_cluster(fake_run)
    assert submit.main(["--config", config, "--run", str(recipe), "--format", "json"]) == 0
    assert len(emitted) == 1
    rows = ledger.read(tmp_path / "ledger.jsonl", cluster())
    assert len(rows) == 1
    assert load_json_str(emitted[0]) == encode_ledger_entry(rows[0])
    assert rows[0]["job_id"] == "55519937"
    assert any("--test-only" in command for command in fake_run.commands())


def test_unknown_output_format_fails_before_remote_work(fake_run: FakeRun) -> None:
    with pytest.raises(ValueError, match="HPC_SUBMIT_FORMAT"):
        submit.main(["--format", "yaml"])
    assert fake_run.commands() == []
