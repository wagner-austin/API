"""The 2026-10-05 double submission, replayed against the campaign command.

Board task 747c391e. A worker resuming the v56 cartridge-QA campaign in
``API-research-w2`` ran ``hpc3-campaign`` while ``API-research-w3`` held the
ledger rows for its twelve live members. w2's own ledger had none, so the
campaign reported ``0 in flight`` and submitted array 57783755 beside them.

Two cases, one per half of the fix. From a linked worktree, the campaign now
reads the main checkout's ledger, so the live members read as in flight. And
when the live jobs were recorded in no ledger this checkout can read -- the
other machine's -- the cluster still names them, and the campaign refuses to
submit rather than call them missing. The worktree is made by real git; the
cluster is the fake runner every CLI test uses.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.cli import campaign as campaign_cli
from tests.conftest import (
    FakeRun,
    ledger_row,
    workspace_document,
    write_file,
    write_workspace,
)
from tests.healthy_cluster import script_healthy_cluster

_WALL_SECONDS = 60
"""Bound on each git call; every one of them answers in milliseconds."""

_RUNS = pathlib.Path("tools", "hpc3", "runs")
_TR = "/pub/wagnera3/LSTM/checkpoints/tr_best.pt"
_AZ = "/pub/wagnera3/LSTM/checkpoints/az_best.pt"
_LIVE_ARRAY = "57783112_[0-1]|abl.bases|PENDING\n"
"""The two members, still pending as one aggregate row under the sweep's name."""


def _git(cwd: pathlib.Path, *arguments: str) -> None:
    """Run git in a directory.

    Args:
        cwd: Where to run it.
        *arguments: Git's arguments.
    """
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments],
        cwd=cwd,
        capture_output=True,
        check=True,
        timeout=_WALL_SECONDS,
    )


def _main_and_worktree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Make a main checkout and one linked worktree of it.

    Args:
        tmp_path: Directory to build in.

    Returns:
        The main checkout's runs directory and the worktree's, resolved.
    """
    main = tmp_path / "API"
    main.mkdir()
    _git(main, "init", "-q")
    write_file(main / _RUNS / "hpc3.json", b"{}\n")
    _git(main, "add", ".")
    _git(main, "commit", "-q", "-m", "runs")
    _git(main, "worktree", "add", "-q", "--detach", str(tmp_path / "API-research-w2"))
    return (main / _RUNS).resolve(), (tmp_path / "API-research-w2" / _RUNS).resolve()


def _campaign(runs: pathlib.Path) -> list[str]:
    """Write the two-member campaign and its workspace into a runs directory.

    Args:
        runs: The directory a worker would run the command from.

    Returns:
        The command's arguments.
    """
    members: list[JSONValue] = [
        {"suffix": "tr", "command": f"python train.py --out {_TR}", "artifact": _TR},
        {"suffix": "az", "command": f"python train.py --out {_AZ}", "artifact": _AZ},
    ]
    document: dict[str, JSONValue] = {
        "project": "abl",
        "name": "bases",
        "members": members,
        "experiment": {"corpus": "v3"},
    }
    write_file(runs / "sweep.json", dump_json_str(document).encode("utf-8"))
    config = write_workspace(runs / "hpc3.json", workspace_document())
    return ["--config", config, "--run", str(runs / "sweep.json")]


class TestTheCampaignThatWasSubmittedTwice:
    def test_a_worktree_sees_the_members_another_checkout_submitted(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        main_runs, tree_runs = _main_and_worktree(tmp_path)
        rows = [
            ledger_row(job_id="57783112_0", name="abl.bases-tr", artifact=_TR),
            ledger_row(job_id="57783112_1", name="abl.bases-az", artifact=_AZ),
        ]
        write_file(
            main_runs / "ledger.jsonl",
            b"".join(dump_json_str(row).encode("utf-8") + b"\n" for row in rows),
        )
        fake_run.add("PRESENT|", stdout=f"ABSENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout=_LIVE_ARRAY)
        script_healthy_cluster(fake_run)

        assert campaign_cli.main(_campaign(tree_runs)) == 0
        assert emitted == [
            "in flight abl.bases-tr <- abl.bases-tr",
            "in flight abl.bases-az <- abl.bases-az",
            "0 done, 2 in flight, 0 submitted, 2 remaining",
        ]
        assert not any("sbatch" in command for command in fake_run.commands())
        assert not (tree_runs / "ledger.jsonl").exists()

    def test_a_member_the_worktree_submits_lands_in_the_main_ledger(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """The other direction: what w2 submits, w3 and the main checkout read."""
        main_runs, tree_runs = _main_and_worktree(tmp_path)
        fake_run.add("PRESENT|", stdout=f"ABSENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        script_healthy_cluster(fake_run)

        assert campaign_cli.main(_campaign(tree_runs)) == 0
        assert emitted[-1] == "0 done, 0 in flight, 2 submitted, 2 remaining"
        recorded = (main_runs / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(recorded) == 2
        assert not (tree_runs / "ledger.jsonl").exists()

    def test_live_members_no_readable_ledger_recorded_are_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """Submitted from another machine's ledger: nothing here records the
        array, but the cluster still names it, and that is enough to refuse."""
        _, tree_runs = _main_and_worktree(tmp_path)
        fake_run.add("PRESENT|", stdout=f"ABSENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout=_LIVE_ARRAY)
        script_healthy_cluster(fake_run)

        with pytest.raises(AppError) as excinfo:
            campaign_cli.main(_campaign(tree_runs))
        assert excinfo.value.code is Hpc3ErrorCode.LIVE_JOB_UNRECORDED
        assert "57783112_[0-1] abl.bases (PENDING)" in excinfo.value.message
        assert not any("sbatch" in command for command in fake_run.commands())
