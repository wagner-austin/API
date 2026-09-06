"""The array submission's own refusals, beyond what the sweep tests drive.

The happy path -- one upload, one preflight, one sbatch, per-member ledger
rows -- is held by ``test_sweep.py`` and ``test_campaign.py`` through the
doors callers actually use. What lives here is the refusal surface: bad
indices, a scheduler rejection carrying the array expression, and the
artifact race checked before anything is uploaded.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.contracts.sweep import SweepSpec
from hpc3.core.array_submit import selected_members, submit_array
from tests.against_hpc3 import decode_sweep_spec
from tests.conftest import FakeRun, cluster, gpus

_AT = "2026-09-01T22:00:00+00:00"


def _sweep(count: int = 3, **overrides: JSONValue) -> SweepSpec:
    """Build a decoded sweep.

    Args:
        count: How many members.
        **overrides: Template fields to replace.

    Returns:
        A validated sweep.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "rung",
        "partition": "free-gpu",
        "gpu": gpus("A100"),
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "env_path": "/pub/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"rung": "774M"},
        "command": "python train.py",
        "artifact": None,
    }
    base.update(overrides)
    members: list[JSONValue] = [
        {"suffix": f"s{i}", "command": f"python train.py --seed {i}", "artifact": None}
        for i in range(count)
    ]
    return decode_sweep_spec({"base": base, "members": members})


def _submit(
    spec: SweepSpec, indices: tuple[int, ...], tmp_path: pathlib.Path
) -> list[tuple[str, str]]:
    return [
        (member.name, member.job_id)
        for member in submit_array(
            spec,
            indices,
            host="hpc3",
            script_dir="/j",
            log_dir="/l",
            ledger_path=tmp_path / "ledger.jsonl",
            submitted_at=_AT,
            submitter="fable-brain-audit-0903",
            cluster=cluster(),
            charge_account="",
        )
    ]


class TestSelectingMembers:
    def test_indices_resolve_to_names_in_index_order(self) -> None:
        assert selected_members(_sweep(), (0, 2)) == [(0, "abl.rung-s0"), (2, "abl.rung-s2")]

    def test_an_index_past_the_table_is_refused(self) -> None:
        """The index list is the campaign's bookkeeping; a position past
        the table means the bookkeeping and the document disagree."""
        with pytest.raises(AppError) as caught:
            selected_members(_sweep(), (0, 3))
        assert caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE
        assert "names no member" in caught.value.message
        assert "3 member(s)" in caught.value.message

    def test_a_negative_index_is_refused(self) -> None:
        with pytest.raises(AppError) as caught:
            selected_members(_sweep(), (-1,))
        assert caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE


class TestRefusals:
    def test_a_scheduler_rejection_names_the_array_and_uploads_nothing_twice(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", returncode=0, stdout="some refusal\nrc=1\n")

        with pytest.raises(AppError) as caught:
            _submit(_sweep(), (0, 1, 2), tmp_path)

        assert caught.value.code is Hpc3ErrorCode.PREFLIGHT_REJECTED
        assert "abl.rung" in caught.value.message
        # The expression is in the message because a sparse campaign
        # resubmission and a full sweep refuse differently, and the reader
        # should not have to reconstruct which this was.
        assert "0-2" in caught.value.message
        assert not (tmp_path / "ledger.jsonl").exists()

    def test_a_live_job_on_a_selected_artifact_refuses_before_any_upload(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """One account enumeration covers every member, and the refusal
        happens before the script exists anywhere."""
        spec = _sweep(artifact=None)
        # Give member s1 an artifact and a live claimant: the ledger row
        # binds the artifact to job 500, and the account says 500 is live.
        armed: dict[str, JSONValue] = {
            "base": {
                "project": "abl",
                "name": "rung",
                "partition": "free-gpu",
                "gpu": gpus("A100"),
                "cpus": 8,
                "mem_gb": 96,
                "minutes": 30,
                "requeue": False,
                "checkpoint_steps": 0,
                "env_path": "/pub/envs/abl-pinned",
                "pinned_packages": {},
                "deterministic": False,
                "experiment": {"rung": "774M"},
                "command": "python train.py",
                "artifact": None,
            },
            "members": [
                {
                    "suffix": "s0",
                    "command": "python train.py --seed 0 --out /pub/a0.pt",
                    "artifact": "/pub/a0.pt",
                },
            ],
        }
        spec = decode_sweep_spec(armed)
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text(
            '{"job_id": "500", "project": "abl", "name": "abl.older", "host": "hpc3", '
            '"partition": "free-gpu", "submitted_at": "2026-09-01T00:00:00+00:00", '
            '"log_dir": "/l", "deterministic": false, "experiment": {"rung": "774M"}, '
            '"image_digest": "", "submitter": "", "artifact": "/pub/a0.pt"}\n',
            encoding="utf-8",
        )
        fake_run.add("squeue --me", stdout="500|abl.older|RUNNING\n")

        with pytest.raises(AppError) as caught:
            _submit(spec, (0,), tmp_path)

        assert caught.value.code is Hpc3ErrorCode.ARTIFACT_ALREADY_IN_FLIGHT
        uploads = [c for c in fake_run.commands() if c.startswith("cat >")]
        assert uploads == []

    def test_a_pending_aggregate_claim_still_refuses(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """The measured hazard, end to end: the ledger holds a TASK id and
        squeue reports it inside a pending aggregate. Unexpanded, the task
        would read as not-live and this submission would race it."""
        armed: dict[str, JSONValue] = {
            "base": {
                "project": "abl",
                "name": "rung",
                "partition": "free-gpu",
                "gpu": gpus("A100"),
                "cpus": 8,
                "mem_gb": 96,
                "minutes": 30,
                "requeue": False,
                "checkpoint_steps": 0,
                "env_path": "/pub/envs/abl-pinned",
                "pinned_packages": {},
                "deterministic": False,
                "experiment": {"rung": "774M"},
                "command": "python train.py",
                "artifact": None,
            },
            "members": [
                {
                    "suffix": "s0",
                    "command": "python train.py --seed 0 --out /pub/a0.pt",
                    "artifact": "/pub/a0.pt",
                },
            ],
        }
        spec = decode_sweep_spec(armed)
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text(
            '{"job_id": "55678543_2", "project": "abl", "name": "abl.older-s2", "host": "hpc3", '
            '"partition": "free-gpu", "submitted_at": "2026-09-01T00:00:00+00:00", '
            '"log_dir": "/l", "deterministic": false, "experiment": {"rung": "774M"}, '
            '"image_digest": "", "submitter": "", "artifact": "/pub/a0.pt"}\n',
            encoding="utf-8",
        )
        fake_run.add("squeue --me", stdout="55678543_[2-3%2]|abl.older|PENDING\n")

        with pytest.raises(AppError) as caught:
            _submit(spec, (0,), tmp_path)

        assert caught.value.code is Hpc3ErrorCode.ARTIFACT_ALREADY_IN_FLIGHT
