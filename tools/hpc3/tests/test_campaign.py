"""Tests for converging a set of runs on a declared end state.

THE SITUATION THAT MOTIVATED IT, on 2026-08-28: ``free-gpu`` preempted five of
seven members inside an hour, and nothing in the package could say which five.
What followed was four hand-written resume documents describing one
experiment, each a transcription of a queue state that had already changed, and
at one point two of them were live writing the same checkpoint.

THE FIRST LIVE RUN, against the real cluster the same evening, with two members
finished and five resumed under different names::

    done      turkic-lstm.bases-tr
    done      turkic-lstm.bases-az
    in flight turkic-lstm.bases-kk <- turkic-lstm.bases-r1-kk
    in flight turkic-lstm.bases-uz <- turkic-lstm.bases-uz-r3
    ...
    2 done, 5 in flight, 0 submitted, 5 remaining

Nothing submitted, and every resume recognised as covering the member it
resumes -- because the ARTIFACT is the identity, not the job name.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.cli import campaign as campaign_cli
from hpc3.contracts.job import JobSpec
from hpc3.core.campaign import (
    existence_command,
    parse_existence,
    plan_campaign,
    require_every_member_declares_an_artifact,
)
from tests.against_hpc3 import decode_job_spec, read_ledger
from tests.conftest import (
    FakeRun,
    gpus,
    project_config,
    script_healthy_cluster,
    workspace_document,
    write_file,
    write_workspace,
)

_TR = "/pub/wagnera3/LSTM/checkpoints/tr_best.pt"
_AZ = "/pub/wagnera3/LSTM/checkpoints/az_best.pt"


def _spec(name: str, artifact: str | None) -> JobSpec:
    """Build a decoded member spec.

    Args:
        name: The member's name within its project.
        artifact: Path it writes, or None.

    Returns:
        A validated spec.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": name,
        "partition": "free-gpu",
        "gpu": gpus("A100"),
        "cpus": 4,
        "mem_gb": 16,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "env_path": "/pub/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B"},
        "command": "python train.py" if artifact is None else f"python train.py --out {artifact}",
        "artifact": artifact,
    }
    return decode_job_spec(base)


class TestEveryMemberMustDeclareAnArtifact:
    """Otherwise the campaign has no definition of finished."""

    def test_declared_artifacts_are_returned_in_order(self) -> None:
        specs = [_spec("tr", _TR), _spec("az", _AZ)]
        assert require_every_member_declares_an_artifact(specs) == [_TR, _AZ]

    def test_a_member_without_one_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            require_every_member_declares_an_artifact([_spec("tr", _TR), _spec("az", None)])
        assert excinfo.value.code is Hpc3ErrorCode.CAMPAIGN_MEMBER_HAS_NO_ARTIFACT

    def test_the_refusal_names_the_member_and_the_right_command(self) -> None:
        """Every cleargbm member runs --no-save-model and declares null,
        correctly. Those sweeps are good sweeps and cannot be campaigns."""
        with pytest.raises(AppError) as excinfo:
            require_every_member_declares_an_artifact([_spec("p6-1", None)])
        assert "abl.p6-1" in str(excinfo.value)
        assert "hpc3-sweep" in str(excinfo.value)


class TestAskingWhichArtifactsExist:
    def test_one_command_covers_every_member(self) -> None:
        """Thirty members over SSH is otherwise thirty round trips, each able
        to fail halfway and leave the plan built from two moments."""
        command = existence_command([_TR, _AZ])
        assert command.count("PRESENT|") == 1
        assert _TR in command
        assert _AZ in command

    def test_a_path_carrying_a_space_is_quoted(self) -> None:
        assert "'/pub/a b.pt'" in existence_command(["/pub/a b.pt"])

    def test_no_artifacts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one artifact"):
            existence_command([])

    def test_present_lines_are_read(self) -> None:
        assert parse_existence(f"PRESENT|{_TR}\nABSENT|{_AZ}\n") == {_TR}

    def test_blank_lines_are_skipped(self) -> None:
        assert parse_existence(f"\nPRESENT|{_TR}\n\n") == {_TR}

    def test_nothing_present_reads_as_empty(self) -> None:
        assert parse_existence(f"ABSENT|{_TR}\nABSENT|{_AZ}\n") == set()

    def test_an_unreadable_line_is_refused_rather_than_read_as_absent(self) -> None:
        """Unknown treated as absent would resubmit a finished job straight
        into the artifact it wrote."""
        with pytest.raises(AppError) as excinfo:
            parse_existence("bash: no such thing\n")
        assert excinfo.value.code is Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE


class TestThePlan:
    def test_a_member_whose_artifact_exists_is_done(self) -> None:
        plan = plan_campaign([_spec("tr", _TR)], present={_TR}, claimed={})
        assert plan["done"] == ["abl.tr"]
        assert plan["missing"] == []

    def test_a_member_a_live_job_is_writing_is_in_flight(self) -> None:
        plan = plan_campaign([_spec("uz", _TR)], present=set(), claimed={_TR: "abl.uz-r3"})
        assert plan["in_flight"] == {"abl.uz": "abl.uz-r3"}
        assert plan["missing"] == []

    def test_a_member_that_is_neither_is_submitted(self) -> None:
        plan = plan_campaign([_spec("tr", _TR)], present=set(), claimed={})
        assert [s["name"] for s in plan["missing"]] == ["tr"]

    def test_in_flight_beats_present(self) -> None:
        """A file existing while a job writes it is a partially-written file,
        not a finished one."""
        plan = plan_campaign([_spec("tr", _TR)], present={_TR}, claimed={_TR: "abl.tr-r2"})
        assert plan["done"] == []
        assert plan["in_flight"] == {"abl.tr": "abl.tr-r2"}

    def test_the_three_groups_partition_the_members(self) -> None:
        specs = [_spec("tr", _TR), _spec("az", _AZ), _spec("kk", "/pub/kk.pt")]
        plan = plan_campaign(specs, present={_TR}, claimed={_AZ: "abl.az-r1"})
        assert plan["done"] == ["abl.tr"]
        assert plan["in_flight"] == {"abl.az": "abl.az-r1"}
        assert [s["name"] for s in plan["missing"]] == ["kk"]


class TestTheCommand:
    def _document(self, tmp_path: pathlib.Path) -> list[str]:
        """Write a two-member campaign and return its arguments.

        Args:
            tmp_path: Directory to write into.

        Returns:
            Arguments excluding the program name.
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
        write_file(tmp_path / "sweep.json", dump_json_str(document).encode("utf-8"))
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        return ["--config", config, "--run", str(tmp_path / "sweep.json")]

    def test_it_submits_only_what_is_missing(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        script_healthy_cluster(fake_run)

        assert campaign_cli.main(self._document(tmp_path)) == 0
        assert emitted[0] == "done      abl.bases-tr"
        assert emitted[1] == "submitted 55519937 abl.bases-az"
        assert emitted[-1] == "1 done, 0 in flight, 1 submitted, 1 remaining"

    def test_only_the_missing_member_reaches_the_ledger(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        script_healthy_cluster(fake_run)

        campaign_cli.main(self._document(tmp_path))
        assert [e["name"] for e in read_ledger(tmp_path / "ledger.jsonl")] == ["abl.bases-az"]

    def test_a_finished_campaign_submits_nothing_and_succeeds(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """Convergence is the success case. A command that exited non-zero
        once finished could not be run on a schedule."""
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nPRESENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")

        assert campaign_cli.main(self._document(tmp_path)) == 0
        assert emitted[-1] == "2 done, 0 in flight, 0 submitted, 0 remaining"
        assert not any("sbatch" in command for command in fake_run.commands())

    def test_a_member_a_live_job_is_writing_is_left_alone(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """The uz_best.pt race, prevented by construction rather than by a
        check that fires after someone has already typed the command."""
        ledger_line = dump_json_str(
            {
                "job_id": "55646157",
                "project": "abl",
                "name": "abl.bases-az-r2",
                "host": "hpc3",
                "partition": "free-gpu",
                "submitted_at": "2026-08-28T21:00:00+00:00",
                "log_dir": "/pub/logs",
                "deterministic": False,
                "experiment": {"corpus": "v3"},
                "image_digest": "",
                "artifact": _AZ,
            }
        )
        args = self._document(tmp_path)
        write_file(tmp_path / "ledger.jsonl", ledger_line.encode("utf-8") + b"\n")
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="55646157|abl.bases-az-r2|RUNNING\n")

        assert campaign_cli.main(args) == 0
        assert emitted[1] == "in flight abl.bases-az <- abl.bases-az-r2"
        assert not any("sbatch" in command for command in fake_run.commands())

    def test_a_member_with_no_artifact_is_refused_before_any_query(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        members: list[JSONValue] = [
            {"suffix": "s1", "command": "python train.py --no-save-model", "artifact": None}
        ]
        document: dict[str, JSONValue] = {
            "project": "abl",
            "name": "rung",
            "members": members,
            "experiment": {"rung": "774M"},
        }
        write_file(tmp_path / "sweep.json", dump_json_str(document).encode("utf-8"))
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())

        with pytest.raises(AppError) as excinfo:
            campaign_cli.main(["--config", config, "--run", str(tmp_path / "sweep.json")])
        assert excinfo.value.code is Hpc3ErrorCode.CAMPAIGN_MEMBER_HAS_NO_ARTIFACT
        assert fake_run.calls == []

    def test_the_budget_is_projected_over_the_gap_only(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """Budgeting for finished members would refuse the last member of a
        long experiment for the cost of ones that paid for themselves."""
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
        write_file(tmp_path / "sweep.json", dump_json_str(document).encode("utf-8"))
        # 0.6 GPU-hours holds ONE 30-minute one-GPU member but not two.
        config = write_workspace(
            tmp_path / "hpc3.json",
            workspace_document(
                projects={
                    "abl": _project_with_cap(0.6),
                }
            ),
        )
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        script_healthy_cluster(fake_run)

        assert campaign_cli.main(["--config", config, "--run", str(tmp_path / "sweep.json")]) == 0
        assert emitted[-1] == "1 done, 0 in flight, 1 submitted, 1 remaining"

    def test_the_config_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            campaign_cli.main(["--run", str(tmp_path / "sweep.json")])

    def test_the_run_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        with pytest.raises(ValueError, match="--run is required"):
            campaign_cli.main(["--config", config])

    def test_the_entrypoint_reads_the_process_arguments(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        argv: list[str],
        frozen_clock: str,
    ) -> None:
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nPRESENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        argv[:] = ["prog", *self._document(tmp_path)]
        with pytest.raises(SystemExit) as excinfo:
            campaign_cli.entrypoint()
        assert excinfo.value.code == 0


def _project_with_cap(gpu_hours: float) -> JSONValue:
    """Build a project config carrying a specific GPU-hour cap.

    Args:
        gpu_hours: The self-imposed ceiling.

    Returns:
        The project entry.
    """
    return project_config(
        budget={
            "self_imposed_gpu_hours": gpu_hours,
            "max_service_units": 0.0,
            "charge_account": "",
        }
    )
