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
    EXISTENCE_CHUNK,
    existence_commands,
    finished_artifacts,
    parse_existence,
    plan_campaign,
    require_every_member_declares_an_artifact,
)
from tests.against_hpc3 import decode_job_spec, decode_ledger_entry, read_ledger
from tests.conftest import (
    FakeRun,
    gpus,
    ledger_row,
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
    def test_a_small_campaign_fits_one_command(self) -> None:
        """Thirty members over SSH is otherwise thirty round trips, each able
        to fail halfway and leave the plan built from two moments."""
        commands = existence_commands([_TR, _AZ])
        assert len(commands) == 1
        assert commands[0].count("PRESENT|") == 1
        assert _TR in commands[0]
        assert _AZ in commands[0]

    def test_a_wide_campaign_chunks_below_the_argument_limit(self) -> None:
        """A 136-member search round packed one ~10 KB command, past
        cmd.exe's 8191-character limit on the Windows submitter -- the
        command reached bash truncated and died mid-loop (vhsearch2-r0).
        Order is preserved across chunks, and every path appears once."""
        artifacts = [f"/pub/wagnera3/rusted/runs/sweeps/wide/arm-s{i}.txt" for i in range(136)]
        commands = existence_commands(artifacts)
        assert len(commands) == 3
        assert all(len(command) < 4500 for command in commands)
        joined = "\n".join(commands)
        assert all(artifact in joined for artifact in artifacts)
        assert joined.index(artifacts[0]) < joined.index(artifacts[EXISTENCE_CHUNK])

    def test_a_path_carrying_a_space_is_quoted(self) -> None:
        assert "'/pub/a b.pt'" in existence_commands(["/pub/a b.pt"])[0]

    def test_no_artifacts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one artifact"):
            existence_commands([])

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


class TestWhichArtifactsAJobActuallyFinished:
    def test_a_completed_job_finishes_its_artifact(self) -> None:
        entries = [decode_ledger_entry(ledger_row(job_id="1", name="abl.tr", artifact=_TR))]
        assert finished_artifacts(entries, {"1": "COMPLETED"}) == {_TR}

    def test_a_preempted_job_does_not(self) -> None:
        """kk_best.pt after a preemption at 1273 seconds is a real file that
        no run finished."""
        entries = [decode_ledger_entry(ledger_row(job_id="1", name="abl.kk", artifact=_TR))]
        assert finished_artifacts(entries, {"1": "PREEMPTED"}) == set()

    def test_a_failed_or_cancelled_job_does_not_either(self) -> None:
        entries = [decode_ledger_entry(ledger_row(job_id="1", name="abl.kk", artifact=_TR))]
        for state in ("FAILED", "CANCELLED", "TIMEOUT"):
            assert finished_artifacts(entries, {"1": state}) == set()

    def test_a_job_with_no_state_has_no_claim(self) -> None:
        entries = [decode_ledger_entry(ledger_row(job_id="1", name="abl.tr", artifact=_TR))]
        assert finished_artifacts(entries, {}) == set()

    def test_a_job_declaring_no_artifact_finishes_nothing(self) -> None:
        entries = [decode_ledger_entry(ledger_row(job_id="1", name="abl.p6", artifact=None))]
        assert finished_artifacts(entries, {"1": "COMPLETED"}) == set()

    def test_one_completed_attempt_is_enough(self) -> None:
        """A member preempted twice and then finished is finished."""
        entries = [
            decode_ledger_entry(ledger_row(job_id="1", name="abl.kk", artifact=_TR)),
            decode_ledger_entry(ledger_row(job_id="2", name="abl.kk-r2", artifact=_TR)),
        ]
        assert finished_artifacts(entries, {"1": "PREEMPTED", "2": "COMPLETED"}) == {_TR}


class TestThePlan:
    def test_a_member_whose_artifact_exists_is_done(self) -> None:
        plan = plan_campaign([_spec("tr", _TR)], present={_TR}, finished={_TR}, claimed={})
        assert plan["done"] == ["abl.tr"]
        assert plan["missing"] == []

    def test_a_member_a_live_job_is_writing_is_in_flight(self) -> None:
        plan = plan_campaign(
            [_spec("uz", _TR)],
            present=set(),
            finished=set(),
            claimed={_TR: "abl.uz-r3"},
        )
        assert plan["in_flight"] == {"abl.uz": "abl.uz-r3"}
        assert plan["missing"] == []

    def test_a_member_that_is_neither_is_submitted(self) -> None:
        plan = plan_campaign([_spec("tr", _TR)], present=set(), finished=set(), claimed={})
        assert [s["name"] for s in plan["missing"]] == ["tr"]

    def test_a_checkpoint_from_a_killed_run_is_not_done(self) -> None:
        """The real defect, and the first thing this command got wrong.

        `bases-kk` was preempted at 1273 seconds having written `kk_best.pt`,
        because a training loop writes its best checkpoint whenever validation
        improves rather than at the end. The campaign reported it done and
        would have stopped resubmitting a member that was under-trained.
        """
        plan = plan_campaign([_spec("kk", _TR)], present={_TR}, finished=set(), claimed={})
        assert plan["done"] == []
        assert [s["name"] for s in plan["missing"]] == ["kk"]

    def test_a_finished_run_whose_output_is_gone_is_resubmitted(self) -> None:
        """The other half of the conjunction. Completion without the file is a
        result someone moved or deleted, and redoing it is right."""
        plan = plan_campaign([_spec("tr", _TR)], present=set(), finished={_TR}, claimed={})
        assert [s["name"] for s in plan["missing"]] == ["tr"]

    def test_in_flight_beats_present(self) -> None:
        """A file existing while a job writes it is a partially-written file,
        not a finished one."""
        plan = plan_campaign(
            [_spec("tr", _TR)],
            present={_TR},
            finished={_TR},
            claimed={_TR: "abl.tr-r2"},
        )
        assert plan["done"] == []
        assert plan["in_flight"] == {"abl.tr": "abl.tr-r2"}

    def test_the_three_groups_partition_the_members(self) -> None:
        specs = [_spec("tr", _TR), _spec("az", _AZ), _spec("kk", "/pub/kk.pt")]
        plan = plan_campaign(specs, present={_TR}, finished={_TR}, claimed={_AZ: "abl.az-r1"})
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

    def _ran(self, tmp_path: pathlib.Path, fake_run: FakeRun, *, finished: list[str]) -> None:
        """Record a job per artifact and say how each of them ended.

        Existence alone no longer means done, so a test that wants a member
        to read as finished has to supply what makes that true: a ledger row
        declaring the artifact, and an accounting state for that job.

        Args:
            tmp_path: Directory holding the ledger.
            fake_run: The runner to script the accounting reply on.
            finished: Artifacts whose job COMPLETED. Any other artifact gets
                a job that was PREEMPTED, which is the kk_best.pt case: a
                real file written by a run that did not finish.
        """
        rows = [
            ledger_row(job_id=f"90{index}", name=f"abl.prior-{index}", artifact=artifact)
            for index, artifact in enumerate((_TR, _AZ))
        ]
        write_file(
            tmp_path / "ledger.jsonl",
            b"".join(dump_json_str(row).encode("utf-8") + b"\n" for row in rows),
        )
        fake_run.add(
            "sacct",
            stdout="".join(
                f"90{index}|abl.prior-{index}|free-gpu|"
                f"{'COMPLETED' if artifact in finished else 'PREEMPTED'}"
                "|60|billing=8,gres/gpu=1|n1\n"
                for index, artifact in enumerate((_TR, _AZ))
            ),
        )

    def test_it_submits_only_what_is_missing(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        self._ran(tmp_path, fake_run, finished=[_TR])
        script_healthy_cluster(fake_run)

        assert campaign_cli.main(self._document(tmp_path)) == 0
        assert emitted[0] == "done      abl.bases-tr"
        # _1, not _0: the sparse array selects by DOCUMENT position, and az
        # is the document's second member -- the property that keeps the
        # task-to-member mapping identical across convergence passes.
        assert emitted[1] == "submitted 55519937_1 abl.bases-az"
        assert emitted[-1] == "1 done, 0 in flight, 1 submitted, 1 remaining"

    def test_only_the_missing_member_reaches_the_ledger(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        self._ran(tmp_path, fake_run, finished=[_TR])
        script_healthy_cluster(fake_run)

        campaign_cli.main(self._document(tmp_path))
        names = [e["name"] for e in read_ledger(tmp_path / "ledger.jsonl")]
        assert names == ["abl.prior-0", "abl.prior-1", "abl.bases-az"]

    def test_a_finished_campaign_submits_nothing_and_succeeds(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """Convergence is the success case. A command that exited non-zero
        once finished could not be run on a schedule."""
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nPRESENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        self._ran(tmp_path, fake_run, finished=[_TR, _AZ])

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
        fake_run.add("PRESENT|", stdout=f"PRESENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="55646157|abl.bases-az-r2|RUNNING\n")
        # tr is genuinely finished; the live az-r2 row is appended after, so
        # az reads as in flight rather than as anything else.
        self._ran(tmp_path, fake_run, finished=[_TR])
        path = tmp_path / "ledger.jsonl"
        write_file(path, path.read_bytes() + ledger_line.encode("utf-8") + b"\n")

        assert campaign_cli.main(args) == 0
        assert emitted[1] == "in flight abl.bases-az <- abl.bases-az-r2"
        assert not any("sbatch" in command for command in fake_run.commands())

    def test_a_campaign_no_job_has_ever_touched_asks_accounting_nothing(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """The first run of a new experiment: an empty ledger means there is
        no job to ask about, and `sacct` with no ids would report the whole
        cluster's history rather than nothing."""
        fake_run.add("PRESENT|", stdout=f"ABSENT|{_TR}\nABSENT|{_AZ}\n")
        fake_run.add("squeue --me", stdout="")
        script_healthy_cluster(fake_run)

        assert campaign_cli.main(self._document(tmp_path)) == 0
        assert not any(command.startswith("sacct") for command in fake_run.commands())
        assert emitted[-1] == "0 done, 0 in flight, 2 submitted, 2 remaining"

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
        self._ran(tmp_path, fake_run, finished=[_TR])
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
        self._ran(tmp_path, fake_run, finished=[_TR, _AZ])
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
