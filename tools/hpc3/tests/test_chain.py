"""Tests for chains: the contract, the resolution, and the submission wiring.

The assertions that matter are about ORDER and about what happens before the
first job exists. A chain that validated stage three only after stage one was
running would be a pipeline that discovers a typo an hour in, and a chain that
wired its dependencies in the wrong direction would run the stages backwards
while reporting success.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.chain import MINIMUM_STAGES, decode_chain_spec, encode_chain_spec
from hpc3.contracts.run import resolve_chain
from hpc3.contracts.workspace import Workspace, decode_workspace
from hpc3.core import audit
from hpc3.core.chain import submit_chain
from tests.against_hpc3 import read_ledger
from tests.conftest import (
    PREFLIGHT_LINE,
    FakeRun,
    LoggedEvent,
    cluster,
    project_config,
    workspace_document,
)

_AT = "2026-08-23T16:00:00+00:00"


def _stage(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build one chain stage document.

    Args:
        **overrides: Fields to replace.

    Returns:
        A stage document.
    """
    base: dict[str, JSONValue] = {
        "suffix": "one",
        "command": "python one.py",
        "artifact": None,
    }
    base.update(overrides)
    return base


def _document(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid chain document.

    Args:
        **overrides: Fields to replace.

    Returns:
        A chain document ready for resolution.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "pipeline",
        "experiment": {"sample": "batch7"},
        "stages": [
            _stage(),
            _stage(suffix="two", command="python two.py"),
        ],
    }
    base.update(overrides)
    return base


def _workspace(**overrides: JSONValue) -> Workspace:
    """Build a workspace whose single project is free and GPU-backed.

    Args:
        **overrides: Project fields to replace.

    Returns:
        The decoded workspace.
    """
    return decode_workspace(
        workspace_document(projects={"abl": project_config(**overrides)}),
        config_dir=pathlib.Path("/tmp"),
    )


def _healthy(fake_run: FakeRun, ids: tuple[str, ...]) -> None:
    """Script a cluster that accepts every stage.

    Args:
        fake_run: The command recorder to script.
        ids: Job ids to hand back, in submission order.
    """
    fake_run.add("test -d", stdout="PRESENT\n")
    fake_run.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")
    for job_id in ids:
        # "&& sbatch " rather than "sbatch", because the UPLOAD command is
        # `cat > '.../abl.pipeline-one.sbatch'` and a bare substring matches
        # the filename too -- which consumed the rule before the submission
        # that needed it. once=True because every stage issues an identical
        # command, and a rule that stayed would hand the same id to all of
        # them, letting a chain that wires every stage to the first one pass.
        fake_run.add("&& sbatch ", stdout=f"Submitted batch job {job_id}\n", once=True)


class TestResolveChain:
    def test_each_stage_is_named_by_the_chain_plus_its_suffix(self) -> None:
        """So two stages of one pipeline sort together in squeue."""
        stages = resolve_chain(_workspace(), _document())["stages"]
        assert [stage["name"] for stage in stages] == ["pipeline-one", "pipeline-two"]

    def test_stages_keep_their_declared_order(self) -> None:
        stages = resolve_chain(_workspace(), _document())["stages"]
        assert [stage["command"] for stage in stages] == ["python one.py", "python two.py"]

    def test_no_stage_carries_a_dependency_yet(self) -> None:
        """The ids do not exist until something has been submitted."""
        stages = resolve_chain(_workspace(), _document())["stages"]
        assert [stage["depends_on"] for stage in stages] == [None, None]

    def test_stages_inherit_the_projects_defaults(self) -> None:
        stages = resolve_chain(_workspace(), _document())["stages"]
        assert {stage["env_path"] for stage in stages} == {"/opt/env"}

    def test_a_stage_may_differ_in_resources_from_its_neighbour(self) -> None:
        """The reason a chain is not a sweep: a training stage holds a GPU and
        the evaluation reading its checkpoints often does not."""
        document = _document(
            stages=[
                _stage(cpus=4),
                _stage(suffix="two", command="python two.py", partition="free", gpu=None, cpus=32),
            ]
        )
        stages = resolve_chain(_workspace(), document)["stages"]
        assert [stage["cpus"] for stage in stages] == [4, 32]
        assert [stage["gpu"] for stage in stages] == [{"model": "A100", "count": 1}, None]
        assert [stage["partition"] for stage in stages] == ["free-gpu", "free"]

    def test_the_chain_layer_sits_between_project_and_stage(self) -> None:
        """Chain-wide overrides apply to stages that do not restate them."""
        document = _document(cpus=12, stages=[_stage(), _stage(suffix="two", cpus=20)])
        stages = resolve_chain(_workspace(), document)["stages"]
        assert [stage["cpus"] for stage in stages] == [12, 20]

    def test_every_stage_carries_the_chains_experiment_plus_its_own_label(self) -> None:
        """Two stages sharing one experiment record would be two rows the
        ledger cannot tell apart."""
        stages = resolve_chain(_workspace(), _document())["stages"]
        assert [stage["experiment"] for stage in stages] == [
            {"sample": "batch7", "stage": "one"},
            {"sample": "batch7", "stage": "two"},
        ]


class TestChainRefusals:
    def test_a_single_stage_is_refused_as_a_run(self) -> None:
        with pytest.raises(JSONTypeError, match="at least 2 stages"):
            resolve_chain(_workspace(), _document(stages=[_stage()]))

    def test_the_minimum_is_two(self) -> None:
        assert MINIMUM_STAGES == 2

    def test_a_stage_may_not_write_its_own_dependency(self) -> None:
        """The chain wires its own, so a hand-written one would be silently
        replaced. Refused rather than overwritten."""
        payload: JSONValue = {
            "stages": [
                {"depends_on": {"kind": "afterok", "job_ids": ["1"]}},
                {},
            ]
        }
        with pytest.raises(JSONTypeError, match="declares 'depends_on'"):
            decode_chain_spec(payload, HPC3, max_service_units=0.0)

    def test_two_stages_sharing_a_suffix_are_refused(self) -> None:
        document = _document(stages=[_stage(), _stage()])
        with pytest.raises(JSONTypeError, match="must not repeat a name"):
            resolve_chain(_workspace(), document)

    def test_a_stage_without_a_suffix_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="'suffix'"):
            resolve_chain(_workspace(), _document(stages=[{"command": "x"}, _stage()]))

    def test_a_slashed_suffix_is_refused(self) -> None:
        """The suffix reaches a log filename; a separator would escape it."""
        with pytest.raises(JSONTypeError, match="path separator"):
            resolve_chain(_workspace(), _document(stages=[_stage(suffix="a/b"), _stage()]))

    def test_an_unknown_stage_field_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_chain(_workspace(), _document(stages=[_stage(minute=60), _stage()]))
        assert excinfo.value.code is Hpc3ErrorCode.RUN_FIELD_UNKNOWN

    def test_a_broken_later_stage_stops_the_whole_chain(self) -> None:
        """The point of validating up front: this would otherwise surface an
        hour after stage one started running."""
        document = _document(
            stages=[_stage(), _stage(suffix="two", partition="nonesuch")],
        )
        with pytest.raises(AppError) as excinfo:
            resolve_chain(_workspace(), document)
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN

    def test_a_stage_on_a_billing_partition_is_refused(self) -> None:
        billing = _stage(suffix="two", partition="standard", gpu=None)
        document = _document(stages=[_stage(), billing])
        with pytest.raises(AppError) as excinfo:
            resolve_chain(_workspace(), document)
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_BILLS

    def test_a_non_object_chain_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_chain_spec([1, 2], HPC3, max_service_units=0.0)

    def test_a_missing_stage_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="'stages'"):
            decode_chain_spec({}, HPC3, max_service_units=0.0)


class TestEncodeChain:
    def test_a_chain_round_trips_through_its_stages(self) -> None:
        spec = resolve_chain(_workspace(), _document())
        encoded = encode_chain_spec(spec)
        assert decode_chain_spec(encoded, HPC3, max_service_units=0.0) == spec


class TestSubmitChain:
    def _submit(self, tmp_path: pathlib.Path) -> list[str]:
        """Submit the standard two-stage chain.

        Args:
            tmp_path: Directory for the ledger.

        Returns:
            The ids assigned, in order.
        """
        spec = resolve_chain(_workspace(), _document())
        submitted = submit_chain(
            spec,
            host="hpc3",
            script_dir="/pub/wagnera3/jobs",
            log_dir="/pub/wagnera3/logs",
            ledger_path=tmp_path / "ledger.jsonl",
            submitted_at=_AT,
            cluster=cluster(),
            charge_account="",
        )
        return [member.job_id for member in submitted]

    def test_stages_are_submitted_in_order(self, tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
        _healthy(fake_run, ("101", "102"))
        assert self._submit(tmp_path) == ["101", "102"]

    def test_the_first_stage_waits_on_nothing(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _healthy(fake_run, ("101", "102"))
        self._submit(tmp_path)
        first = next(c.stdin_bytes for c in fake_run.calls if c.stdin_bytes is not None)
        assert b"--dependency" not in first

    def test_the_second_stage_waits_on_the_first_by_its_real_id(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _healthy(fake_run, ("101", "102"))
        self._submit(tmp_path)
        uploads = [c.stdin_bytes for c in fake_run.calls if c.stdin_bytes is not None]
        assert b"#SBATCH --dependency=afterok:101" in uploads[1]

    def test_the_wait_is_afterok_not_afterany(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A stage that runs after its input failed computes a second wrong
        answer and spends the wall clock doing it."""
        _healthy(fake_run, ("101", "102"))
        self._submit(tmp_path)
        uploads = [c.stdin_bytes for c in fake_run.calls if c.stdin_bytes is not None]
        assert b"afterany" not in uploads[1]
        assert b"--kill-on-invalid-dep=yes" in uploads[1]

    def test_every_stage_lands_in_the_ledger(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _healthy(fake_run, ("101", "102"))
        self._submit(tmp_path)
        recorded = read_ledger(tmp_path / "ledger.jsonl")
        assert [entry["job_id"] for entry in recorded] == ["101", "102"]
        assert [entry["name"] for entry in recorded] == ["abl.pipeline-one", "abl.pipeline-two"]

    def test_a_stage_that_fails_leaves_the_earlier_ones_findable(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """Nothing is rolled back: stage one is a real job and it is fine."""
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")
        fake_run.add("&& sbatch ", stdout="Submitted batch job 101\n", once=True)
        fake_run.add("&& sbatch ", returncode=1, stderr="Invalid account\n", once=True)

        with pytest.raises(AppError) as excinfo:
            self._submit(tmp_path)
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
        assert [e["job_id"] for e in read_ledger(tmp_path / "ledger.jsonl")] == ["101"]

    def test_it_records_the_pipeline_and_its_order(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        """Order is the content: which stage waited on which is not
        recoverable once the jobs age out of squeue."""
        _healthy(fake_run, ("101", "102"))
        self._submit(tmp_path)
        chains = [event for event in logged if event.event == audit.CHAIN_SUBMITTED]
        assert len(chains) == 1
        assert chains[0].fields["job_ids"] == "101,102"
        assert chains[0].fields["stages"] == 2
