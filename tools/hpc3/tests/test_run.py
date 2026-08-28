"""Tests for resolving a run document against a workspace.

Two properties matter here and they pull in opposite directions: a run
document should be short, and a short document must not be able to reach a
job a full spec could not. Every override therefore lands in an ordinary job
object that goes through the same decoder a hand-written spec does.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.run import resolve_run, resolve_sweep
from hpc3.contracts.sweep import expand_sweep
from hpc3.contracts.workspace import Workspace, decode_workspace
from tests.conftest import project_config, workspace_document

_DIR = pathlib.Path("/tmp/ws")


def _workspace(**project_overrides: JSONValue) -> Workspace:
    """Build a one-project workspace.

    Args:
        **project_overrides: Fields to replace in the project's defaults.

    Returns:
        The decoded workspace.
    """
    return decode_workspace(
        workspace_document(projects={"abl": project_config(**project_overrides)}),
        config_dir=_DIR,
    )


def _run(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a run document.

    Args:
        **overrides: Fields to add or replace.

    Returns:
        The document.
    """
    document: dict[str, JSONValue] = {
        "project": "abl",
        "name": "armB-s42",
        "command": "python train.py --arm B",
        "artifact": None,
        "experiment": {"arm": "B", "seed": "42"},
    }
    document.update(overrides)
    return document


def _sweep(count: int = 3, **overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a sweep document.

    Args:
        count: How many members.
        **overrides: Fields to add or replace.

    Returns:
        The document.
    """
    members: list[JSONValue] = [
        {"suffix": f"s{i}", "command": f"python train.py --seed {i}", "artifact": None}
        for i in range(count)
    ]
    document: dict[str, JSONValue] = {
        "project": "abl",
        "name": "rung",
        "members": members,
        "experiment": {"rung": "774M"},
    }
    document.update(overrides)
    return document


class TestResolveRun:
    def test_a_four_field_document_becomes_a_complete_spec(self) -> None:
        """The whole point: a run says what differs, not what it inherits."""
        spec = resolve_run(_workspace(), _run())
        assert spec == {
            "project": "abl",
            "name": "armB-s42",
            "partition": "free-gpu",
            "gpu": {"model": "A100", "count": 1},
            "cpus": 8,
            "mem_gb": 96,
            "minutes": 30,
            "requeue": False,
            "checkpoint_steps": 0,
            "image": {
                "path": "/pub/images/v1/abl.sif",
                "sha256": "a" * 64,
                "binds": ["/pub"],
            },
            "env_path": "/opt/env",
            "pinned_packages": {},
            "deterministic": False,
            "depends_on": None,
            "experiment": {"arm": "B", "seed": "42"},
            "command": "python train.py --arm B",
            "artifact": None,
        }

    def test_changing_a_project_default_changes_every_run(self) -> None:
        """One edit, not one per document -- the reason this layer exists."""
        moved = resolve_run(_workspace(env_path="/opt/env-next"), _run())
        assert moved["env_path"] == "/opt/env-next"

    def test_an_override_wins_over_the_default(self) -> None:
        assert resolve_run(_workspace(), _run(cpus=16))["cpus"] == 16

    def test_an_override_does_not_leak_into_the_next_run(self) -> None:
        workspace = _workspace()
        resolve_run(workspace, _run(cpus=16))
        assert resolve_run(workspace, _run())["cpus"] == 8

    def test_several_overrides_apply_together(self) -> None:
        spec = resolve_run(_workspace(), _run(minutes=600, requeue=True, checkpoint_steps=500))
        assert (spec["minutes"], spec["requeue"], spec["checkpoint_steps"]) == (600, True, 500)

    def test_an_undeclared_project_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(project="sirius"))
        assert excinfo.value.code is Hpc3ErrorCode.WORKSPACE_PROJECT_UNKNOWN

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="run must be a JSON object"):
            resolve_run(_workspace(), "armB")

    def test_a_missing_name_is_refused(self) -> None:
        document = _run()
        del document["name"]
        with pytest.raises(JSONTypeError):
            resolve_run(_workspace(), document)

    def test_a_missing_command_is_refused(self) -> None:
        document = _run()
        del document["command"]
        with pytest.raises(JSONTypeError):
            resolve_run(_workspace(), document)


class TestUnknownFieldsAreRefused:
    def test_a_misspelled_override_is_refused_not_ignored(self) -> None:
        """'minute' would silently run at the project default instead."""
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(minute=600))
        assert excinfo.value.code is Hpc3ErrorCode.RUN_FIELD_UNKNOWN
        assert "minute" in excinfo.value.message

    def test_the_message_names_what_is_accepted(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(gpu_count=2))
        assert "'gpu'" in excinfo.value.message

    def test_every_unknown_field_is_named_at_once(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(minute=600, cpu=4))
        assert "['cpu', 'minute']" in excinfo.value.message

    def test_a_sweep_field_is_not_accepted_by_a_run(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(members=[]))
        assert excinfo.value.code is Hpc3ErrorCode.RUN_FIELD_UNKNOWN

    def test_a_run_field_is_not_accepted_by_a_sweep(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_sweep(_workspace(), _sweep(command="python train.py"))
        assert excinfo.value.code is Hpc3ErrorCode.RUN_FIELD_UNKNOWN


class TestOverridingCannotEvadeARule:
    def test_a_long_preemptible_override_still_needs_protection(self) -> None:
        """The rule applies to the merged result, not to the document."""
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(minutes=600))
        assert excinfo.value.code is Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED

    def test_overriding_onto_a_free_32gb_partition_is_admitted(self) -> None:
        """It was refused as billing until the factor was measured properly."""
        resolved = resolve_run(
            _workspace(),
            _run(partition="free-gpu32", gpu={"model": "L40S", "count": 1}),
        )
        assert resolved["partition"] == "free-gpu32"

    def test_a_project_declaring_a_billing_partition_is_refused_at_resolution(self) -> None:
        """The project cannot consent on the run's behalf, because there is
        nothing to consent with."""
        workspace = _workspace(partition="standard", gpu=None)
        with pytest.raises(AppError) as excinfo:
            resolve_run(workspace, _run())
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_BILLS

    def test_an_override_onto_a_billing_partition_is_refused(self) -> None:
        """The rule applies to the merged result, so a free project cannot be
        overridden onto a partition that charges."""
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(partition="gpu"))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_BILLS

    def test_an_override_onto_a_partition_without_that_gpu_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(gpu={"model": "L40S", "count": 1}))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_GPU_MISMATCH

    def test_overriding_a_gpu_project_to_cpu_only_is_refused(self) -> None:
        """The merged result would hold a GPU node to do CPU work, and Slurm
        would accept it -- so the refusal has to happen here."""
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(gpu=None))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_GPU_MISMATCH

    def test_overriding_both_partition_and_gpu_reaches_cpu_only(self) -> None:
        """The pair that IS coherent: a run may move to CPU work outright."""
        resolved = resolve_run(_workspace(), _run(partition="free", gpu=None))
        assert resolved["gpu"] is None
        assert resolved["partition"] == "free"

    def test_an_override_past_the_partition_ceiling_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(minutes=5000, requeue=True, checkpoint_steps=10))
        assert excinfo.value.code is Hpc3ErrorCode.TIME_LIMIT_EXCEEDS_PARTITION

    def test_an_override_to_a_generic_gpu_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(gpu={"model": "gpu", "count": 1}))
        assert excinfo.value.code is Hpc3ErrorCode.GPU_TYPE_UNPINNED


class TestResolveSweep:
    def test_every_member_inherits_the_projects_defaults(self) -> None:
        specs = expand_sweep(resolve_sweep(_workspace(), _sweep()))
        assert [s["name"] for s in specs] == ["rung-s0", "rung-s1", "rung-s2"]
        assert {s["cpus"] for s in specs} == {8}
        assert {s["project"] for s in specs} == {"abl"}

    def test_each_member_keeps_its_own_command(self) -> None:
        specs = expand_sweep(resolve_sweep(_workspace(), _sweep()))
        assert [s["command"] for s in specs] == [
            "python train.py --seed 0",
            "python train.py --seed 1",
            "python train.py --seed 2",
        ]

    def test_an_override_applies_to_the_whole_sweep(self) -> None:
        specs = expand_sweep(
            resolve_sweep(_workspace(), _sweep(minutes=600, requeue=True, checkpoint_steps=250))
        )
        assert {s["minutes"] for s in specs} == {600}
        assert {s["checkpoint_steps"] for s in specs} == {250}

    def test_a_sweep_past_the_qos_ceiling_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_sweep(_workspace(), _sweep(count=25))
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="sweep must be a JSON object"):
            resolve_sweep(_workspace(), "rung")

    def test_no_members_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            resolve_sweep(_workspace(), _sweep(members=[]))

    def test_a_missing_member_list_is_refused(self) -> None:
        document = _sweep()
        del document["members"]
        with pytest.raises(JSONTypeError):
            resolve_sweep(_workspace(), document)

    def test_a_non_object_member_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            resolve_sweep(_workspace(), _sweep(members=["s0"]))

    def test_a_repeated_suffix_is_refused(self) -> None:
        """Two jobs sharing a name would interleave into one log file."""
        members: list[JSONValue] = [
            {"suffix": "s0", "command": "python a.py", "artifact": None},
            {"suffix": "s0", "command": "python b.py", "artifact": None},
        ]
        with pytest.raises(JSONTypeError, match="must not repeat a suffix"):
            resolve_sweep(_workspace(), _sweep(members=members))

    def test_an_undeclared_project_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_sweep(_workspace(), _sweep(project="zodiac"))
        assert excinfo.value.code is Hpc3ErrorCode.WORKSPACE_PROJECT_UNKNOWN


class TestARunMayNotRemoveTheProjectsImage:
    """Overriding the image is normal; deleting it would switch off the rule.

    ``_check_gpu_project_is_imaged`` refuses to onboard GPU work with no
    image. Without this, that refusal would be one line of JSON away from
    being bypassed per submission, which is not a rule.
    """

    def test_a_run_nulling_the_image_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(image=None))
        assert excinfo.value.code is Hpc3ErrorCode.RUN_REMOVES_IMAGE

    def test_the_refusal_says_to_omit_the_field_instead(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(image=None))
        assert "Omit the field" in str(excinfo.value)

    def test_a_run_pinning_a_different_image_is_admitted(self) -> None:
        """Rolling an image version forward one experiment at a time is the
        reason overriding stays legal."""
        newer: JSONValue = {
            "path": "/pub/images/v2/abl.sif",
            "sha256": "b" * 64,
            "binds": ["/pub"],
        }
        spec = resolve_run(_workspace(), _run(image=newer))
        assert spec["image"] == newer

    def test_a_run_omitting_the_field_inherits_the_projects_image(self) -> None:
        spec = resolve_run(_workspace(), _run())
        image = spec["image"]
        if image is None:
            raise AssertionError("the run did not inherit the project's image")
        assert image["path"] == "/pub/images/v1/abl.sif"

    def test_a_cpu_project_with_no_image_is_unaffected(self) -> None:
        """Nothing to remove, so nothing to refuse."""
        spec = resolve_run(
            _workspace(partition="free", gpu=None, image=None, env_path="/pub/envs/cleargbm"),
            _run(image=None),
        )
        assert spec["image"] is None
