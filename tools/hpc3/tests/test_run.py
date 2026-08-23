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
        {"suffix": f"s{i}", "command": f"python train.py --seed {i}"} for i in range(count)
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
            "gpu": "A100",
            "gpu_count": 1,
            "cpus": 8,
            "mem_gb": 96,
            "minutes": 30,
            "requeue": False,
            "checkpoint_steps": 0,
            "accept_billing": False,
            "env_path": "/pub/envs/abl-pinned",
            "pinned_packages": {},
            "experiment": {"arm": "B", "seed": "42"},
            "command": "python train.py --arm B",
        }

    def test_changing_a_project_default_changes_every_run(self) -> None:
        """One edit, not one per document -- the reason this layer exists."""
        moved = resolve_run(_workspace(env_path="/pub/envs/abl-next"), _run())
        assert moved["env_path"] == "/pub/envs/abl-next"

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
            resolve_run(_workspace(), _run(gpus=2))
        assert "'gpu_count'" in excinfo.value.message

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

    def test_overriding_onto_a_billing_partition_still_needs_consent(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(partition="free-gpu32", gpu="L40S"))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_BILLS_WITHOUT_CONSENT

    def test_a_project_consenting_to_billing_carries_that_consent(self) -> None:
        """Whether a body of work may cost money is a property of the work."""
        workspace = _workspace(partition="free-gpu32", gpu="L40S", accept_billing=True)
        assert resolve_run(workspace, _run())["accept_billing"] is True

    def test_an_override_onto_a_partition_without_that_gpu_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(gpu="L40S"))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_GPU_MISMATCH

    def test_an_override_past_the_partition_ceiling_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(minutes=5000, requeue=True, checkpoint_steps=10))
        assert excinfo.value.code is Hpc3ErrorCode.TIME_LIMIT_EXCEEDS_PARTITION

    def test_an_override_to_a_generic_gpu_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_run(_workspace(), _run(gpu="gpu"))
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
            {"suffix": "s0", "command": "python a.py"},
            {"suffix": "s0", "command": "python b.py"},
        ]
        with pytest.raises(JSONTypeError, match="must not repeat a suffix"):
            resolve_sweep(_workspace(), _sweep(members=members))

    def test_an_undeclared_project_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            resolve_sweep(_workspace(), _sweep(project="zodiac"))
        assert excinfo.value.code is Hpc3ErrorCode.WORKSPACE_PROJECT_UNKNOWN
