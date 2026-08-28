"""Tests for the workspace contract.

The condition this file guards against is two commands reading different
answers to the same question -- which ledger, which host, which cap. That is
not a crash; it is a clean triage board while jobs run unwatched.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.workspace import (
    DEFAULT_QUIET_SECONDS,
    PROJECT_FIELDS,
    Workspace,
    decode_workspace,
    encode_project_config,
    encode_workspace,
    require_project_config,
)
from tests.against_hpc3 import decode_project_config
from tests.conftest import gpus, project_config, workspace_document

_DIR = pathlib.Path("/tmp/ws")


def _decode(**overrides: JSONValue) -> Workspace:
    """Decode a workspace built from the shared baseline.

    Args:
        **overrides: Top-level fields to replace.

    Returns:
        The decoded workspace.
    """
    return decode_workspace(workspace_document(**overrides), config_dir=_DIR)


class TestDecodeWorkspace:
    def test_it_reads_every_field(self) -> None:
        workspace = decode_workspace(workspace_document(), config_dir=_DIR)
        assert workspace["host"] == "hpc3"
        assert workspace["root"] == "/pub/w"
        assert workspace["quiet_seconds"] == 1800
        assert workspace["projects"]["abl"]["budget"] == {
            "self_imposed_gpu_hours": 100.0,
            "max_service_units": 0.0,
            "charge_account": "",
        }
        assert sorted(workspace["projects"]) == ["abl"]

    def test_a_relative_ledger_resolves_against_the_document(self) -> None:
        """So a workspace can be committed beside its runs and used anywhere."""
        workspace = decode_workspace(
            workspace_document(ledger="runs/ledger.jsonl"), config_dir=pathlib.Path("/home/a/w")
        )
        assert pathlib.Path(workspace["ledger"]) == pathlib.Path("/home/a/w/runs/ledger.jsonl")

    def test_an_absolute_ledger_is_left_alone(self) -> None:
        workspace = decode_workspace(
            workspace_document(ledger=str(pathlib.Path("/var/ledger.jsonl"))), config_dir=_DIR
        )
        assert pathlib.Path(workspace["ledger"]) == pathlib.Path("/var/ledger.jsonl")

    def test_quiet_seconds_defaults_when_omitted(self) -> None:
        document = workspace_document()
        del document["quiet_seconds"]
        workspace = decode_workspace(document, config_dir=_DIR)
        assert workspace["quiet_seconds"] == DEFAULT_QUIET_SECONDS

    def test_a_zero_quiet_threshold_is_refused(self) -> None:
        """It would report every running job as silent, which reads as noise."""
        with pytest.raises(JSONTypeError, match="at least 1"):
            _decode(quiet_seconds=0)

    def test_a_non_integer_quiet_threshold_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be an integer"):
            _decode(quiet_seconds="1800")

    def test_a_boolean_quiet_threshold_is_refused(self) -> None:
        """True is an int in Python; it is not a number of seconds."""
        with pytest.raises(JSONTypeError, match="must be an integer"):
            _decode(quiet_seconds=True)

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_workspace("hpc3", config_dir=_DIR)

    def test_an_empty_host_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            _decode(host="")

    def test_a_relative_root_is_refused(self) -> None:
        with pytest.raises(ValueError, match="absolute POSIX path"):
            _decode(root="pub/w")

    def test_an_empty_project_table_is_refused(self) -> None:
        """A workspace that declares no project can submit nothing."""
        with pytest.raises(JSONTypeError, match="at least one project"):
            _decode(projects={})

    def test_a_non_object_project_table_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            _decode(projects="abl")

    def test_a_missing_project_table_is_refused(self) -> None:
        document = workspace_document()
        del document["projects"]
        with pytest.raises(JSONTypeError):
            decode_workspace(document, config_dir=_DIR)

    def test_a_project_name_the_layout_rejects_is_refused(self) -> None:
        """A name unusable in squeue must not enter through the config either."""
        with pytest.raises(JSONTypeError):
            _decode(projects={"My Project": project_config()})

    def test_several_projects_are_all_kept(self) -> None:
        workspace = _decode(
            projects={
                "abl": project_config(),
                "sirius": project_config(cpus=4),
                "turkic-lstm": project_config(gpu=gpus("V100")),
            }
        )
        assert sorted(workspace["projects"]) == ["abl", "sirius", "turkic-lstm"]
        assert workspace["projects"]["sirius"]["cpus"] == 4


class TestDecodeProjectConfig:
    def test_it_reads_every_field(self) -> None:
        """PROJECT_FIELDS plus the budget, which is decoded and not overridable.

        The gap between the two is the assertion: PROJECT_FIELDS is exactly
        what a RUN may replace, and a cap a run can replace is not a cap.
        """
        config = decode_project_config(project_config())
        assert sorted(config.keys()) == sorted([*PROJECT_FIELDS, "budget"])
        assert "budget" not in PROJECT_FIELDS

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_project_config("free-gpu")

    def test_a_partition_this_cluster_lacks_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_project_config(project_config(partition="turbo"))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN

    def test_a_generic_gpu_is_refused_with_the_pinning_code(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_project_config(project_config(gpu=gpus("gpu")))
        assert excinfo.value.code is Hpc3ErrorCode.GPU_TYPE_UNPINNED

    def test_a_cpu_only_project_states_null_and_is_admitted(self) -> None:
        config = decode_project_config(project_config(partition="free", gpu=None))
        assert config["gpu"] is None
        assert config["partition"] == "free"

    def test_a_negative_checkpoint_interval_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be negative"):
            decode_project_config(project_config(checkpoint_steps=-1))

    def test_zero_checkpoint_steps_means_no_checkpointing(self) -> None:
        assert decode_project_config(project_config(checkpoint_steps=0))["checkpoint_steps"] == 0

    def test_a_non_positive_resource_is_refused(self) -> None:
        for field in ("cpus", "mem_gb", "minutes"):
            with pytest.raises(JSONTypeError, match="at least 1"):
                decode_project_config(project_config(**{field: 0}))

    def test_a_zero_gpu_count_is_refused_rather_than_meaning_cpu_only(self) -> None:
        with pytest.raises(JSONTypeError, match="at least 1"):
            decode_project_config(project_config(gpu=gpus("A100", 0)))

    def test_an_empty_env_path_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_project_config(project_config(env_path=""))

    def test_defaults_that_only_combine_badly_are_still_accepted(self) -> None:
        """A run may override any of these, so rejecting here refuses a
        legitimate shape: the cross-field rules belong at resolution.

        A billing partition declared as a default is admitted HERE and refused
        when a run resolves against it -- the project config is a set of
        values, not a submission.
        """
        config = decode_project_config(project_config(partition="standard", gpu=None))
        assert config["partition"] == "standard"


class TestRoundTrip:
    def test_a_project_config_round_trips(self) -> None:
        payload = project_config()
        assert encode_project_config(decode_project_config(payload)) == payload

    def test_a_workspace_round_trips_with_the_ledger_resolved(self) -> None:
        """Encoding emits the resolved ledger, so this is equivalence not identity."""
        workspace = decode_workspace(workspace_document(), config_dir=_DIR)
        encoded = encode_workspace(workspace)
        assert decode_workspace(encoded, config_dir=_DIR) == workspace

    def test_the_encoded_ledger_is_the_resolved_one(self) -> None:
        workspace = decode_workspace(workspace_document(), config_dir=_DIR)
        assert encode_workspace(workspace)["ledger"] == workspace["ledger"]


class TestRequireProjectConfig:
    def test_a_declared_project_is_returned(self) -> None:
        workspace = decode_workspace(workspace_document(), config_dir=_DIR)
        assert require_project_config(workspace, "abl")["gpu"] == {"model": "A100", "count": 1}

    def test_an_undeclared_project_is_refused(self) -> None:
        workspace = decode_workspace(workspace_document(), config_dir=_DIR)
        with pytest.raises(AppError) as excinfo:
            require_project_config(workspace, "sirius")
        assert excinfo.value.code is Hpc3ErrorCode.WORKSPACE_PROJECT_UNKNOWN

    def test_the_message_lists_what_is_declared(self) -> None:
        """The cause is nearly always a typo, and the list answers it."""
        workspace = decode_workspace(
            workspace_document(projects={"abl": project_config(), "zodiac": project_config()}),
            config_dir=_DIR,
        )
        with pytest.raises(AppError) as excinfo:
            require_project_config(workspace, "zodaic")
        assert "'abl', 'zodiac'" in excinfo.value.message
