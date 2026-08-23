"""Tests for preflight: the layer between "tests pass" and "it is running".

The parser fixtures are real ``sbatch --test-only`` lines measured from HPC3
on 2026-08-22, including the stray token between the timestamp and ``using``
that a naive split on whitespace gets wrong.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.job import JobSpec
from hpc3.contracts.preflight import encode_preflight_result
from hpc3.core.preflight import check_env_path, preflight
from tests.against_hpc3 import decode_job_spec, decode_preflight_result, parse_test_only
from tests.conftest import ABL_PINNED_DISTRIBUTIONS, FakeRun, cluster

_REAL_LINE = (
    "sbatch: Job 55516995 to start at 2026-08-22T03:23:00 a using 4 processors "
    "on nodes hpc3-gpu-16-02 in partition free-gpu"
)


def _spec(**overrides: JSONValue) -> JobSpec:
    """Build a decoded job spec.

    Args:
        **overrides: Fields to replace.

    Returns:
        A validated spec.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "arm-b-42",
        "partition": "free-gpu",
        "gpu": "A100",
        "gpu_count": 1,
        "cpus": 4,
        "mem_gb": 16,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "accept_billing": False,
        "env_path": "/pub/wagnera3/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B"},
        "command": "python train.py",
    }
    base.update(overrides)
    return decode_job_spec(base)


class TestParseTestOnly:
    def test_it_parses_the_measured_line(self) -> None:
        result = parse_test_only(_REAL_LINE + "\n")
        assert result == {
            "start_estimate": "2026-08-22T03:23:00",
            "processors": 4,
            "node_list": "hpc3-gpu-16-02",
            "partition": "free-gpu",
        }

    def test_the_stray_token_after_the_timestamp_is_dropped(self) -> None:
        """Slurm prints '... at <time> a using ...'; the 'a' is not the time."""
        assert parse_test_only(_REAL_LINE)["start_estimate"] == "2026-08-22T03:23:00"

    def test_it_finds_the_line_among_warnings(self) -> None:
        output = "sbatch: warning: quota low\n" + _REAL_LINE + "\nrc=0\n"
        assert parse_test_only(output)["processors"] == 4

    def test_a_multi_node_allocation_keeps_the_whole_node_list(self) -> None:
        line = _REAL_LINE.replace("hpc3-gpu-16-02", "hpc3-gpu-16-[00-05]")
        assert parse_test_only(line)["node_list"] == "hpc3-gpu-16-[00-05]"

    def test_output_with_no_estimate_is_unparsable(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_test_only("sbatch: warning: quota low\n")
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_UNPARSABLE

    def test_a_truncated_line_is_unparsable(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_test_only("sbatch: Job 1 to start at 2026-08-22T03:23:00 a using 4\n")
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_UNPARSABLE

    def test_a_missing_partition_anchor_is_unparsable(self) -> None:
        line = _REAL_LINE[: _REAL_LINE.find(" in partition ")]
        with pytest.raises(AppError) as excinfo:
            parse_test_only(line)
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_UNPARSABLE

    def test_a_non_numeric_processor_count_is_unparsable(self) -> None:
        line = _REAL_LINE.replace(" using 4 processors", " using many processors")
        with pytest.raises(AppError) as excinfo:
            parse_test_only(line)
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_UNPARSABLE

    def test_a_partition_this_cluster_lacks_is_refused_by_the_contract(self) -> None:
        """The scheduler echoing an unknown partition means the wrong machine."""
        line = _REAL_LINE.replace("in partition free-gpu", "in partition turbo")
        with pytest.raises(AppError) as excinfo:
            parse_test_only(line)
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN


class TestCheckEnvPath:
    def test_a_present_env_returns_the_verified_bin_dir(self, fake_run: FakeRun) -> None:
        fake_run.add("test -d", stdout="PRESENT\n")
        assert check_env_path("hpc3", _spec()) == "/pub/wagnera3/envs/abl-pinned/bin"

    def test_it_probes_the_bin_directory_not_the_root(self, fake_run: FakeRun) -> None:
        """An empty directory of the right name would otherwise pass."""
        fake_run.add("test -d", stdout="PRESENT\n")
        check_env_path("hpc3", _spec())
        assert "/pub/wagnera3/envs/abl-pinned/bin" in fake_run.calls[0].remote_command

    def test_an_absent_env_is_refused(self, fake_run: FakeRun) -> None:
        fake_run.add("test -d", stdout="ABSENT\n")
        with pytest.raises(AppError) as excinfo:
            check_env_path("hpc3", _spec())
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PATH_MISSING
        assert "/pub/wagnera3/envs/abl-pinned/bin" in excinfo.value.message


class TestPreflight:
    def test_a_clean_preflight_returns_the_verdict(self, fake_run: FakeRun) -> None:
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout=_REAL_LINE + "\nrc=0\n")

        result = preflight(_spec(), host="hpc3", script_dir="/j", log_dir="/l", cluster=cluster())

        assert result["partition"] == "free-gpu"
        assert result["processors"] == 4

    def test_it_tests_the_real_rendered_script_by_path(self, fake_run: FakeRun) -> None:
        """A reconstruction from flags could pass while the real script fails."""
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout=_REAL_LINE + "\nrc=0\n")
        preflight(_spec(), host="hpc3", script_dir="/j", log_dir="/l", cluster=cluster())

        commands = fake_run.commands()
        assert "cat > '/j/abl.arm-b-42.sbatch'" in commands
        assert any("sbatch --test-only abl.arm-b-42.sbatch" in c for c in commands)

    def test_nothing_is_queued(self, fake_run: FakeRun) -> None:
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout=_REAL_LINE + "\nrc=0\n")
        preflight(_spec(), host="hpc3", script_dir="/j", log_dir="/l", cluster=cluster())

        real_submits = [c for c in fake_run.commands() if "sbatch " in c and "--test-only" not in c]
        assert real_submits == []

    def test_a_missing_env_stops_before_the_scheduler_is_asked(self, fake_run: FakeRun) -> None:
        fake_run.add("test -d", stdout="ABSENT\n")
        with pytest.raises(AppError) as excinfo:
            preflight(_spec(), host="hpc3", script_dir="/j", log_dir="/l", cluster=cluster())
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PATH_MISSING
        assert not any("--test-only" in c for c in fake_run.commands())

    def test_a_rejection_carries_slurms_own_reason(self, fake_run: FakeRun) -> None:
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add(
            "--test-only",
            stdout="allocation failure: Invalid account or account/partition\nrc=1\n",
        )
        with pytest.raises(AppError) as excinfo:
            preflight(_spec(), host="hpc3", script_dir="/j", log_dir="/l", cluster=cluster())
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_REJECTED
        assert "Invalid account" in excinfo.value.message


class TestPreflightChecksEnvironmentIdentity:
    """Existence, then identity, then the scheduler -- in that order.

    The path check catches a typo. This catches the more expensive mistake:
    a real environment that is the wrong one.
    """

    def test_a_pinned_environment_that_matches_is_admitted(self, fake_run: FakeRun) -> None:
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("importlib.metadata", stdout=ABL_PINNED_DISTRIBUTIONS)
        fake_run.add("--test-only", stdout=_REAL_LINE + "\nrc=0\n")

        spec = _spec(pinned_packages={"torch": "2.6.0+cu124", "transformers": "4.46.3"})
        result = preflight(spec, host="hpc3", script_dir="/j", log_dir="/l", cluster=cluster())
        assert result["partition"] == "free-gpu"

    def test_the_wrong_environment_stops_before_the_scheduler_is_asked(
        self, fake_run: FakeRun
    ) -> None:
        """envs/abl instead of envs/abl-pinned: transformers 5.15.1, torch 2.11.0."""
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("importlib.metadata", stdout="torch==2.11.0+cu128\ntransformers==5.15.1\n")

        spec = _spec(pinned_packages={"transformers": "4.46.3"})
        with pytest.raises(AppError) as excinfo:
            preflight(spec, host="hpc3", script_dir="/j", log_dir="/l", cluster=cluster())

        assert excinfo.value.code is Hpc3ErrorCode.ENV_PACKAGE_MISMATCH
        assert not any("--test-only" in c for c in fake_run.commands())
        assert not any(c.startswith("cat >") for c in fake_run.commands())

    def test_an_unpinned_project_never_asks_the_environment(self, fake_run: FakeRun) -> None:
        """A compiled payload should not pay for a round trip it cannot use."""
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout=_REAL_LINE + "\nrc=0\n")

        preflight(_spec(), host="hpc3", script_dir="/j", log_dir="/l", cluster=cluster())
        assert not any("importlib.metadata" in c for c in fake_run.commands())


class TestPreflightResultContract:
    def test_a_valid_result_round_trips(self) -> None:
        payload: dict[str, JSONValue] = {
            "start_estimate": "2026-08-22T03:23:00",
            "processors": 4,
            "node_list": "hpc3-gpu-16-02",
            "partition": "free-gpu",
        }
        assert encode_preflight_result(decode_preflight_result(payload)) == payload

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_preflight_result("free-gpu")

    def test_zero_processors_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_preflight_result(
                {
                    "start_estimate": "t",
                    "processors": 0,
                    "node_list": "n",
                    "partition": "free-gpu",
                }
            )

    def test_an_empty_node_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_preflight_result(
                {
                    "start_estimate": "t",
                    "processors": 1,
                    "node_list": "",
                    "partition": "free-gpu",
                }
            )

    def test_an_empty_start_estimate_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_preflight_result(
                {
                    "start_estimate": "",
                    "processors": 1,
                    "node_list": "n",
                    "partition": "free-gpu",
                }
            )
