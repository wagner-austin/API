"""Tests pinning the cuBLAS determinism variable.

These are two strings shared across dependency tiers: a trainer sets them
in-process before touching CUDA, and a job submitter writes them into a batch
script so they are present before the process starts. The submitter cannot
depend on torch, so it cannot reach the trainer's copy.

Pinning the values is the point. If they drifted, nothing would raise -- the
trainer would be deterministic, the submitted job would not, and the two would
quietly stop being comparable while both reported success.
"""

from __future__ import annotations

import pytest

from platform_core.determinism_env import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
    DETERMINISM_ENV_VAR,
    DETERMINISM_OFF,
    DETERMINISM_ON,
    determinism_requested,
)


def test_the_variable_is_the_one_cublas_reads() -> None:
    """PyTorch names this exact variable in the error it raises without it."""
    assert CUBLAS_WORKSPACE_ENV_VAR == "CUBLAS_WORKSPACE_CONFIG"


def test_the_workspace_value_is_the_documented_deterministic_setting() -> None:
    assert CUBLAS_DETERMINISTIC_WORKSPACE == ":4096:8"


def test_the_value_is_a_workspace_spec_not_a_flag() -> None:
    """A truthy-looking value such as "1" is accepted by the shell and
    rejected by cuBLAS; the shape is what makes it usable."""
    assert CUBLAS_DETERMINISTIC_WORKSPACE.startswith(":")
    assert CUBLAS_DETERMINISTIC_WORKSPACE.count(":") == 2


class TestDeterminismRequested:
    """The posture a launcher states, read by the process that can honour it.

    The launcher writes this variable into a batch script; the trainer reads
    it before any CUDA work. Both take the name from one definition, so a
    launcher cannot export something nothing reads.
    """

    def test_the_variable_is_not_named_for_any_cluster(self) -> None:
        """A local worker reading HPC3_DETERMINISTIC would be plainly wrong,
        and would eventually be set wrong."""
        assert DETERMINISM_ENV_VAR == "TRAIN_DETERMINISTIC"
        assert "HPC3" not in DETERMINISM_ENV_VAR

    def test_absent_means_on_because_that_is_the_platform_default(self) -> None:
        """The local worker predates any launcher and pinned determinism
        unconditionally; it must keep behaving that way."""
        assert determinism_requested(None) is True

    def test_an_explicit_on_is_on(self) -> None:
        assert determinism_requested(DETERMINISM_ON) is True

    def test_an_explicit_off_is_off(self) -> None:
        assert determinism_requested(DETERMINISM_OFF) is False

    def test_the_two_values_are_the_ones_a_shell_can_write(self) -> None:
        assert (DETERMINISM_ON, DETERMINISM_OFF) == ("1", "0")

    def test_an_unreadable_value_raises_rather_than_guessing(self) -> None:
        """Guessing "on" wastes wall clock on a run meant to be fast; guessing
        "off" produces a run recorded as deterministic that is not. Neither is
        a safe default, so there is no default."""
        with pytest.raises(ValueError, match="Refusing to guess"):
            determinism_requested("true")

    def test_common_truthy_spellings_are_all_refused(self) -> None:
        """ "yes" and "True" are exactly what someone reaches for, and each
        would otherwise resolve to whichever branch happened to be last."""
        for spelling in ("true", "True", "yes", "on", "", " 1"):
            with pytest.raises(ValueError):
                determinism_requested(spelling)

    def test_the_message_names_the_variable_and_both_valid_values(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            determinism_requested("maybe")
        assert DETERMINISM_ENV_VAR in str(excinfo.value)
        assert "'1'" in str(excinfo.value)
        assert "'0'" in str(excinfo.value)
