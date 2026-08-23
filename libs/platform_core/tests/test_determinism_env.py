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

from platform_core.determinism_env import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
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
