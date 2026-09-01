"""Session guard: this suite REQUIRES a CUDA device, and says so loudly.

Refusal rather than skip, deliberately. A skip is an exemption that reads as
green -- the suite would pass on a GPU-less runner having tested nothing,
which is exactly the silent hole the 100% bar exists to close. The NavProbe
precedent: a GPU package's suite runs where a GPU is, and the Makefile
header tells the operator which machines those are.
"""

from __future__ import annotations

import os

import pytest
import torch
from platform_core.determinism_env import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Refuse to start without a CUDA device, and pin cuBLAS's workspace.

    The pin must happen HERE, not in the tests: ``CUBLAS_WORKSPACE_CONFIG``
    is read once when the cuBLAS handle is created, the kernel tests create
    that handle long before any CLI test pins determinism, and a late pin
    leaves ``use_deterministic_algorithms(True)`` refusing every cuBLAS call
    for the rest of the process -- measured as three train-CLI tests failing
    in the full suite while passing in isolation. ``os.putenv`` because it
    reaches the real process environment cuBLAS's getenv reads, and because
    a write is not the config read the env guard exists to stop.

    Args:
        session: Pytest's session object, unused.

    Raises:
        RuntimeError: When no CUDA device is available.
    """
    os.putenv(CUBLAS_WORKSPACE_ENV_VAR, CUBLAS_DETERMINISTIC_WORKSPACE)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ordered_kernels' suite REQUIRES a CUDA device: every kernel line "
            "is real GPU work, and skipping would report green on nothing. "
            "Run make check on a GPU machine (austinpc, sedona)."
        )
