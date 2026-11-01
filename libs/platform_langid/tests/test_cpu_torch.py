"""The venv holds the CPU build of torch (MCPs board task 3e7cfd51).

The detector runs on ``DEFAULT_DEVICE`` "cpu" and no consumer reserves a GPU,
so the pyproject takes torch and torchaudio from the PyTorch CPU index.
PyPI's Linux wheel is the CUDA build, with 14 nvidia packages no test or
deployment loads; a source left out of a dependency, or a lock resolved from
PyPI, would install it again with every test still passing. This reads the
installed versions, whose ``+cpu`` local label only the CPU index's wheels
carry.
"""

from __future__ import annotations

from importlib.metadata import version

import pytest


@pytest.mark.parametrize("distribution", ["torch", "torchaudio"])
def test_the_installed_build_is_the_cpu_one(distribution: str) -> None:
    assert version(distribution).endswith("+cpu"), version(distribution)
