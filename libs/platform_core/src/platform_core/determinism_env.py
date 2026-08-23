"""The environment variable that has to be set before CUDA starts.

This lives in ``platform_core`` rather than beside the code that applies
determinism, because two things in different dependency tiers need the exact
same string and they must never disagree:

* :mod:`platform_ml.determinism` sets it in-process, immediately before any
  CUDA work, for a trainer that is about to run.
* A job submitter writes it into a batch script, so it is already in the
  environment when the payload starts. A submitter runs on a laptop and must
  not depend on torch, which is what ``platform_ml`` pulls in.

A duplicated literal here would be the worst kind: the two copies would drift,
nothing would fail, and the runs would silently stop being comparable.

The timing constraint is the reason the value matters at all. ``cuBLAS`` reads
this variable once, when its handle is created, which happens on first use.
Setting it afterwards is accepted without error and has no effect -- so the
only safe places are before the process starts, or before the process touches
CUDA.
"""

from __future__ import annotations

CUBLAS_WORKSPACE_ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"
"""Name of the variable cuBLAS reads when it creates its handle."""

CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"
"""The workspace setting that makes cuBLAS reductions reproducible.

``torch.use_deterministic_algorithms(True)`` raises a ``RuntimeError`` naming
this variable when it is absent, so a run configured for determinism without
it fails loudly rather than producing quietly non-reproducible numbers. That
enforcement is the reason this pairing is safe to rely on.
"""


__all__ = ["CUBLAS_DETERMINISTIC_WORKSPACE", "CUBLAS_WORKSPACE_ENV_VAR"]
