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


DETERMINISM_ENV_VAR = "TRAIN_DETERMINISTIC"
"""Name of the variable a launcher sets to state a run's determinism posture.

Deliberately not named for any cluster. A trainer honours whichever launcher
started it, and a worker running locally in Docker reading a variable called
``HPC3_DETERMINISTIC`` would be plainly wrong -- and would be set wrong.
The submitter imports this name rather than spelling it, exactly as it does
for :data:`CUBLAS_WORKSPACE_ENV_VAR`.
"""

DETERMINISM_ON = "1"
"""Value requesting kernel-level determinism."""

DETERMINISM_OFF = "0"
"""Value explicitly declining it, in exchange for speed."""


def determinism_requested(raw: str | None) -> bool:
    """Interpret the determinism posture a launcher asked for.

    Takes the value rather than reading the environment, for two reasons: the
    monorepo routes every environment read through the sanctioned config
    accessors, and a pure function needs no hook to be testable.

    Args:
        raw: The variable's value, or None when it is unset. Callers obtain
            it with ``_optional_env_str(DETERMINISM_ENV_VAR)``.

    Returns:
        Whether determinism should be applied. An ABSENT variable means True:
        determinism is this platform's default, and the local worker that
        predates any launcher must keep behaving as it did. Absence is the
        only thing treated as a default -- a value that is present says what
        it says.

    Raises:
        ValueError: If the variable is present but is neither
            :data:`DETERMINISM_ON` nor :data:`DETERMINISM_OFF`. A typo must
            not resolve to either posture: guessing "on" wastes wall clock
            on a run the operator wanted fast, and guessing "off" produces a
            run recorded as deterministic that is not, which is the failure
            this whole variable exists to prevent.
    """
    if raw is None:
        return True
    if raw == DETERMINISM_ON:
        return True
    if raw == DETERMINISM_OFF:
        return False
    raise ValueError(
        f"{DETERMINISM_ENV_VAR}={raw!r} is neither {DETERMINISM_ON!r} nor "
        f"{DETERMINISM_OFF!r}. Refusing to guess: a run whose determinism "
        "posture is unknown cannot be compared with one whose is."
    )


__all__ = [
    "CUBLAS_DETERMINISTIC_WORKSPACE",
    "CUBLAS_WORKSPACE_ENV_VAR",
    "DETERMINISM_ENV_VAR",
    "DETERMINISM_OFF",
    "DETERMINISM_ON",
    "determinism_requested",
]
