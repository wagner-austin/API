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

from typing import Protocol

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


BLAS_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
"""The thread-count variables a CPU numeric stack reads when it loads.

The CPU analogue of :data:`CUBLAS_WORKSPACE_ENV_VAR`. A multi-threaded BLAS
splits a reduction across threads and the thread COUNT decides the
partitioning, so two runs that split the same sum differently reach different
bits. Floating-point addition is not associative, and nothing about seeding
touches this.

Measured rather than assumed (2026-08-25, numpy 2.3.5 / scipy-openblas
0.3.30, 24 cores): at a fixed thread count the result is bit-identical run
after run, and across 1, 8 and 24 threads it is not -- three different
answers from identical bytes. The hazard is therefore an unrecorded INPUT,
not an unpredictable library.

All four are named because a numpy wheel may be linked against OpenBLAS or
MKL, and numexpr reads its own. Setting only the one that happens to matter
today leaves the record claiming a posture the next wheel will not honour.

The timing constraint is the same as cuBLAS's and just as unforgiving: these
are read when the native library is loaded, which happens on first import of
numpy or its dependents. Setting them afterwards is accepted in silence and
does nothing.
"""

SINGLE_THREAD = "1"
"""One thread, which is what makes a BLAS reduction order deterministic.

Deterministic rather than fast, deliberately. A parallel reduction can be
made reproducible only by fixing the split as well as the count, which no
portable interface exposes, so the reproducible choice is the serial one.
The cost is real and belongs to whoever chooses it -- which is why this is a
constant to pass rather than a default anything applies.
"""


class SetEnvProtocol(Protocol):
    """A writer for one process environment variable.

    A write-only seam rather than a mapping, for two reasons. Production
    passes ``os.putenv``, which reaches the real process environment that a
    C library's ``getenv`` reads -- the only environment cuBLAS or OpenBLAS
    consults. And the monorepo bans reading config out of ``os.environ``,
    correctly: configuration comes from the config layer. Writing a variable
    that a native library requires is a different act, and this Protocol
    keeps the two from being confused.

    Deliberately no read side. ``os.putenv`` does not update ``os.environ``,
    so a "did it get set?" helper built on the Python mapping would report
    False on a correctly configured process.

    Parameters are positional-only: ``os.putenv`` names them ``name`` and
    ``value``, and a Protocol that named them otherwise would reject the one
    implementation that matters.

    Here rather than beside either pinner because both need it: the torch one
    writes the cuBLAS workspace, the CPU one writes thread counts, and a
    second copy of this Protocol would be two spellings of one seam.
    """

    def __call__(self, key: str, value: str, /) -> None: ...


__all__ = [
    "BLAS_THREAD_ENV_VARS",
    "CUBLAS_DETERMINISTIC_WORKSPACE",
    "CUBLAS_WORKSPACE_ENV_VAR",
    "DETERMINISM_ENV_VAR",
    "DETERMINISM_OFF",
    "DETERMINISM_ON",
    "SINGLE_THREAD",
    "SetEnvProtocol",
    "determinism_requested",
]
