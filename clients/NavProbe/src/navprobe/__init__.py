"""Reproducibility instrument for simulated navigation.

Answers one question about a simulator before any policy result computed on it
can be trusted: given a fixed seed and a fixed action sequence, does the same
rollout produce the same bytes? The published determinism measurements for
GPU-batched simulators were taken with rendering disabled, so the rendered
observation stream those measurements are used to justify has never been
checked.

The package is layered so the answer does not depend on owning a GPU. Each layer
depends only on the ones above it in this list:

* :mod:`navprobe.canonical` turns floats and text into canonical bytes.
* :mod:`navprobe.digest` folds canonical bytes into stable digests.
* :mod:`navprobe.records` declares the typed records.
* :mod:`navprobe.wireformat` carries the codec primitives every record shares.
* :mod:`navprobe.codecs` holds one encode/decode module per record type.
* :mod:`navprobe.rollout` drives an injected simulator to a run record.
* :mod:`navprobe.comparison` folds two run records into a verdict.
* :mod:`navprobe.storage` moves records across a process boundary.
* :mod:`navprobe.experiment` composes the above into a determinism trial.
* :mod:`navprobe.crossprocess` compares trials that did not share a process.

Only the outermost adapter touches a real simulator. Everything above it is
pure and exercised against real in-repo implementations rather than mocks.

The shared failure base lives here rather than in a dedicated error module:
the monorepo forbids per-project ``errors.py`` files so that projects cannot
grow a parallel error framework alongside ``platform_core``. Concrete
exceptions are defined by the module that raises them, next to the code whose
contract they describe.
"""

from __future__ import annotations


class NavProbeError(Exception):
    """Base for every failure this package raises.

    Carries a stable machine-readable code alongside a human-readable message
    so callers can branch on the code and operators can read the message. There
    is deliberately no catch-all code: a new failure mode gets a new one so it
    can be grepped, asserted on, and counted.

    Code format is ``NP-<AREA>-<NNN>``.

    Args:
        code: Stable identifier, e.g. ``"NP-CANON-001"``.
        message: Human-readable description of what went wrong.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


__all__ = ["NavProbeError"]
