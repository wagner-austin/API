"""Which cross-card controls a measurement applies, named so a flag can pick one.

The naming, not the applying. :func:`~platform_ml.determinism.apply_determinism`
applies them and records what it applied; this says what there is to ask for,
and does it without importing torch for the reason :mod:`probe_shapes` does:
three callers only ever NAME an arm -- two command lines parsing a flag before
a device exists, and the reports, which read finished records on a laptop.

WHY IT IS NOT STILL INSIDE ``cli/probe_trace``. It was, until the isolated
GEMM probe needed the same four arms on 2026-08-29. A second spelling of the
table would let the two drift, and then "the trace ran arm X" and "the GEMM
probe ran arm X" could mean two different postures while reading identically
in both records -- which is the precise failure naming an arm exists to
prevent.
"""

from __future__ import annotations

from typing import Final

CONTROLS_FLAG = "--controls"

#: Which cross-card controls a measurement applies, by the name the flag takes.
#:
#: WHY FOUR ARMS AND NOT TWO. The two controls address disjoint halves of a
#: model -- split-K governs cuBLASLt matmuls, the math pin governs attention,
#: and neither moves the other's tensors. So "both" answers whether agreement
#: is achievable, and the two single-control arms answer which control bought
#: which tensor. A two-arm flag would force a later attribution question to be
#: a code change rather than a run.
#:
#: ``none`` is first and is the arm every measurement command was fixed at
#: before the controls existed: it applies nothing, so whatever the launcher
#: exported still governs and a command's workspace observation still reports
#: the real environment.
CONTROL_ARMS: Final[dict[str, tuple[bool, bool]]] = {
    "none": (False, False),
    "split-k": (True, False),
    "attention": (False, True),
    "both": (True, True),
}


def require_control_arm(raw: str) -> tuple[bool, bool]:
    """Resolve the ``--controls`` value to a posture.

    Args:
        raw: The flag's value.

    Returns:
        ``(remove_split_k, math_attention)``.

    Raises:
        ValueError: When the name is not one of :data:`CONTROL_ARMS`. Refused
            rather than defaulted: a measurement whose arm was guessed is one
            whose record names a condition it may not have run under, which is
            the defect a workspace observation already exists to prevent one
            control at a time.
    """
    arm = CONTROL_ARMS.get(raw)
    if arm is None:
        raise ValueError(
            f"{CONTROLS_FLAG} must be one of {', '.join(sorted(CONTROL_ARMS))}; got {raw!r}"
        )
    return arm


__all__ = [
    "CONTROLS_FLAG",
    "CONTROL_ARMS",
    "require_control_arm",
]
