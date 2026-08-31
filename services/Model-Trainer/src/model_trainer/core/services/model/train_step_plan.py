"""What a train-step probe measures, named so records can be read on a laptop.

The naming, not the arithmetic, split from :mod:`train_step_probe` for the
reason :mod:`probe_shapes` is split from :mod:`known_answer_probe`: a report
reading finished records should not import torch to format a label.

WHY A TRAIN-STEP PROBE EXISTS AT ALL. Every cross-card agreement measured in
this experiment so far -- the ladder, the forward traces, the isolated GEMMs,
the real-workload scorer -- is a FORWARD pass. Training is the workload the
reproducibility question was asked for, and a backward pass is not a forward
pass run twice: its GEMMs have different shapes (the reduction runs over the
batch dimension for a weight gradient and over the output dimension for an
input gradient), the embedding gradient is a scatter-add with its own
deterministic path, and none of it has ever been compared across cards here.
"Forward passes only" was a stated scope limit of the page this experiment
writes to; this probe is the instrument that closes it.

WHY THE RUNG SET EXCLUDES ``xl`` BY DEFAULT. A backward pass holds a gradient
beside every parameter, so the 1.5-billion-parameter rung needs about 12.4 GB
before an activation -- into a 16 GB V100 with a preemptible 55-minute wall,
plus digesting three billion floats. The contrast the experiment needs is one
rung the four-card forward trace agreed on and the rung it first broke on
(``large``), and that fits. The rungs are a required flag, so a card with the
memory can still be handed ``xl``.
"""

from __future__ import annotations

import hashlib
from typing import Final

from model_trainer.core.services.model.trace_plan import FIELD_SEPARATOR, LOSS_NAME

TRAIN_STEP_EXPERIMENT = "train-step-attribution"

#: The rungs a cluster job walks by default: the smallest rung, the largest
#: rung the four-card forward trace agreed on everywhere, and the rung it
#: first broke on. See the module docstring for why ``xl`` is absent.
TRAIN_STEP_RUNGS: Final[tuple[str, ...]] = ("tiny", "medium", "large")

RUNGS_FLAG = "--rungs"

#: The step size of the one SGD update the probe takes. The VALUE is
#: arbitrary -- the probe compares bits, not convergence -- but it is a
#: module constant rather than a flag because two records that stepped by
#: different amounts would carry different bytes while sharing a label, which
#: is the collision a label exists to prevent.
TRAIN_STEP_LR = 1e-3

#: Observation kind for a parameter's gradient after ``loss.backward()``.
GRAD_KIND = "grad"

#: Observation kind for a parameter's value after the SGD update.
UPDATED_KIND = "updated"


def require_train_rungs(raw: str) -> tuple[str, ...]:
    """Parse the ``--rungs`` flag into rung names.

    Args:
        raw: The flag's value, comma-separated, e.g. ``tiny,medium,large``.

    Returns:
        The names, in the order given.

    Raises:
        ValueError: When the value is empty, carries an empty item, or
            repeats a rung. A repeated rung would collide with itself in the
            record's observation names, and the refusal belongs here rather
            than at ``run_record``, further from the cause. Whether each name
            IS a rung is checked by ``require_probe_shape`` at the call site,
            which owns that table.
    """
    names = tuple(item.strip() for item in raw.split(","))
    if any(not name for name in names):
        raise ValueError(f"{RUNGS_FLAG} must be comma-separated rung names; got {raw!r}")
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"a train step cannot walk one rung twice: {duplicated}")
    return names


def train_step_label(rungs: tuple[str, ...], controls: str, kernel: str) -> str:
    """Name the record one run produces.

    The rung set is digested into the label for the reason ``trace_label``
    digests its rungs; the control arm and the kernel arm are in it for the
    reason ``gemm_label_for`` carries both: each changes what was computed,
    and ``agree_across_runs`` must not pair a treated run with an untreated
    one and report the experiment working as a card misbehaving.

    Args:
        rungs: The rung names, in the order they will be walked, already
            known distinct via :func:`require_train_rungs`.
        controls: The ``--controls`` value as the operator typed it.
        kernel: The ``--kernel`` value.

    Returns:
        e.g. ``train-step-3x1a2b3c4d5e6f-both-cublas``.
    """
    digest = hashlib.sha256(FIELD_SEPARATOR.join(rungs).encode("utf-8")).hexdigest()
    return f"train-step-{len(rungs)}x{digest[:12]}-{controls}-{kernel}"


def train_tensor_name(rung: str, kind: str, path: str, suffix: str) -> str:
    """Name one measurement of one parameter's tensor.

    Four fields where a forward-trace name has seven: a train step has no
    execution-order counter because parameters are walked in
    ``named_parameters`` order, which is the module tree and is identical on
    every card by construction -- the model is built from one seed before
    anything runs.

    Args:
        rung: The rung the step ran.
        kind: :data:`GRAD_KIND` or :data:`UPDATED_KIND`.
        path: The parameter's dotted path, e.g. ``transformer.wte.weight``.
        suffix: ``digest48`` or ``sum``, from :mod:`trace_plan`.

    Returns:
        e.g. ``large|grad|transformer.wte.weight|digest48``.
    """
    return FIELD_SEPARATOR.join((rung, kind, path, suffix))


def train_loss_name(rung: str) -> str:
    """Name a rung's reported loss.

    Args:
        rung: The rung.

    Returns:
        e.g. ``large|loss``.
    """
    return f"{rung}{FIELD_SEPARATOR}{LOSS_NAME}"


__all__ = [
    "GRAD_KIND",
    "RUNGS_FLAG",
    "TRAIN_STEP_EXPERIMENT",
    "TRAIN_STEP_LR",
    "TRAIN_STEP_RUNGS",
    "UPDATED_KIND",
    "require_train_rungs",
    "train_loss_name",
    "train_step_label",
    "train_tensor_name",
]
