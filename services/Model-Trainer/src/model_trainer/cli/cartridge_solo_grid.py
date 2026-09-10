"""Declared solo cells at 7B: the hyperparameter grid and the epochs line.

WHY THIS EXISTS (board tasks ``47d5f8c6`` and ``e03cd293``, both
operator-directed). The chain of eliminations at 7B: solo cartridge
gains collapse to ~0.10-0.16 nats against the family's ~0.83; headroom
explains the ceiling (~0.4-0.5 available); NF4 is exonerated (paired
difference bounded under 0.075 nats); the lr x slots grid REFUTED the
tuning hypothesis in its measured ranges -- the recorded knobs sit at
the grid's maximum. The one training axis left is EXPOSURE, so the
epochs line runs 12/24/48 epochs at the anchor knobs. Past that axis
the standing conclusion becomes that KV-prefix capacity does not
transfer to this architecture at this scale -- a finding about the
method, not the tuning.

CELLS ARE DECLARED, NOT CROSSED. The first arm crossed two axes; a
third crossed axis would run twenty-four cells to answer a three-cell
question, so the surface is now a tuple of explicit cells, each
carrying its own knobs AND its own observation token as data. The
recorded grid's eight tokens and label segment are preserved byte for
byte in the ``grid`` set, and the epochs line's 12-epoch cell reuses
the anchor token outright -- that cell IS the anchor configuration, so
its rows pair by name with the certified ``445e345f``/``bc18d701``
records, and it must reproduce them bit for bit or the line is refused
as unreadable at the verdict step.

ONE JOB WALKS ONE SET. The base loads once and every cell's cartridges
train against it; window and stride still come from the recorded plan
row through the same hook every sweep reads.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)
from typing_extensions import TypedDict

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli.cartridge_headroom import GEOMETRY_PLAN_NAME
from model_trainer.cli.cartridge_solo_seeds import SOLO_SEEDS, resolve_precision
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import (
    held_out_gain,
    train_cartridge,
)
from model_trainer.core.services.model.cartridge_plans import (
    base_short,
    corpus_digest,
    digest_parts,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_sweep_checkpoint import (
    bind_cells,
    checkpointed_cells,
)

_log = get_logger(__name__)

MODEL_FLAG = "--model-id"
CORPUS_FLAG = "--corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"
PRECISION_FLAG = "--precision"
CELLS_FLAG = "--cells"

_FLAGS = (MODEL_FLAG, CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG, PRECISION_FLAG, CELLS_FLAG)

#: Fixed for the reason every experiment name here is: the string is what
#: two records are grouped by, and a drifting one silently unpairs them.
SOLO_GRID_EXPERIMENT = "cartridge-solo-grid"

#: The in-grid anchor: exactly the recorded solo knobs. Pinned equal to
#: the plan row by test and by image smoke, and shared verbatim by both
#: declared cell sets below.
ANCHOR_LEARNING_RATE = 0.01
ANCHOR_SLOTS = 64
ANCHOR_EPOCHS = 12
ANCHOR_TOKEN = "lr0.01-c64"


class SoloGridCell(TypedDict):
    """One declared measurement cell, its knobs and its name as data.

    The token is declared rather than derived so a set can preserve the
    exact observation names an earlier record certified (the ``grid``
    set's eight tokens, the anchor token shared across sets) without a
    naming rule that branches on which knobs happen to differ.

    Attributes:
        token: The observation-name segment for this cell, unique within
            its set.
        learning_rate: AdamW step size for the slot masters.
        num_slots: Prefix positions per cartridge.
        epochs: Passes over the training windows.
    """

    token: str
    learning_rate: float
    num_slots: int
    epochs: int


class SoloCellSet(TypedDict):
    """One declared, runnable collection of cells.

    Attributes:
        name: The ``--cells`` selector value that names this set.
        label_segment: The label fragment identifying the set's shape --
            declared so the recorded grid label form stays byte-stable.
        cells: The cells, in run order.
    """

    name: str
    label_segment: str
    cells: tuple[SoloGridCell, ...]


#: The recorded lr x slots grid (task 47d5f8c6), tokens and label
#: segment byte-equal to the certified bc18d701 record's.
GRID_CELL_SET: SoloCellSet = {
    "name": "grid",
    "label_segment": "l4-c2",
    "cells": (
        {"token": "lr0.001-c64", "learning_rate": 0.001, "num_slots": 64, "epochs": 12},
        {"token": "lr0.001-c256", "learning_rate": 0.001, "num_slots": 256, "epochs": 12},
        {"token": "lr0.003-c64", "learning_rate": 0.003, "num_slots": 64, "epochs": 12},
        {"token": "lr0.003-c256", "learning_rate": 0.003, "num_slots": 256, "epochs": 12},
        {"token": ANCHOR_TOKEN, "learning_rate": 0.01, "num_slots": 64, "epochs": 12},
        {"token": "lr0.01-c256", "learning_rate": 0.01, "num_slots": 256, "epochs": 12},
        {"token": "lr0.03-c64", "learning_rate": 0.03, "num_slots": 64, "epochs": 12},
        {"token": "lr0.03-c256", "learning_rate": 0.03, "num_slots": 256, "epochs": 12},
    ),
}

#: The epochs line (task e03cd293): exposure alone, at the anchor knobs.
#: The 12-epoch cell reuses the anchor token verbatim because it IS the
#: anchor configuration -- its rows pair by name with the certified
#: records and must reproduce them bit for bit.
EPOCHS_LINE_CELL_SET: SoloCellSet = {
    "name": "epochs-line",
    "label_segment": "e12.24.48",
    "cells": (
        {"token": ANCHOR_TOKEN, "learning_rate": 0.01, "num_slots": 64, "epochs": 12},
        {"token": "lr0.01-c64-e24", "learning_rate": 0.01, "num_slots": 64, "epochs": 24},
        {"token": "lr0.01-c64-e48", "learning_rate": 0.01, "num_slots": 64, "epochs": 48},
    ),
}


def cell_set_for(selector: str) -> SoloCellSet:
    """Resolve the ``--cells`` selector to a declared set.

    An explicit map with loud refusal, like the loading policy: a
    guessed set would run a measurement no record could name.

    Args:
        selector: The set name from the command line.

    Returns:
        The declared cell set.

    Raises:
        ValueError: For a selector no set here names.
    """
    if selector == GRID_CELL_SET["name"]:
        return GRID_CELL_SET
    if selector == EPOCHS_LINE_CELL_SET["name"]:
        return EPOCHS_LINE_CELL_SET
    raise ValueError(
        f"unknown cell set {selector!r}; the declared sets are "
        f"{GRID_CELL_SET['name']!r} and {EPOCHS_LINE_CELL_SET['name']!r}, and a "
        f"guessed one would run a measurement no record could name"
    )


def solo_grid_label(
    *,
    model_id: str,
    seeds: Sequence[int],
    label_segment: str,
    precision_token: str,
    digest: str,
) -> str:
    """Build the label identifying one cell-set measurement.

    Args:
        model_id: The base measured.
        seeds: The seeds trained per cell.
        label_segment: The set's declared label fragment.
        precision_token: ``""`` for policy precision, or the stored-bf16
            token.
        digest: The corpus digest.

    Returns:
        The label.
    """
    return (
        f"cartridge-solo-grid-{base_short(model_id)}{precision_token}"
        f"-{label_segment}-n{len(seeds)}-{digest}"
    )


def measure_solo_grid(
    corpus: pathlib.Path,
    *,
    model_id: str,
    load_precision: QuantizationConfig | StoredBf16Precision | None,
    seeds: Sequence[int],
    cells: Sequence[SoloGridCell],
    device: str,
    checkpoints: pathlib.Path,
) -> tuple[tuple[Observation, ...], str]:
    """Train one solo cartridge per (cell, seed) and score each alone.

    The corpus is tokenized and split once and the base loads once;
    every cell trains against the same frozen weights and scores on the
    same held-out windows, so cells differ in exactly their declared
    knobs. Window and stride come from the recorded plan row through the
    same hook every sweep reads.

    RESUMABLE PER (CELL, SEED), which is finer than the grid cell it rolls
    up into. One cartridge training is the largest thing an eviction can
    destroy, and the per-seed gains are already recorded rows in their own
    right, so nothing is lost by checkpointing at that grain. The mean and
    spread are recomputed from all of a cell's seeds on every run --
    persisting them instead would let a resume report a spread over fewer
    draws than the record claims.

    Args:
        corpus: Directory of markdown documents; its held-out split is
            what every cartridge is scored on.
        model_id: The base to load.
        load_precision: What the loader is handed, from
            :func:`~model_trainer.cli.cartridge_solo_seeds.resolve_precision`.
        seeds: Seeds to train per cell.
        cells: The declared cells, run in order; at least one, tokens
            unique.
        device: Device to measure on.
        checkpoints: Directory holding this sweep's checkpoint.

    Returns:
        ``(observations, digest)``: per cell, one gain row per seed plus
        the cell's mean and spread; plus the corpus/window counts.

    Raises:
        ValueError: When the seed list or cell list is empty, or two
            cells share a token -- a colliding token would silently
            merge two cells' rows into one unreadable series.
        AppError: Propagated from the corpus, loading and training
            layers.
    """
    if len(seeds) == 0:
        raise ValueError(
            "no seeds named; a reliability measurement is denominators and zero draws is not one"
        )
    if len(cells) == 0:
        raise ValueError("no cells named; an empty set would still carry a label")
    tokens = [cell["token"] for cell in cells]
    if len(tokens) != len(set(tokens)):
        raise ValueError(
            f"duplicate cell token(s) in {tokens}; a colliding token would merge "
            f"two cells' rows into one unreadable series"
        )
    geometry = require_cartridge_plan(
        _measurement_hooks.base_lora_sweep_plans(), GEOMETRY_PLAN_NAME
    )
    documents = _test_hooks.read_corpus_documents(corpus)
    digest = corpus_digest(documents)
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(model_id)
    encoded = [tokenizer.encode(document) for document in documents]
    train, held_out = split_by_stride(
        build_windows(encoded, window=geometry["window"], device=device),
        held_out_stride=geometry["held_out_stride"],
    )
    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(model_id, load_precision))
    # The windows are on ``device``; the loader answers for where the
    # model materialised (job 55812484's lesson, pinned by the same
    # recording fake as the sibling CLIs).
    base.to(device)

    short = base_short(model_id)
    observations: list[Observation] = [
        Observation(
            name=f"grid-{short}_characters",
            value=float(sum(len(document) for document in documents)),
        ),
        Observation(name=f"grid-{short}_train_windows", value=float(len(train))),
        Observation(name=f"grid-{short}_held_out_windows", value=float(len(held_out))),
    ]

    def _train_and_score(unit: tuple[SoloGridCell, int]) -> tuple[Observation, ...]:
        """Train one cartridge under one cell's knobs at one seed.

        Args:
            unit: The grid cell whose knobs to train under, and the seed.

        Returns:
            That draw's gain, as one observation.
        """
        cell, seed = unit
        token = cell["token"]
        slots = train_cartridge(
            base,
            train,
            num_slots=cell["num_slots"],
            seed=seed,
            epochs=cell["epochs"],
            learning_rate=cell["learning_rate"],
        )
        gain = held_out_gain(CartridgeModel(base=base, slots=slots), held_out)
        _log.info("grid %s %s seed %d: gain %.4f", short, token, seed, gain)
        return (Observation(name=f"grid-{short}-{token}-seed{seed}_gain", value=gain),)

    # THE CELL SET IS IN THE MEASUREMENT NAME, not in the inputs digest. Two
    # cell-set selectors over one corpus are two measurements, not a conflict
    # between them: naming them apart lets both keep a checkpoint in the same
    # directory, where folding the set into the digest would make the second
    # run meet the first one's file and be refused as foreign.
    produced = checkpointed_cells(
        checkpoints,
        measurement=(f"solo-grid-{short}-{digest_parts([cell['token'] for cell in cells])[:12]}"),
        inputs_digest=digest_parts(
            [digest, str(geometry["window"]), str(geometry["held_out_stride"])]
        ),
        cells=bind_cells(
            [(f"{cell['token']}-seed{seed}", (cell, seed)) for cell in cells for seed in seeds],
            _train_and_score,
        ),
    )

    # WALKED IN THE DECLARED ORDER RATHER THAN OVER WHAT CAME BACK, so each
    # cell's mean and spread stay beside the seeds they reduce and the record
    # keeps the row order it has always had.
    for cell in cells:
        token = cell["token"]
        gains: list[float] = []
        for seed in seeds:
            cell_observations = produced[f"{token}-seed{seed}"]
            observations.extend(cell_observations)
            gains.append(cell_observations[0]["value"])
        observations.append(
            Observation(name=f"grid-{short}-{token}_gain_mean", value=sum(gains) / len(gains))
        )
        observations.append(
            Observation(
                name=f"grid-{short}-{token}_gain_spread",
                value=max(gains) - min(gains),
            )
        )
    return tuple(observations), digest


def solo_grid_run_record(
    corpus: pathlib.Path,
    *,
    model_id: str,
    precision: str,
    cells: str,
    device: str,
    checkpoints: pathlib.Path,
) -> RunRecord:
    """Pin determinism, run the selected cell set, and record it.

    Args:
        corpus: The corpus directory.
        model_id: The base to measure.
        precision: The precision selector, resolved through the solo
            CLI's :func:`resolve_precision`.
        cells: The cell-set selector, resolved through
            :func:`cell_set_for`.
        device: Device to measure on.
        checkpoints: Directory holding this sweep's checkpoint, so an evicted
            run resumes at its last completed (cell, seed) rather than at
            zero.

    Returns:
        The record.

    Raises:
        ValueError: Propagated from the two selectors and
            :func:`measure_solo_grid`.
        AppError: Propagated from the corpus, loading and training
            layers.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    load_precision, precision_token = resolve_precision(model_id, precision)
    cell_set = cell_set_for(cells)
    observations, digest = measure_solo_grid(
        corpus,
        model_id=model_id,
        load_precision=load_precision,
        seeds=SOLO_SEEDS,
        cells=cell_set["cells"],
        device=device,
        checkpoints=checkpoints,
    )
    return run_record(
        experiment=SOLO_GRID_EXPERIMENT,
        label=solo_grid_label(
            model_id=model_id,
            seeds=SOLO_SEEDS,
            label_segment=cell_set["label_segment"],
            precision_token=precision_token,
            digest=digest,
        ),
        fingerprint=fingerprint,
        observations=observations,
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected cell set and write the record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults
            to the process arguments.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value,
            or a required flag is absent.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    # RESOLVED BEFORE THE RUN, because the checkpoint directory derives from
    # where the artifact goes. A DERIVED path rather than a flag of its own:
    # a second flag could disagree with this one, and a resumed sweep would
    # then look for its work in a directory nothing wrote to.
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = solo_grid_run_record(
        pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        model_id=cli_args.require_flag(parsed, MODEL_FLAG),
        precision=cli_args.require_flag(parsed, PRECISION_FLAG),
        cells=cli_args.require_flag(parsed, CELLS_FLAG),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        checkpoints=out.parent / "checkpoints",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "solo grid %s %s -> %s",
        record["label"],
        describe_run_fingerprint(record["fingerprint"]),
        out,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="cartridge-solo-grid",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "ANCHOR_EPOCHS",
    "ANCHOR_LEARNING_RATE",
    "ANCHOR_SLOTS",
    "ANCHOR_TOKEN",
    "EPOCHS_LINE_CELL_SET",
    "GRID_CELL_SET",
    "SOLO_GRID_EXPERIMENT",
    "SoloCellSet",
    "SoloGridCell",
    "cell_set_for",
    "entrypoint",
    "main",
    "measure_solo_grid",
    "solo_grid_label",
    "solo_grid_run_record",
]


# Without this, `python -m model_trainer.cli.cartridge_solo_grid` imports
# the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
