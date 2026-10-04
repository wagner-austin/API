"""The full-wiki family: one base plan, and every axis one field away from it.

WHY THESE ARE BUILT RATHER THAN WRITTEN OUT. Until 2026-10-04 every plan in
:mod:`cartridge_qa_plans` was a nineteen-field literal copied from its
neighbour, and the table's own tests had to check by spreading the base that
a rung moved only the field it named -- because nothing about the literals
made that true. Here it is true by construction: each cell is the base with
one field replaced, so a cell CANNOT drift in a field its axis does not own,
and a field added to :class:`~model_trainer.core.contracts.qa_plan.QaPlan`
later reaches every cell through the base instead of through nineteen edits.

THE NAMES ARE WRITTEN OUT EVEN THOUGH THE PLANS ARE NOT, and the reason is
the other end of the lookup. A run document names a plan, and both a person
and ``tools/hpc3``'s committed-run check find that plan by reading this
source for the name as a quoted literal. A name assembled by an f-string is
in no file anywhere, so the first version of this module -- which derived
every name from its base model and value -- made every cell unfindable by
grep and failed that check against image v55, whose run documents had to be
cancelled. Each cell is therefore declared as a ``(name, value)`` pair: the
name is stated once, beside the one value it stands for, and nothing derives
it.

WHY THE FULL WIKI AND NOT THE API-CODEBASE WIKI. The previous ladder top
(``pythia-6.9b-api-wiki-qa``) and the previous capacity axis
(``gpt2-large-api-wiki-qa-slots-*``) were declared against the api-codebase
wiki, which yielded 237 items on 2026-09-09 and 275 on 2026-10-04 -- under a
240-item cap both times, against the 250 the declared 0.02 needs. So every
one of them was refused by
:func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_question_set`
before it loaded a model: the scale rung and the capacity curve existed as
plans and could not produce a number. ``~/PROJECTS/wiki`` yields 3,278 to
3,827 items across every cell below (measured 2026-10-04, 885 pages,
1,634,715 gpt2 tokens), fifteen times the gate, so the family lives there.

THE FOUR AXES, and which field each one owns:

  scale     ``model_id`` (and ``precision_selector``, which the base forces)
  capacity  ``num_slots``, on the rung where the crossing was reported
  window    ``window``
  epochs    ``epochs``

The base plan is each axis's own point at its base value and is not repeated
as a cell: a second name for the same measurement would let one result be
filed under two labels.

THE WINDOW AXIS MOVES THE QUESTION SET, AND SO IT READS DIFFERENTLY. Items
are built from the held-out windows, so a cell with another ``window`` asks
other questions (3,753 items at 128, 3,827 at 256, 3,278 at 512). Raw
accuracy across those cells is not a curve of anything. What the axis
compares is each cell's WITHIN-cell gap -- cartridge against BM25, against
long context -- which is paired on that cell's own items and gated by that
cell's own item count. The slot and epoch axes keep the base's items, so
their accuracies are directly comparable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from model_trainer.core.contracts.qa_plan import QaPlan

#: The base every cell is one field away from, by name in the plan table.
FULL_WIKI_BASE: Final[str] = "gpt2-full-wiki-qa"

#: The rungs above the base: name, base model, and the precision it loads at.
#:
#: THE LADDER RUNS TO pythia-6.9b BECAUSE THE CROSSING IT REPORTS IS AT ~774M:
#: a ladder that stopped at gpt2-xl stopped one rung after its own finding.
#: pythia-6.9b is the base the cartridge sweeps already run on an A30
#: (``gpu_pinned_because: bf16-7b-weights-13.8GB-plus-training-need-24GB``),
#: so its profile is proven rather than assumed, and it loads at the declared
#: ``stored-bf16`` because fp32 is 27.6GB against that card's 24GB.
SCALE_RUNGS: Final[tuple[tuple[str, str, str], ...]] = (
    ("gpt2-medium-full-wiki-qa", "gpt2-medium", "policy"),
    ("gpt2-large-full-wiki-qa", "gpt2-large", "policy"),
    ("gpt2-xl-full-wiki-qa", "gpt2-xl", "policy"),
    ("pythia-6.9b-full-wiki-qa", "EleutherAI/pythia-6.9b", "stored-bf16"),
)

#: The rung the capacity axis sits on. gpt2-large, because that is where the
#: cartridge was reported to cross retrieval, and the claim the axis tests --
#: "a fixed slot budget loses to an unbounded index" -- is a claim about the
#: crossing.
SLOT_AXIS_RUNG: Final[str] = "gpt2-large-full-wiki-qa"

#: The capacity cells: a doubling ladder through the 128 every older plan froze.
SLOT_CELLS: Final[tuple[tuple[str, int], ...]] = (
    ("gpt2-large-full-wiki-qa-slots-32", 32),
    ("gpt2-large-full-wiki-qa-slots-64", 64),
    ("gpt2-large-full-wiki-qa-slots-128", 128),
    ("gpt2-large-full-wiki-qa-slots-256", 256),
)

#: Every capacity cell is scored under the TIGHTEST common budget, 1024 less
#: the largest prefix. Sizing each cell to ``1024 - num_slots`` would hand the
#: 32-slot cell the most evidence and the 256-slot cell the least, so the
#: smallest cartridge would also face the strongest retriever and the axis
#: would move two things in opposite directions at once.
SLOT_AXIS_MAX_SEQ_LEN: Final[int] = 768

#: Training window sizes beside the base's 256.
WINDOW_CELLS: Final[tuple[tuple[str, int], ...]] = (
    ("gpt2-full-wiki-qa-window-128", 128),
    ("gpt2-full-wiki-qa-window-512", 512),
)

#: Passes over the training windows beside the base's 12. On the LOSS axis 12
#: sits at the maximum and 24 halves the gain; whether that transfers to
#: ANSWERABILITY is the open question, so the curve runs both sides of 12.
EPOCH_CELLS: Final[tuple[tuple[str, int], ...]] = (
    ("gpt2-full-wiki-qa-epochs-3", 3),
    ("gpt2-full-wiki-qa-epochs-6", 6),
    ("gpt2-full-wiki-qa-epochs-24", 24),
)


def scale_rungs(base: QaPlan) -> dict[str, QaPlan]:
    """Build the ladder: the base with its model and that model's precision.

    Args:
        base: The full-wiki base plan.

    Returns:
        One plan per :data:`SCALE_RUNGS` entry, keyed by its declared name.
    """
    return {
        name: {**base, "model_id": model_id, "precision_selector": selector}
        for name, model_id, selector in SCALE_RUNGS
    }


def slot_cells(rung: QaPlan) -> dict[str, QaPlan]:
    """Build the capacity axis on one rung, every cell under one budget.

    Args:
        rung: The plan named :data:`SLOT_AXIS_RUNG`, from :func:`scale_rungs`.

    Returns:
        One plan per :data:`SLOT_CELLS` entry, keyed by its declared name.
    """
    return {
        name: {**rung, "num_slots": count, "max_seq_len": SLOT_AXIS_MAX_SEQ_LEN}
        for name, count in SLOT_CELLS
    }


def window_cells(base: QaPlan) -> dict[str, QaPlan]:
    """Build the window axis on the base.

    Args:
        base: The full-wiki base plan.

    Returns:
        One plan per :data:`WINDOW_CELLS` entry, keyed by its declared name.
    """
    return {name: {**base, "window": window} for name, window in WINDOW_CELLS}


def epoch_cells(base: QaPlan) -> dict[str, QaPlan]:
    """Build the epochs axis on the base.

    Args:
        base: The full-wiki base plan.

    Returns:
        One plan per :data:`EPOCH_CELLS` entry, keyed by its declared name.
    """
    return {name: {**base, "epochs": epochs} for name, epochs in EPOCH_CELLS}


def merged_plan_tables(*tables: Mapping[str, QaPlan]) -> dict[str, QaPlan]:
    """Join plan tables, refusing any name two of them both define.

    A plain dict merge would keep whichever table came last, so a built cell
    named like an authored plan would replace it silently -- and the record
    would carry the authored plan's name over the built plan's numbers.

    Args:
        *tables: The tables, in order.

    Returns:
        Every plan, keyed by its one name.

    Raises:
        ValueError: Naming the first plan defined twice.
    """
    merged: dict[str, QaPlan] = {}
    for table in tables:
        for name, plan in table.items():
            if name in merged:
                raise ValueError(
                    f"plan {name!r} is defined twice; one name must identify one "
                    f"measurement, or its record cannot say which one it is"
                )
            merged[name] = plan
    return merged


def full_wiki_family(base: QaPlan) -> dict[str, QaPlan]:
    """Build every full-wiki cell: the ladder and the three axes.

    Args:
        base: The full-wiki base plan.

    Returns:
        Every cell, keyed by name. The base itself is not among them; it is
        each axis's point at its base value.

    Raises:
        ValueError: From :func:`merged_plan_tables` if two axes declare a
            cell alike.
    """
    rungs = scale_rungs(base)
    return merged_plan_tables(
        rungs,
        slot_cells(rungs[SLOT_AXIS_RUNG]),
        window_cells(base),
        epoch_cells(base),
    )


__all__ = [
    "EPOCH_CELLS",
    "FULL_WIKI_BASE",
    "SCALE_RUNGS",
    "SLOT_AXIS_MAX_SEQ_LEN",
    "SLOT_AXIS_RUNG",
    "SLOT_CELLS",
    "WINDOW_CELLS",
    "epoch_cells",
    "full_wiki_family",
    "merged_plan_tables",
    "scale_rungs",
    "slot_cells",
    "window_cells",
]
