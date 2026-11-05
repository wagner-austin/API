"""Measure one written edit on a language model against the model untouched.

Board task 7964f9f6, criterion A2. One model is loaded once. It is measured
untouched, the edit is applied, it is measured again, and the edit is taken
back out with every parameter's bytes confirmed restored. Both measurements
are on the same card in the same process, so nothing but the edit separates
them.

TWO INSTRUMENTS, AND WHICH QUESTION EACH ANSWERS.

* **The target**: greedy repetition loops (:mod:`repetition`), the failure the
  edit is written to remove. Compared prompt by prompt.
* **The collateral**: the extraction-eval cloze set, scored by the same
  :func:`~model_trainer.core.services.model.cloze.score.score_cloze_items`
  every floor in this repository was measured with. An edit that fixes loops
  by damaging what the model knows has not fixed anything, and the per-item
  outcomes say which items moved, not only whether the total did.

BOTH ARE PAIRED, so both get an exact McNemar test on their discordant pairs
(:func:`platform_core.power_distributions.exact_mcnemar_p`): items right
before and wrong after against the reverse, prompts looping before and not
after against the reverse. A total that did not move can hide many items
moving both ways, and the discordant counts are reported for that reason.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.power_distributions import exact_mcnemar_p
from platform_core.run_record import Observation
from typing_extensions import TypedDict

from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItem
from model_trainer.core.contracts.weight_program import WeightProgram
from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.cloze.score import score_cloze_items
from model_trainer.core.services.model.weight_space.program import (
    ProgramEffect,
    apply_weight_program,
    remove_weight_program,
)
from model_trainer.core.services.model.weight_space.repetition import (
    RepetitionReport,
    RepetitionSettings,
    measure_repetition,
)
from model_trainer.core.types import TracedLMModelProto


class LMEditOutcome(TypedDict):
    """Both measurements of one edit.

    Attributes:
        effect: What the edit wrote.
        control_cloze: The untouched model's cloze result.
        edited_cloze: The edited model's.
        control_repetition: The untouched model's continuations.
        edited_repetition: The edited model's.
    """

    effect: ProgramEffect
    control_cloze: ClozeEvalResult
    edited_cloze: ClozeEvalResult
    control_repetition: RepetitionReport
    edited_repetition: RepetitionReport


class PairedCounts(TypedDict):
    """A paired comparison's two discordant cells and its exact p.

    Attributes:
        lost: Pairs true under control and false after the edit.
        gained: Pairs false under control and true after the edit.
        p_value: The exact two-sided McNemar p over ``lost + gained``.
    """

    lost: int
    gained: int
    p_value: float


def paired_counts(control: Sequence[bool], edited: Sequence[bool]) -> PairedCounts:
    """Count a paired comparison's discordant pairs and test them.

    Args:
        control: One outcome per pair under the control.
        edited: The same pairs' outcomes after the edit.

    Returns:
        The two discordant cells and the exact McNemar p.

    Raises:
        ValueError: If the two sequences differ in length, which would pair
            one item's outcome with another's.
    """
    if len(control) != len(edited):
        raise ValueError(f"cannot pair {len(control)} outcomes with {len(edited)}")
    lost = sum(1 for first, second in zip(control, edited, strict=True) if first and not second)
    gained = sum(1 for first, second in zip(control, edited, strict=True) if second and not first)
    return PairedCounts(
        lost=lost, gained=gained, p_value=exact_mcnemar_p(min(lost, gained), lost + gained)
    )


def run_lm_edit_arm(
    *,
    model: TracedLMModelProto,
    encoder: Encoder,
    items: Sequence[ClozeItem],
    prompts: Sequence[Sequence[int]],
    program: WeightProgram,
    settings: RepetitionSettings,
    device: str,
    max_seq_len: int,
) -> LMEditOutcome:
    """Measure the untouched model, the edited model, and restore it.

    THE ORDER IS FIXED: every control measurement is taken before the edit is
    applied, because the edit changes the weights in place and the only
    other copy is the snapshot inside the applied program.

    Args:
        model: The model, edited in place and restored before return.
        encoder: The tokenizer both instruments use.
        items: The cloze items.
        prompts: The repetition prompts.
        program: The edit.
        settings: How continuations are generated and judged.
        device: Where to run.
        max_seq_len: The cloze scorer's token budget.

    Returns:
        Both measurements and what the edit wrote.

    Raises:
        AppError: Propagated from the applier, the restore or the scorers.
    """
    control_cloze = score_cloze_items(
        items=items, model=model, encoder=encoder, device=device, max_seq_len=max_seq_len
    )
    control_repetition = measure_repetition(model, prompts, settings, device=device)
    applied = apply_weight_program(model, program)
    edited_cloze = score_cloze_items(
        items=items, model=model, encoder=encoder, device=device, max_seq_len=max_seq_len
    )
    edited_repetition = measure_repetition(model, prompts, settings, device=device)
    remove_weight_program(model, applied)
    return LMEditOutcome(
        effect=applied["effect"],
        control_cloze=control_cloze,
        edited_cloze=edited_cloze,
        control_repetition=control_repetition,
        edited_repetition=edited_repetition,
    )


def _mean(values: Sequence[float]) -> float:
    """Average a non-empty sequence.

    Args:
        values: The numbers.

    Returns:
        Their mean.
    """
    return sum(values) / float(len(values))


def lm_edit_observations(outcome: LMEditOutcome) -> tuple[Observation, ...]:
    """Name every number the arm produced.

    Args:
        outcome: What the arm measured.

    Returns:
        Both models' cloze accuracy and loop rate, the paired discordant
        counts and exact p for each instrument, mean seq-rep-n, and the
        size of the edit.
    """
    control_correct = [row["correct"] for row in outcome["control_cloze"]["outcomes"]]
    edited_correct = [row["correct"] for row in outcome["edited_cloze"]["outcomes"]]
    cloze = paired_counts(control_correct, edited_correct)
    # "Not looping" is the good outcome, so it is what counts as true here:
    # `gained` is then prompts the edit freed from a loop.
    loops = paired_counts(
        [not looped for looped in outcome["control_repetition"]["looped"]],
        [not looped for looped in outcome["edited_repetition"]["looped"]],
    )
    prompts = float(len(outcome["control_repetition"]["looped"]))
    effect = outcome["effect"]
    return (
        Observation(name="cloze_items", value=float(outcome["control_cloze"]["total"])),
        Observation(name="control_cloze_correct", value=float(outcome["control_cloze"]["correct"])),
        Observation(name="control_cloze_accuracy", value=outcome["control_cloze"]["accuracy"]),
        Observation(name="edited_cloze_correct", value=float(outcome["edited_cloze"]["correct"])),
        Observation(name="edited_cloze_accuracy", value=outcome["edited_cloze"]["accuracy"]),
        Observation(name="cloze_right_to_wrong", value=float(cloze["lost"])),
        Observation(name="cloze_wrong_to_right", value=float(cloze["gained"])),
        Observation(name="cloze_mcnemar_p", value=cloze["p_value"]),
        Observation(name="prompts", value=prompts),
        Observation(
            name="control_looped",
            value=float(sum(outcome["control_repetition"]["looped"])),
        ),
        Observation(
            name="edited_looped",
            value=float(sum(outcome["edited_repetition"]["looped"])),
        ),
        Observation(name="loops_removed", value=float(loops["gained"])),
        Observation(name="loops_introduced", value=float(loops["lost"])),
        Observation(name="loop_mcnemar_p", value=loops["p_value"]),
        Observation(name="control_seq_rep", value=_mean(outcome["control_repetition"]["seq_rep"])),
        Observation(name="edited_seq_rep", value=_mean(outcome["edited_repetition"]["seq_rep"])),
        Observation(name="elements_changed", value=float(effect["elements_changed"])),
        Observation(name="elements_total", value=float(effect["elements_total"])),
        Observation(
            name="fraction_changed",
            value=float(effect["elements_changed"]) / float(effect["elements_total"]),
        ),
    )


__all__ = [
    "LMEditOutcome",
    "PairedCounts",
    "lm_edit_observations",
    "paired_counts",
    "run_lm_edit_arm",
]
