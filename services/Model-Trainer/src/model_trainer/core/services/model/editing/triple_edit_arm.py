"""Write curated triples into a model's weights and ask the question set.

THE ARM BOARD TASK 3fc98ed6 EXISTS FOR. Three earlier arms tested a METHOD
against a fixed corpus and each came back negative; none tested the
REPRESENTATION. AKEW's 20-40x gap between clean triples and prose-extracted
ones says representation is where the bottleneck may be, so this arm holds the
method constant -- the same rank-one edit the weight-injection arm used -- and
changes only what it is fed.

WHAT MAKES THE NUMBER COMPARABLE. The items, the corpus split, the budget and
the scorer are the question-set arm's, unchanged, so this arm's accuracy sits
in the same table as base, cartridge, BM25, dense, fused and oracle. The
triples are grounded in the TRAINING half only, which is the same half the
cartridge trains on: an edit sourced from a held-out sentence would be handed
the answer to a question the other arms had to infer.

TWO FAILURE MODES, KEPT APART, because criterion 4 of that task says they are
different findings and the second is the interesting one:

* triples that could not be produced -- counted by the gate, reported per
  clause, and NOT charged to the method;
* triples produced, edited into the weights, and still not answering --
  charged to the method, and the only result that says anything about
  locate-then-edit on a clean representation.

An edit-success rate is reported beside the accuracy and is deliberately not
the verdict: AKEW's own point is that the two come apart, so a run that moves
target likelihood and not accuracy has to be able to say so.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.run_record import Observation
from typing_extensions import TypedDict

from model_trainer.core.contracts.cloze import ClozeItem
from model_trainer.core.contracts.knowledge_edit import (
    EditSite,
    RankOneEditRecord,
    render_edit_prompt,
    resolve_edit_module,
)
from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.cloze.score import score_cloze_items
from model_trainer.core.services.model.editing.activations import capture_module_io
from model_trainer.core.services.model.editing.apply import apply_rank_one_edit
from model_trainer.core.services.model.editing.fact_token import fact_token_position
from model_trainer.core.services.model.editing.grounding import (
    GATE_CLAUSES,
    GateReport,
    TripleCandidate,
    as_edit_request,
    gate_candidates,
)
from model_trainer.core.services.model.editing.rank_one import solve_right_vector
from model_trainer.core.services.model.editing.value_optimisation import (
    optimise_target_output,
    target_token_nll,
)
from model_trainer.core.types import TracedLMModelProto


class TargetLikelihood(TypedDict):
    """How surprised the model is by one triple's answer, before and after.

    Attributes:
        item_id: The triple this measures.
        before: Total NLL of ``prompt + target`` before any edit.
        after: The same after every edit was applied.
    """

    item_id: str
    before: float
    after: float


class TripleEditOutcome(TypedDict):
    """Everything one run of this arm produced.

    Attributes:
        gate: What the curation cost, per clause.
        edits: One record per applied edit.
        likelihoods: Per accepted triple, its answer's surprise before and
            after.
        base_accuracy: The question set's accuracy before any edit.
        accuracy_curve: Its accuracy after each edit, one entry per applied
            edit, in the order they were applied. The last entry is the fully
            edited model's accuracy; the shape of the rest is what says
            whether the cost is per association or per interference.
    """

    gate: GateReport
    edits: tuple[RankOneEditRecord, ...]
    likelihoods: tuple[TargetLikelihood, ...]
    base_accuracy: float
    accuracy_curve: tuple[float, ...]


def accepted_candidates(
    candidates: Sequence[TripleCandidate], report: GateReport
) -> tuple[TripleCandidate, ...]:
    """Select the candidates whose verdict accepted them.

    Args:
        candidates: The judged candidates, in the order they were judged.
        report: That judgement.

    Returns:
        Only the accepted ones, in the same order.
    """
    return tuple(
        candidate
        for candidate, verdict in zip(candidates, report["verdicts"], strict=True)
        if verdict["accepted"]
    )


def edit_one_triple(
    *,
    model: TracedLMModelProto,
    encoder: Encoder,
    candidate: TripleCandidate,
    site: EditSite,
    value_steps: int,
    value_learning_rate: float,
    device: str,
) -> RankOneEditRecord:
    """Write one accepted triple into the weights.

    The key vector is the module's own INPUT at the keyed token, which is what
    makes the solve's denominator that input's squared norm -- never small for
    a real activation, so the refusal in
    :func:`~model_trainer.core.services.model.editing.rank_one.solve_right_vector`
    guards a degenerate case rather than the normal one.

    Args:
        model: The model, edited in place.
        encoder: The tokenizer every position and score is measured under.
        candidate: An accepted triple.
        site: Where to write.
        value_steps: Optimisation steps for the value vector.
        value_learning_rate: Step size for it.
        device: Device to run on.

    Returns:
        What was written.

    Raises:
        AppError: Propagated from the position lookup, the capture, the value
            optimisation, the solve or the apply. None of them is caught here:
            an arm that skipped a failed edit and reported the accuracy of the
            rest would be reporting a different experiment.
    """
    request = as_edit_request(candidate)
    prompt = render_edit_prompt(request)
    position = fact_token_position(
        prompt=prompt, subject=request["subject"], strategy=site["fact_token"], encoder=encoder
    )
    prompt_ids = encoder.encode(prompt).ids
    target_ids = encoder.encode(request["target_new"]).ids
    module_name = resolve_edit_module(site)

    sequence: list[list[int]] = [prompt_ids]
    captured = capture_module_io(
        model=model,
        module_name=module_name,
        input_ids=torch.tensor(sequence, dtype=torch.long).to(device),
        position=position,
    )
    target_output = optimise_target_output(
        model=model,
        module_name=module_name,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        position=position,
        current_output=captured["module_output"],
        steps=value_steps,
        learning_rate=value_learning_rate,
        device=device,
    )
    solve = solve_right_vector(
        target_output=target_output,
        current_output=captured["module_output"],
        current_input=captured["module_input"],
        left=captured["module_input"],
    )
    return apply_rank_one_edit(
        model=model,
        site=site,
        item_id=request["item_id"],
        left=captured["module_input"],
        right=solve["vector"],
        denominator=solve["denominator"],
    )


def target_likelihoods(
    *,
    model: TracedLMModelProto,
    encoder: Encoder,
    candidates: Sequence[TripleCandidate],
    device: str,
) -> tuple[float, ...]:
    """Score how surprised the model is by each triple's own answer.

    The TARGET's tokens only -- see
    :func:`~model_trainer.core.services.model.editing.value_optimisation.target_token_nll`
    for the measurement this replaced and why the whole-sequence version
    reported every working edit as a large regression.

    Args:
        model: The model to ask.
        encoder: The tokenizer.
        candidates: Accepted triples.
        device: Device to run on.

    Returns:
        One total NLL per candidate, in order.

    Raises:
        AppError: Propagated from the prompt renderer.
    """
    scores: list[float] = []
    for candidate in candidates:
        request = as_edit_request(candidate)
        scores.append(
            target_token_nll(
                model=model,
                prompt_ids=encoder.encode(render_edit_prompt(request)).ids,
                target_ids=encoder.encode(request["target_new"]).ids,
                device=device,
            )
        )
    return tuple(scores)


def run_triple_edit_arm(
    *,
    model: TracedLMModelProto,
    encoder: Encoder,
    items: Sequence[ClozeItem],
    candidates: Sequence[TripleCandidate],
    training_text: str,
    site: EditSite,
    value_steps: int,
    value_learning_rate: float,
    device: str,
    max_seq_len: int,
) -> TripleEditOutcome:
    """Gate the triples, edit the accepted ones, and re-ask the question set.

    THE ORDER MATTERS AND IS NOT NEGOTIABLE. Every measurement of the
    unedited model is taken BEFORE the first edit, because the edits change
    the weights in place and there is no second copy of the base.

    Args:
        model: The base model, edited in place and left edited.
        encoder: The tokenizer every arm is measured under.
        items: The held-out question set.
        candidates: The curated triples, accepted and rejected alike.
        training_text: The half the triples must be grounded in.
        site: Where each edit is written.
        value_steps: Optimisation steps per edit.
        value_learning_rate: Step size for the value vector.
        device: Device to run on.
        max_seq_len: Token budget for scoring.

    Returns:
        The gate's report, the applied edits, the per-triple likelihoods and
        both accuracies.

    Raises:
        AppError: Propagated from the gate's consumers, the edits or the
            scorer.
    """
    report = gate_candidates(candidates, training_text=training_text)
    accepted = accepted_candidates(candidates, report)

    before_accuracy = score_cloze_items(
        items=items, model=model, encoder=encoder, device=device, max_seq_len=max_seq_len
    )["accuracy"]
    before = target_likelihoods(model=model, encoder=encoder, candidates=accepted, device=device)

    # THE ACCURACY IS RE-MEASURED AFTER EVERY EDIT, not only at the end, and
    # that curve is the arm's most informative output. A single before-and-
    # after cannot distinguish "each association costs the model something"
    # from "thirteen rank-one updates at one site interfere", and those are
    # different findings about a representation.
    edits: list[RankOneEditRecord] = []
    curve: list[float] = []
    for candidate in accepted:
        edits.append(
            edit_one_triple(
                model=model,
                encoder=encoder,
                candidate=candidate,
                site=site,
                value_steps=value_steps,
                value_learning_rate=value_learning_rate,
                device=device,
            )
        )
        curve.append(
            score_cloze_items(
                items=items, model=model, encoder=encoder, device=device, max_seq_len=max_seq_len
            )["accuracy"]
        )

    after = target_likelihoods(model=model, encoder=encoder, candidates=accepted, device=device)

    return TripleEditOutcome(
        gate=report,
        edits=tuple(edits),
        likelihoods=tuple(
            TargetLikelihood(item_id=candidate["item_id"], before=first, after=second)
            for candidate, first, second in zip(accepted, before, after, strict=True)
        ),
        base_accuracy=before_accuracy,
        accuracy_curve=tuple(curve),
    )


def triple_edit_observations(outcome: TripleEditOutcome) -> tuple[Observation, ...]:
    """Name every number this arm produced.

    THE COST LINE IS IN THE RECORD, not in a write-up. A representation that
    works and costs more to produce than writing the answers by hand is a
    negative result, and it can only be read as one if the reject rate travels
    with the accuracy.

    Args:
        outcome: What the arm produced.

    Returns:
        The named numbers: the curation's cost per clause, both accuracies and
        their difference, the applied edit count, and the mean target
        likelihood before and after.
    """
    report = outcome["gate"]
    curated = report["accepted"] + report["rejected"]
    curve = outcome["accuracy_curve"]
    # The fully edited model's accuracy is the curve's last point, and the
    # curve is empty exactly when nothing was accepted -- in which case the
    # edited model IS the base model, which is a measurement rather than a
    # missing one.
    edited = curve[-1] if curve else outcome["base_accuracy"]
    named: list[Observation] = [
        Observation(name="triples_curated", value=float(curated)),
        Observation(name="triples_accepted", value=float(report["accepted"])),
        Observation(name="triples_rejected", value=float(report["rejected"])),
        Observation(
            name="triples_reject_rate",
            value=float(report["rejected"]) / float(curated) if curated else 0.0,
        ),
        Observation(name="edits_applied", value=float(len(outcome["edits"]))),
        Observation(name="base_accuracy", value=outcome["base_accuracy"]),
        Observation(name="edited_accuracy", value=edited),
        Observation(name="edited_accuracy_gain", value=edited - outcome["base_accuracy"]),
    ]
    for clause in GATE_CLAUSES:
        count = dict(report["failures_by_clause"])[clause]
        named.append(Observation(name=f"gate_rejects_{clause}", value=float(count)))
    for applied, accuracy in enumerate(curve, start=1):
        named.append(Observation(name=f"accuracy_after_{applied}_edits", value=accuracy))
    likelihoods = outcome["likelihoods"]
    if likelihoods:
        improved = sum(1 for row in likelihoods if row["after"] < row["before"])
        named.append(
            Observation(
                name="target_nll_before",
                value=sum(row["before"] for row in likelihoods) / float(len(likelihoods)),
            )
        )
        named.append(
            Observation(
                name="target_nll_after",
                value=sum(row["after"] for row in likelihoods) / float(len(likelihoods)),
            )
        )
        # REPORTED, NOT THE VERDICT. AKEW's point is that edit success and
        # downstream answering come apart, so an arm that could not say "every
        # edit took and the accuracy still fell" would be unable to report the
        # only interesting outcome it has.
        named.append(
            Observation(name="edit_success_rate", value=float(improved) / float(len(likelihoods)))
        )
    return tuple(named)


__all__ = [
    "TargetLikelihood",
    "TripleEditOutcome",
    "accepted_candidates",
    "edit_one_triple",
    "run_triple_edit_arm",
    "target_likelihoods",
    "triple_edit_observations",
]
