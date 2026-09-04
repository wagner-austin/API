"""Three arms on one question set: no corpus, corpus as a prefix, corpus in context.

THE QUESTION THE LOSS ARMS DO NOT ANSWER. Every cartridge number measured
before this module was a held-out loss, and
:mod:`model_trainer.core.contracts.cloze` already says why that is not enough:
"a model can memorise text word-by-word and still fail every question about
it." A prefix that lowers perplexity on wiki prose has shown it learned the
corpus's TEXT. This asks whether the model can USE the corpus, on sentences the
cartridge never read.

THE THREE ARMS, scored on identical items so every comparison is paired:

    base        the model alone. Chance is 1/(distractors+1); a base model
                that has never seen this corpus should sit near it.
    cartridge   the same model with the trained prefix in front. The corpus
                compressed into a fixed number of slots.
    retrieval   the same model with the evidence in its context window. The
                corpus as raw tokens, paying per token, every time.

THE RETRIEVAL ARM IS DELIBERATELY THE STRONGEST ONE AVAILABLE. It is handed
the training sentences that actually contain the answer term -- oracle
retrieval, with no retriever to blame and no ranking to lose. That makes it an
UPPER BOUND on what any real retrieval pipeline could achieve, which is the
only version worth comparing a cartridge against: beating a weak retriever
would say nothing.

AND IT IS BOUNDED BY THE CONTEXT WINDOW, WHICH IS THE WHOLE TRADE. Evidence is
truncated to whatever is left after the item itself, because
:func:`sequence_nll` truncates the TAIL of a rendering -- so an unbudgeted
prompt would cut off the answer and score a question the model was never
shown. A cartridge has no such bound: its cost is paid once at training time
and occupies a fixed number of slots regardless of how much corpus it read.
:func:`evidence_budget_tokens` reports what was actually available, so a
retrieval arm that lost because it could not fit its evidence says so rather
than looking like a model that did not know.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)

from model_trainer.core.contracts.cloze import (
    BLANK_MARKER,
    ClozeEvalResult,
    ClozeItem,
    ClozeItemOutcome,
)
from model_trainer.core.contracts.paired_comparison import (
    PairedComparison,
    PairedItemOutcome,
    summarise_pairs,
)
from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.corpus_cloze import sentences
from model_trainer.core.types import LogitsOutProto, ScoreableLMProto

#: Tokens held back from the evidence budget.
#:
#: The budget is computed from the longest candidate rendering, but a
#: tokenizer is not additive -- joining evidence to a template can merge or
#: split tokens at the seam, so the concatenation may be a token or two longer
#: than its parts. Reserving a few is cheaper than truncating an answer.
EVIDENCE_MARGIN_TOKENS = 8

#: Separator between the evidence and the question.
#:
#: A blank line and a marker, because the alternative -- running evidence
#: straight into the item -- produces a sentence the model reads as one
#: continuous passage, and the item's own text then looks like a continuation
#: of the evidence rather than a question about it.
EVIDENCE_JOINER = "\n\n"


def longest_rendering_tokens(item: ClozeItem, encoder: Encoder) -> int:
    """Tokens the item's longest candidate rendering occupies.

    The LONGEST rather than the mean: every candidate must fit, and a budget
    computed from a shorter one would truncate exactly the renderings whose
    answers are longest.

    Args:
        item: The item to measure.
        encoder: Tokenizer the scorer will use.

    Returns:
        Token count of the longest rendering.
    """
    candidates = [item["answer"], *item["distractors"]]
    return max(
        len(encoder.encode(item["template"].replace(BLANK_MARKER, candidate)).ids)
        for candidate in candidates
    )


def evidence_budget_tokens(item: ClozeItem, encoder: Encoder, *, max_seq_len: int) -> int:
    """How many tokens of evidence this item can carry and still be scored.

    Args:
        item: The item the evidence will be prepended to.
        encoder: Tokenizer the scorer will use.
        max_seq_len: The scorer's token budget.

    Returns:
        Tokens available for evidence. Zero or negative means the item alone
        already fills the window, and no retrieval is possible for it -- which
        is a real answer about the model's window, not a failure to compute.
    """
    return max_seq_len - longest_rendering_tokens(item, encoder) - EVIDENCE_MARGIN_TOKENS


def evidence_for(term: str, documents: Sequence[str]) -> str:
    """Collect the training sentences that mention a term.

    Oracle retrieval: the evidence is selected by knowing the answer, which no
    real pipeline can do. That is deliberate -- see this module's docstring.

    Args:
        term: The answer term to find evidence for.
        documents: Training document bodies to search.

    Returns:
        The matching sentences joined by spaces, in document order. Empty when
        the term appears in no sentence, which for an item built by
        :func:`~corpus_cloze.build_items` cannot happen: a term qualifies only
        if it occurs in the training text.
    """
    found = [
        sentence for document in documents for sentence in sentences(document) if term in sentence
    ]
    return " ".join(found)


def with_evidence(
    item: ClozeItem,
    evidence: str,
    encoder: Encoder,
    *,
    max_seq_len: int,
) -> ClozeItem:
    """Prepend as much evidence as the item can carry.

    The evidence is truncated from its END rather than its start, so the
    sentences nearest the question survive; the encoder's own ids are cut and
    decoded back, because truncating characters would risk splitting a token
    and changing what the earlier text says.

    Args:
        item: The item to augment.
        evidence: Retrieved text.
        encoder: Tokenizer the scorer will use.
        max_seq_len: The scorer's token budget.

    Returns:
        A new item whose template carries the evidence.

    Raises:
        AppError: With ``CLOZE_ITEM_UNSCOREABLE`` when the item leaves no room
            for evidence, or when no evidence was found for it.

            THIS REFUSES RATHER THAN RETURNING THE ITEM UNCHANGED, which is
            what it did first. An unchanged item is a BASE-arm question
            wearing the retrieval arm's name: the arm would report an accuracy
            averaged over some questions that had the evidence and some that
            never received it, and nothing downstream could tell which. A
            window too small for the plan's items is a misconfiguration, and
            the measurement should stop rather than quietly weaken.
    """
    budget = evidence_budget_tokens(item, encoder, max_seq_len=max_seq_len)
    if budget <= 0:
        raise AppError(
            ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE,
            (
                f"item {item['item_id']!r} leaves {budget} token(s) for evidence within "
                f"max_seq_len={max_seq_len}, so the retrieval arm cannot be given any; "
                f"shorten the items or raise the window rather than scoring this arm on "
                f"a question it never received evidence for"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE),
        )
    if evidence == "":
        raise AppError(
            ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE,
            (
                f"no evidence was found for item {item['item_id']!r}, whose answer is "
                f"{item['answer']!r}; items are built only from terms that occur in the "
                f"training text, so an empty result means the evidence and the items "
                f"were drawn from different corpora"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE),
        )
    ids = encoder.encode(evidence).ids
    kept = encoder.decode(ids[:budget]) if len(ids) > budget else evidence
    return ClozeItem(
        item_id=item["item_id"],
        template=f"{kept}{EVIDENCE_JOINER}{item['template']}",
        answer=item["answer"],
        distractors=list(item["distractors"]),
    )


def retrieval_items(
    items: Sequence[ClozeItem],
    documents: Sequence[str],
    encoder: Encoder,
    *,
    max_seq_len: int,
) -> list[ClozeItem]:
    """Build the retrieval arm's item set.

    Args:
        items: The shared question set.
        documents: Training documents the evidence is drawn from.
        encoder: Tokenizer the scorer will use.
        max_seq_len: The scorer's token budget.

    Returns:
        One item per input, each carrying whatever evidence fits.
    """
    return [
        with_evidence(
            item, evidence_for(item["answer"], documents), encoder, max_seq_len=max_seq_len
        )
        for item in items
    ]


def answer_span(
    full: Sequence[int], before: Sequence[int], after: Sequence[int]
) -> tuple[int, int]:
    """Locate the answer's tokens inside a rendering, without assuming addition.

    TOKEN COUNTS ARE NOT ADDITIVE, and computing this as ``len(before)`` is
    the bug that produced it. Byte-pair encoding re-tokenises across a join:
    measured on gpt2, appending the answer ``"AI"`` to one item's prefix left
    the id count UNCHANGED at 22, because the appended text merged into the
    prefix's final token rather than adding one. A span located by
    ``len(encode(before))`` is therefore wrong exactly where the tokenizer
    merged, and wrong silently -- it would have scored the wrong tokens and
    reported them as the answer's.

    So the span is found by agreement instead: the leading ids the rendering
    shares with the context before the blank, and the trailing ids it shares
    with the context after it. Whatever is left is the answer, plus any
    boundary token the merge absorbed. Including that token is deliberate --
    it is the same token in every arm, so it cannot bias a comparison, and
    excluding it would need the tokenizer to explain a merge it does not
    report.

    Args:
        full: Ids of the whole rendering.
        before: Ids of the text preceding the blank.
        after: Ids of the text following it.

    Returns:
        ``(start, stop)`` bounds of the answer's tokens in ``full``. ``stop``
        is never below ``start``.
    """
    start = 0
    while start < len(before) and start < len(full) and before[start] == full[start]:
        start += 1
    trailing = 0
    while (
        trailing < len(after)
        and trailing < len(full) - start
        and after[len(after) - 1 - trailing] == full[len(full) - 1 - trailing]
    ):
        trailing += 1
    return start, len(full) - trailing


def answer_nll(
    item: ClozeItem,
    model: ScoreableLMProto,
    encoder: Encoder,
    *,
    device: str,
    max_seq_len: int,
) -> float:
    """Negative log-likelihood the model assigns to the ANSWER's own tokens.

    THE MEASUREMENT WITH NO DISTRACTOR POLICY IN IT, and it exists because the
    multiple-choice arms turned out to be dominated by one. Measured on gpt2
    over the same 24 items: with one repeated distractor triple the base
    scored 0.2500 -- chance exactly -- and the cartridge 0.5417 at p=0.006;
    with distractors rotated per item the base scored 0.5417 and the cartridge
    0.5833 at p=1.0. Same corpus, same items, same models, opposite
    conclusions. The first looked like a finding and was a property of three
    words.

    Scoring the answer span directly removes the knob. There is nothing to
    choose except the item, so the number cannot be moved by how the wrong
    candidates were picked. It answers a narrower question than the accuracy
    arms -- "does this model find the true continuation less surprising here"
    rather than "can it pick it out of a line-up" -- and that question has one
    answer rather than one per policy.

    Only the answer's tokens are counted. The template's are identical across
    arms and would swamp a short answer's contribution, which is how a
    difference in one word gets averaged into invisibility.

    Args:
        item: The item whose answer is scored.
        model: The model to score under. A cartridge-wrapped model supplies
            its own prefix, so the arms differ only in this argument.
        encoder: Tokenizer.
        device: Torch device string.
        max_seq_len: Token budget; the rendering is truncated to it.

    Returns:
        Summed negative log-likelihood over the answer's tokens.

    Raises:
        AppError: With ``CLOZE_ITEM_UNSCOREABLE`` when the answer contributes
            no token to score -- an empty answer, or one truncation removed
            entirely. Returning zero would read as a perfect score.
    """
    before, _, after = item["template"].partition(BLANK_MARKER)
    full_ids = encoder.encode(before + item["answer"] + after).ids[:max_seq_len]
    start, stop = answer_span(full_ids, encoder.encode(before).ids, encoder.encode(after).ids)
    # A CAUSAL MODEL CANNOT SCORE THE SEQUENCE'S FIRST TOKEN: nothing precedes
    # it, so there is no distribution that predicted it. An item whose template
    # BEGINS with the blank starts its answer at position zero, and the naive
    # slice `logits[0, start - 1 : ...]` then indexes -1, which Python reads as
    # the LAST position and silently yields an empty selection rather than an
    # error. Scoring begins at the first token that has a predecessor.
    start = max(start, 1)
    if stop <= start:
        raise AppError(
            ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE,
            (
                f"item {item['item_id']!r} contributes no scoreable answer token: the "
                f"rendering's context accounts for all {len(full_ids)} of its tokens. "
                f"An empty answer does this; so does a max_seq_len that truncated the "
                f"answer away; and so does a one-token answer at the very start of the "
                f"template, which no causal model can score because nothing precedes it"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE),
        )

    # EVAL MODE IS SET HERE RATHER THAN ASSUMED OF THE CALLER, which is what
    # `score_cloze_items` does for the same reason. `train_cartridge` leaves
    # the base in TRAINING mode, so a measurement that inherited the caller's
    # mode would run GPT-2's three 0.1 dropouts and return a different number
    # every call. That exact oversight, in a scratch script, once produced a
    # 0.43 logit difference that was read as evidence that composition order
    # matters; it does not, and the difference was dropout.
    model.eval()
    # Allocated and filled rather than `torch.tensor([...])`, which is typed
    # as returning Any and would put an unchecked value into the scoring path.
    input_ids = torch.empty((1, len(full_ids)), dtype=torch.long, device=device)
    for position, token in enumerate(full_ids):
        input_ids[0, position] = token
    with torch.no_grad():
        out = model.forward(input_ids=input_ids, labels=input_ids)
    if not isinstance(out, LogitsOutProto):
        raise AppError(
            ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE,
            (
                "this model's forward returned no per-token scores, so the likelihood "
                "of one answer cannot be separated from the rest of the sentence; a "
                "loss is a mean over every predicted token and cannot say what one "
                "of them cost"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE),
        )
    # Position i predicts token i+1, so the logits scoring token `start` sit
    # at `start - 1`. Slicing after the log-softmax rather than before keeps
    # the normaliser over the whole vocabulary.
    log_probs = torch.log_softmax(out.logits[0, start - 1 : stop - 1, :], dim=-1)
    targets = input_ids[0, start:stop]
    picked = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -float(picked.sum().item())


def answer_nll_pairs(
    items: Sequence[ClozeItem],
    baseline: ScoreableLMProto,
    treatment: ScoreableLMProto,
    encoder: Encoder,
    *,
    device: str,
    max_seq_len: int,
) -> PairedComparison:
    """Compare two models on the answer-token likelihood of every item.

    Args:
        items: The shared question set.
        baseline: The control model.
        treatment: The model being tested.
        encoder: Tokenizer.
        device: Torch device string.
        max_seq_len: Token budget.

    Returns:
        The paired comparison, in units of summed negative log-likelihood over
        each answer. ``improved`` counts items the treatment found less
        surprising.
    """
    return summarise_pairs(
        [
            PairedItemOutcome(
                index=index,
                baseline=answer_nll(
                    item, baseline, encoder, device=device, max_seq_len=max_seq_len
                ),
                treatment=answer_nll(
                    item, treatment, encoder, device=device, max_seq_len=max_seq_len
                ),
            )
            for index, item in enumerate(items)
        ]
    )


def _as_loss(outcome: ClozeItemOutcome) -> float:
    """Read one item's correctness as a loss, so the paired machinery applies.

    Zero for right and one for wrong. The comparison layer is written around
    "lower is better" and its McNemar test conditions on discordant pairs,
    which is exactly the classical use of that test: two classifiers on one
    item set. Mapping correctness onto the same axis reuses it rather than
    writing a second, differently-shaped comparison beside it.

    Args:
        outcome: One item's outcome.

    Returns:
        0.0 when the model chose the answer, 1.0 otherwise.
    """
    return 0.0 if outcome["correct"] else 1.0


def compare_arms(baseline: ClozeEvalResult, treatment: ClozeEvalResult) -> PairedComparison:
    """Compare two arms item by item.

    Args:
        baseline: The control arm's result.
        treatment: The arm being tested.

    Returns:
        The paired comparison. ``mean_baseline`` and ``mean_treatment`` are
        ERROR RATES, because correctness enters as a loss; subtract from one
        for accuracy. ``improved`` counts items the treatment got right and
        the baseline got wrong.

    Raises:
        KeyError: If the two arms did not score the same items. That is a
            caller mistake and a silent one otherwise -- pairing by position
            would compare different questions and report the difference as an
            effect.
    """
    by_id = {outcome["item_id"]: outcome for outcome in treatment["outcomes"]}
    pairs = [
        PairedItemOutcome(
            index=index,
            baseline=_as_loss(outcome),
            treatment=_as_loss(by_id[outcome["item_id"]]),
        )
        for index, outcome in enumerate(baseline["outcomes"])
    ]
    return summarise_pairs(pairs)


__all__ = [
    "EVIDENCE_JOINER",
    "EVIDENCE_MARGIN_TOKENS",
    "answer_nll",
    "answer_nll_pairs",
    "answer_span",
    "compare_arms",
    "evidence_budget_tokens",
    "evidence_for",
    "longest_rendering_tokens",
    "retrieval_items",
    "with_evidence",
]
