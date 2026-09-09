"""What NAMES a run: which questions it asked, and which plan asked them.

TWO IDENTITIES, ONE ROLE. :func:`question_set_digest` answers "were these
the same questions" and :func:`qa_plan_label` answers "was this the same
measurement asking them". Both are derived from a run's inputs and both
exist so two records cannot be differenced unless they describe the same
thing; the label joined the digest here on 2026-09-09, when the plan table
it used to live in passed the 600-line ceiling and the split by role put
declaring a measurement on one side and naming it on the other.

THE DEFECT THIS EXISTS TO CLOSE, measured on this repository's own records.
Two `cartridge-question-set` records written five days apart carry the same
experiment, the same label -- plan fields plus a corpus digest -- the same
fingerprint, and the same empty payload digest. One asked 24 questions and the
other 32, because a defect in how held-out windows were chosen was fixed
between them, and no field of the record shape moved. Every identity a reader
has says the two are the same measurement, so
:func:`~platform_core.run_record.compare_runs` subtracts one from the other
and reports a difference in accuracy that is really a difference in the
question set.

The plan label cannot cover this and should not try. It names what was
REQUESTED -- window, slots, seeds, the corpus that went in -- while the item
set is what the code DERIVED from that request, so the two move independently
by construction. A digest over the derived items is the only thing that
separates a change of question from a change of answer.

WHY THE WHOLE ITEM AND NOT ITS IDENTIFIER. The distractor policy alone moved
the base model on this corpus between chance (0.2500) and 0.5417 while every
`item_id` stayed the same: identifiers are document-plus-term, and the
candidates offered beside the answer are not in them. A digest over ids would
have called those two runs identical, which is the exact failure this module
is named for.

ORDER IS PRESERVED rather than sorted, following
:func:`~model_trainer.cli.continuations.manifest_digest`: the items are built
by one deterministic pass over the corpus, so a reordering is a real change in
what ran and is worth catching rather than normalising away.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from platform_core.json_utils import JSONValue, dump_json_str

from model_trainer.core.contracts.cloze import ClozeItem, encode_cloze_item
from model_trainer.core.services.model.cartridge_qa_plans import QaPlan

QUESTION_SET_DIGEST_PREFIX = "sha256:"
"""What a digest from this module starts with, so a reader knows the algorithm."""


def question_set_digest(items: Sequence[ClozeItem]) -> str:
    """Digest a question set's full content in the order it was built.

    Args:
        items: The items a run measured, in build order.

    Returns:
        :data:`QUESTION_SET_DIGEST_PREFIX` followed by the hex SHA-256 of the
        canonical JSON of every item's encoding. An empty set digests to the
        digest of ``[]`` rather than to an empty string, because a run that
        asked nothing is a fact about that run and not a missing measurement.
    """
    encoded: list[JSONValue] = [encode_cloze_item(item) for item in items]
    canonical = dump_json_str(encoded)
    return f"{QUESTION_SET_DIGEST_PREFIX}{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def qa_plan_label(name: str, plan: QaPlan, *, digest: str) -> str:
    """Build the label that identifies one plan's numbers on one corpus.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the corpus, from
            :func:`~cartridge_plans.corpus_digest`.

    Returns:
        The label, e.g.
        ``gpt2-wiki-qa-gpt2-w256-s4-c128-m896-e12-lr0.01-d3-n120-seeds7.8.9-1a2b3c4d``.
    """
    seeds = ".".join(str(seed) for seed in plan["seeds"])
    return (
        f"{name}"
        f"-{plan['model_id']}"
        f"-w{plan['window']}"
        f"-s{plan['held_out_stride']}"
        f"-c{plan['num_slots']}"
        f"-m{plan['max_seq_len']}"
        f"-e{plan['epochs']}"
        f"-lr{plan['learning_rate']}"
        f"-d{plan['distractor_count']}"
        f"-n{plan['max_items']}"
        f"-seeds{seeds}"
        f"-{digest[:12]}"
    )


__all__ = [
    "QUESTION_SET_DIGEST_PREFIX",
    "qa_plan_label",
    "question_set_digest",
]
