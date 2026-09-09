"""The triple-edit arm's plans, and the choices baked into them.

One plan names a question-set plan to borrow the corpus, the split, the items
and the budget from, plus the three things only an edit needs: where to write,
how long to look for the value vector, and how big a step to take.

THE LAYER IS DECLARED AND NOT SWEPT, and that is a stated limitation rather
than an oversight. Sweeping it and reporting the best rung would make this
arm's number the maximum over a search the other arms did not get, which is
how a comparison stops being one. Layer 6 of gpt2's 12 is the middle of the
band locate-then-edit work targets for this model family; a later run that
wants the sweep should add rungs and report every one.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict

from model_trainer.core.contracts.knowledge_edit import EditSite
from model_trainer.core.services.model.editing.curated_triples import (
    CORPUS_DIGEST,
    ME_WIKI_PUBLIC_TRIPLES,
)
from model_trainer.core.services.model.editing.grounding import TripleCandidate


class TripleEditPlan(TypedDict):
    """One complete, reproducible triple-edit measurement.

    Attributes:
        qa_plan: Which :data:`~model_trainer.core.services.model.cartridge_qa_plans.QA_PLANS`
            entry supplies the corpus split, the item set and the budget. Named
            rather than restated so this arm cannot drift from the question set
            it is compared against.
        corpus_digest: The corpus the plan's triples were curated against. The
            run refuses a corpus whose digest differs, because a reject rate
            measured elsewhere would be about a text nobody curated for.
        candidates: The curated triples, accepted and rejected alike. IN THE
            PLAN rather than imported by the arm, so a plan is a complete
            description of one measurement -- and so the arm is reachable with
            a small corpus and a few triples instead of only by running the
            real one.
        site: Where each edit is written.
        value_steps: Optimisation steps spent finding each value vector. Fixed
            rather than early-stopped, so two runs of one plan spend the same
            compute and their cost lines mean the same thing.
        value_learning_rate: AdamW step size for that search.
    """

    qa_plan: str
    corpus_digest: str
    candidates: tuple[TripleCandidate, ...]
    site: EditSite
    value_steps: int
    value_learning_rate: float


#: Fixed rather than a flag, and distinct from every other experiment's name.
TRIPLE_EDIT_EXPERIMENT: Final[str] = "corpus-representation-triple-edit"

#: The site every plan writes at. One object, so a rung of the dose curve
#: cannot differ from the reference plan in the site by accident.
_GPT2_SITE: Final[EditSite] = {
    "layer": 6,
    "module_template": "transformer.h.{}.mlp.c_proj",
    "fact_token": "subject_last",
}

TRIPLE_EDIT_PLANS: Final[dict[str, TripleEditPlan]] = {
    "gpt2-triples": {
        "qa_plan": "gpt2-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": _GPT2_SITE,
        "value_steps": 25,
        "value_learning_rate": 0.5,
    },
    # THE DOSE CURVE, and it exists because the reference plan's first edit
    # alone cost 0.1563 accuracy while installing its own association almost
    # perfectly. That is either a fact about editing this representation or a
    # fact about how hard the reference plan pushes: 25 AdamW steps at 0.5
    # drive the training loss to 0.003 and leave a delta of norm ~100 on a
    # 768-wide activation. A single rung cannot tell those apart, so these
    # walk the dose down by two orders of magnitude at fixed site and fixed
    # question set. Every rung is reported; none is the arm's number.
    "gpt2-triples-dose-25s-lr005": {
        "qa_plan": "gpt2-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": _GPT2_SITE,
        "value_steps": 25,
        "value_learning_rate": 0.05,
    },
    "gpt2-triples-dose-10s-lr005": {
        "qa_plan": "gpt2-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": _GPT2_SITE,
        "value_steps": 10,
        "value_learning_rate": 0.05,
    },
    "gpt2-triples-dose-5s-lr005": {
        "qa_plan": "gpt2-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": _GPT2_SITE,
        "value_steps": 5,
        "value_learning_rate": 0.05,
    },
    "gpt2-triples-dose-5s-lr001": {
        "qa_plan": "gpt2-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": _GPT2_SITE,
        "value_steps": 5,
        "value_learning_rate": 0.01,
    },
    # THE SCALE LADDER. The cartridge arm's verdict INVERTED between 355M and
    # 774M, so a representation result measured only at 124M says what that
    # one said: something about gpt2. These rungs hold the dose, the corpus,
    # the triples and the question set fixed and move only the base.
    #
    # THE DOSE IS 10 x 0.05 AT EVERY RUNG, and it is chosen rather than swept:
    # on gpt2 it is the cheapest rung whose edit success is 1.00, so every
    # association really lands and the accuracy it costs is the price of
    # landing them. The reference plan's 25 x 0.5 is excluded from the ladder
    # because at that dose the edits damage each other (success falls to 0.77),
    # which would confound scale with over-driving.
    #
    # THE LAYER IS THE SAME RELATIVE DEPTH, not the same index: 6 of gpt2's 12
    # is halfway down, so the rungs take 12 of 24, 18 of 36 and 24 of 48. That
    # is a CHOICE and it is not the reference implementation's -- ROME targets
    # layer 17 of GPT-2 XL's 48, about a third down. A negative ladder at half
    # depth therefore does not rule out a shallower site, and the write-up has
    # to say so.
    "gpt2-medium-triples": {
        "qa_plan": "gpt2-medium-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": {
            "layer": 12,
            "module_template": "transformer.h.{}.mlp.c_proj",
            "fact_token": "subject_last",
        },
        "value_steps": 10,
        "value_learning_rate": 0.05,
    },
    "gpt2-large-triples": {
        "qa_plan": "gpt2-large-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": {
            "layer": 18,
            "module_template": "transformer.h.{}.mlp.c_proj",
            "fact_token": "subject_last",
        },
        "value_steps": 10,
        "value_learning_rate": 0.05,
    },
    "gpt2-xl-triples": {
        "qa_plan": "gpt2-xl-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": {
            "layer": 24,
            "module_template": "transformer.h.{}.mlp.c_proj",
            "fact_token": "subject_last",
        },
        "value_steps": 10,
        "value_learning_rate": 0.05,
    },
    # THE SAME LADDER AT THE REFERENCE IMPLEMENTATION'S OWN DEPTH. The rungs
    # above edit halfway down because that is where the gpt2 plan happened to
    # sit; ROME targets layer 17 of gpt2-xl's 48, about a third. A negative
    # result at half depth is a result about half depth, so the fraction is
    # the one variable this second ladder moves: 17/48 = 0.354, giving 4 of
    # 12, 8 of 24, 13 of 36 and ROME's own 17 of 48.
    #
    # Everything else -- dose, corpus, triples, fact token, question set -- is
    # identical to the half-depth ladder, so a rung here is differenceable
    # against its half-depth twin and the difference is the depth.
    "gpt2-triples-rome-depth": {
        "qa_plan": "gpt2-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": {
            "layer": 4,
            "module_template": "transformer.h.{}.mlp.c_proj",
            "fact_token": "subject_last",
        },
        "value_steps": 10,
        "value_learning_rate": 0.05,
    },
    "gpt2-medium-triples-rome-depth": {
        "qa_plan": "gpt2-medium-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": {
            "layer": 8,
            "module_template": "transformer.h.{}.mlp.c_proj",
            "fact_token": "subject_last",
        },
        "value_steps": 10,
        "value_learning_rate": 0.05,
    },
    "gpt2-large-triples-rome-depth": {
        "qa_plan": "gpt2-large-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": {
            "layer": 13,
            "module_template": "transformer.h.{}.mlp.c_proj",
            "fact_token": "subject_last",
        },
        "value_steps": 10,
        "value_learning_rate": 0.05,
    },
    "gpt2-xl-triples-rome-depth": {
        "qa_plan": "gpt2-xl-wiki-qa",
        "corpus_digest": CORPUS_DIGEST,
        "candidates": ME_WIKI_PUBLIC_TRIPLES,
        "site": {
            "layer": 17,
            "module_template": "transformer.h.{}.mlp.c_proj",
            "fact_token": "subject_last",
        },
        "value_steps": 10,
        "value_learning_rate": 0.05,
    },
}


def triple_edit_plan_label(name: str, plan: TripleEditPlan, *, digest: str) -> str:
    """Name one run of one plan, carrying what makes it that run.

    Args:
        name: The plan's key.
        plan: The plan.
        digest: Corpus digest, truncated into the label the way the
            question-set arm truncates its own.

    Returns:
        The label.
    """
    site = plan["site"]
    return (
        f"{name}-{plan['qa_plan']}-L{site['layer']}-{site['fact_token']}"
        f"-vs{plan['value_steps']}-vlr{plan['value_learning_rate']}-{digest[:12]}"
    )


__all__ = [
    "TRIPLE_EDIT_EXPERIMENT",
    "TRIPLE_EDIT_PLANS",
    "TripleEditPlan",
    "triple_edit_plan_label",
]
