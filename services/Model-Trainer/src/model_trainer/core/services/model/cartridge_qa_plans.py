"""Which question-set measurements exist, and what identifies each one's numbers.

A SEPARATE PLAN FROM :mod:`cartridge_plans`, DELIBERATELY. That table's plans
sweep slot counts and compose cartridges, and are identified by a label naming
those fields. This one runs a fixed cartridge against a question set and is
identified by the question set's shape -- how many distractors, how many items.
Folding both into one TypedDict would give every plan fields the other half
does not use, and a label that names them anyway.

The two also answer different questions and must not be differenced: a loss
plan reports how surprising held-out prose is, and this reports whether the
model can use what the prose said. :data:`QA_EXPERIMENT` differs from
``CARTRIDGE_EXPERIMENT`` for exactly that reason -- the comparability layer
refuses to subtract records from different experiments, which is the behaviour
wanted here.

THERE IS NO ``require_qa_plan`` HERE, DELIBERATELY.
:func:`~cartridge_plans.require_cartridge_plan` is generic over the plan type,
so a second lookup beside it would be six lines that differ only in the noun
in their error message -- and would be the copy that stops matching when the
first one learns something.

WHAT MOVED THE ANSWER MOST, and why ``distractor_count`` is in the label. The
multiple-choice arms turned out to be dominated by which wrong candidates were
offered: measured on gpt2 over 24 items, a set built with one repeated
distractor triple put the base model at chance (0.2500) and the cartridge at
0.5417; rotating distractors per item put the base at 0.5417 and the cartridge
at 0.5833. Same corpus, same items, same models. Any field that can do that
belongs in the identity of the measurement.

TWO KINDS OF ENTRY. The plans below are WRITTEN: the me-wiki ladder that
produced the retracted headline, kept so its labels still resolve, and the
full-wiki base. Everything else is BUILT from that base by
:func:`~model_trainer.core.services.model.cartridge_qa_axes.full_wiki_family`
-- the scale ladder to pythia-6.9b and the slot, window and epoch axes --
so a cell differs from the base in the one field its axis owns and nowhere
else, by construction rather than by copying.
"""

from __future__ import annotations

from typing import Final

from platform_core.power_distributions import McNemarTest

from model_trainer.core.contracts.qa_plan import QaPlan
from model_trainer.core.services.model.cartridge_qa_axes import (
    FULL_WIKI_BASE,
    full_wiki_family,
    merged_plan_tables,
)

#: The written plans. ``gpt2-wiki-qa`` mirrors ``gpt2-wiki``'s corpus,
#: window, split and schedule so the two measurements describe the same
#: cartridge, and adds only what a question set needs.
#:
#: EVERY PLAN DECLARES 0.02, AND IT IS CHOSEN RATHER THAN DERIVED. Saying so
#: is the whole point of this paragraph, because the field's previous value
#: failed for the opposite reason.
#:
#: The anchors available to this programme:
#:
#:   WRAP, LLM rephrasing of C4 into four styles, 1:1 synthetic mix   +0.020
#:     (Maini et al., arXiv 2401.16380, 13 zero-shot QA benchmarks)
#:   this machine's extraction ablation, sentence-permuted copies     +0.029
#:   this machine's extraction ablation, 7:1 OSCAR dilution removed   +0.061
#:   this machine's extraction ablation, hub-slug markers             +0.004  (noise)
#:
#: 0.020 IS THE MOST CONSERVATIVE ANCHOR AVAILABLE, NOT A MEASURED FLOOR. It
#: is the minimum over a heterogeneous list, and a minimum over a mixed list
#: is a rule, not a derivation: the entry it lands on is WRAP -- published
#: literature, a different intervention, someone else's corpus -- so one
#: smaller published effect from any adjacent paper would move this number
#: without anything about THIS programme changing.
#:
#: It is chosen conservatively, and the asymmetry is the justification. A
#: too-LARGE smallest-effect-of-interest licenses a verdict that should not
#: exist; a too-SMALL one only refuses runs. Those errors are not
#: symmetric, and the expensive one is already on this programme's record.
#:
#: WHAT WOULD REPLACE IT: no decision this programme has taken yet fixes an
#: SEI the way ``clients/RustedWarfareBot``'s adoption log fixes theirs, where
#: the threshold is read off the smallest effect the campaign ever ACTED on.
#: When this programme first acts on an effect, that effect becomes the
#: anchor and this number should be replaced by it.
#:
#: A BETTER-DERIVED CANDIDATE WAS OFFERED AND DECLINED, recorded so nobody
#: has to rediscover it. **0.0294** -- the minimum over the effects the
#: SIBLING extraction ablation actually acted on, which is the rusted-shaped
#: derivation this one lacks. Its case, from
#: ``~/PROJECTS/wiki/pages/wiki-corpus-extraction-ablation.md``:
#:
#:   ACTED ON   augmentation, arm D - arm C, +2.94 points, p = 2e-3 / 2e-5 /
#:              4e-6, verdict "decisive". An entire follow-up experiment
#:              exists to decompose that arm, which is what makes it a
#:              DECISION rather than a reading.
#:   ACTED ON   dilution, arm B - arm A, -6.14 points, p to 9e-17; every
#:              later arm carries the 7:1 mixture. abs(-6.14) > 2.94, so the
#:              minimum over acted-on levers is still augmentation -- the
#:              check that makes 0.0294 a derivation rather than a lucky pick.
#:   DECLINED   the hub-slug marker, +0.36 points, p ~ 0.6, abandoned after
#:              all three specified conditions and a 355M rung at -0.37.
#:
#: (The ablation names that lever "augmentation" and the anchor list above
#: names its mechanism, "sentence-permuted copies". One lever, two names:
#: ``| D | C + four sentence-permuted copies of each page |``.)
#:
#: DECLINED FOR TWO REASONS, neither of which is that the derivation is
#: wrong. (1) The extraction ablation is a SIBLING programme -- different
#: intervention, different training regime, shared corpus family and units
#: -- so 0.0294 is borrowed too, one degree closer to home, and the
#: replacement condition above has no "closer" tier. (2) Between two
#: borrowed candidates the asymmetry is the only argument not about
#: provenance, and it favours the smaller.
#:
#: THE SEI WAS SELF-SEALING UNTIL THE CORPUS MOVED, and that is recorded
#: because the way out was corpus, not a different number. At 0.02 every
#: plan this table held on 2026-09-09 was refused -- the me-wiki plans at
#: 32 items, the api-codebase plans at 235 and 224 against the 250 needed --
#: so nothing could produce a result, and the condition that would retire
#: the number could never fire. The full wiki yields 3,278 to 3,827 items
#: per cell, fifteen times the gate, which is what the built family below
#: runs on.
#:
#: WHAT IT COSTS, STATED PLAINLY BECAUSE IT IS THE POINT. Resolving an effect
#: ``e`` needs about ``d / e`` items, where ``d`` is the fewest disagreements
#: that can ever reject -- and ``d`` IS NOT A CONSTANT OF THE ARITHMETIC, it
#: is a function of the two fields beside this one. At alpha 0.05 it is 5
#: under mid-p and 6 under exact;
#: :func:`~model_trainer.core.services.model.cartridge_qa_power.smallest_rejecting_discordant`
#: computes it rather than quoting it, and should be preferred over the table
#: below the moment any plan declares something other than mid-p at 0.05:
#:
#:   SEI       mid-p (d=5)   exact (d=6)
#:   0.05          100           120
#:   0.029         173           207
#:   0.02          250           300
#:
#: Raising the SEI so a small corpus clears it is the one thing this field
#: must never be used for: a threshold chosen so the corpus you have clears
#: it is not a threshold. Every plan stays gated by
#: :func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_question_set`
#: against the item count its corpus actually yields.
_WRITTEN: Final[dict[str, QaPlan]] = {
    "gpt2-wiki-qa": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 120,
        "smallest_effect_of_interest": 0.02,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "retrieved_chunks": 5,
        "expansion_feedback_chunks": 3,
        "expansion_terms": 5,
        "rerank_candidates": 20,
        "precision_selector": "policy",
    },
    # THE RETRACTED LADDER. Every field except `model_id` is copied from
    # `gpt2-wiki-qa` deliberately: the 2026-09-07 verdict -- that a cartridge
    # loses to BM25 on both accuracy and latency -- was measured at 124M, and
    # these rungs produced the "beats retrieval from ~774M" headline that was
    # withdrawn the day it was registered, because it rested on 32 items.
    # They stay so the labels those runs carried still resolve; the gate
    # refuses every one of them before a model loads.
    "gpt2-medium-wiki-qa": {
        "model_id": "gpt2-medium",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 120,
        "smallest_effect_of_interest": 0.02,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "retrieved_chunks": 5,
        "expansion_feedback_chunks": 3,
        "expansion_terms": 5,
        "rerank_candidates": 20,
        "precision_selector": "policy",
    },
    "gpt2-large-wiki-qa": {
        "model_id": "gpt2-large",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 120,
        "smallest_effect_of_interest": 0.02,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "retrieved_chunks": 5,
        "expansion_feedback_chunks": 3,
        "expansion_terms": 5,
        "rerank_candidates": 20,
        "precision_selector": "policy",
    },
    "gpt2-xl-wiki-qa": {
        "model_id": "gpt2-xl",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 120,
        "smallest_effect_of_interest": 0.02,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "retrieved_chunks": 5,
        "expansion_feedback_chunks": 3,
        "expansion_terms": 5,
        "rerank_candidates": 20,
        "precision_selector": "policy",
    },
    # THE INSTRUMENT THIS AXIS HAD BEEN SAID TO LACK, AND THE CORPUS WAS
    # NEVER WHAT IT LACKED. The retracted headline came from 32 items. The
    # response was to raise the cap to 240 and reach for the api-codebase
    # wiki, which yielded 237 on 2026-09-09 -- refused by thirteen items --
    # and the reading taken from that was that the axis needs an instrument
    # it does not have. Measured instead, the same day: ~/PROJECTS/wiki
    # offered 3,735 items, floor 0.00134, fifteen times below the declared
    # 0.02. On 2026-10-04 it offers 3,827 over 885 pages.
    #
    # `max_items` IS 4000 SO IT CANNOT BIND ON THIS CORPUS. A cap below what
    # the corpus offers makes the CAP the measurement's limit and hides the
    # corpus behind it -- which is how a 240-item cap came to be read as an
    # unavailable instrument.
    #
    # gpt2, BECAUSE THAT IS WHERE THE RETRACTED COMPARISON WAS MADE, and this
    # is the base every built cell in `cartridge_qa_axes` is one field away
    # from -- the scale ladder to pythia-6.9b, and the slot, window and epoch
    # axes.
    FULL_WIKI_BASE: {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 4000,
        "smallest_effect_of_interest": 0.02,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "retrieved_chunks": 5,
        "expansion_feedback_chunks": 3,
        "expansion_terms": 5,
        "rerank_candidates": 20,
        "precision_selector": "policy",
    },
}

#: Every plan: the written ones and the full-wiki family built from the base.
QA_PLANS: Final[dict[str, QaPlan]] = merged_plan_tables(
    _WRITTEN, full_wiki_family(_WRITTEN[FULL_WIKI_BASE])
)


__all__ = [
    "QA_PLANS",
]
