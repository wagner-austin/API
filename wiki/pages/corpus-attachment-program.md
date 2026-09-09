---
title: Attaching a corpus to a model — four arms, the one that works from 774M, and the one that clean input does not rescue
tags: [ml, model-trainer, cartridges, retrieval, model-editing, research-program]
related:
  - "[[model-trainer-composition-ceiling]]"
  - "[[model-trainer-cartridge-question-set]]"
  - "[[model-trainer-noise-floor-is-a-range]]"
  - "[[model-trainer-companioned-training-recipe]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_report.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_retrieval.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_dense.py
  - services/Model-Trainer/src/model_trainer/core/services/model/corpus_cloze.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/apply.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/grounding.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/curated_triples.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/triple_edit_arm.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/triple_edit_plans.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/value_optimisation.py
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py": 19bb7f1c1e1ee5e71677b6d37d986f7073723b44
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_report.py": e1016b716bea823766336a2b157a1c9587395259
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_retrieval.py": 36950df0ed6f7a4af8ae37e3550482f1203aa376
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_dense.py": 883e01b8afba699654da81e92e8e37eca608b234
  "services/Model-Trainer/src/model_trainer/core/services/model/corpus_cloze.py": 509132ebe55fc8717e973b3183bd3018d9b0ec58
  "services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py": 8f2fd9d682790c501b7255cba8c6187a33a7b81c
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/apply.py": 411c47f975fad5b700bcc6948da496f9af9c4c73
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/grounding.py": 3315e270ab5f129eb92acc25bc90aa16d610ac0b
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/curated_triples.py": c8294f15c0d14b61604c02418df58f0f9dfbfec7
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/triple_edit_arm.py": 4e6d8ca11243efbf52623609e452e2c1088346b9
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/triple_edit_plans.py": f7af025c96945aa117cba11617c24a35fecd0784
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/value_optimisation.py": 2d71c8069f08a946dd47bccf8e03d97825a3e6c7
provenance:
  - "all arms measured 2026-09-08/09 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1, --controls none"
  - "records qa-svc-{gpt2,gpt2-medium,gpt2-large,gpt2-xl}.json, 32 items each, corpus digest e2f23c635583, one card, one determinism setting"
  - "DETERMINISM CERTIFICATE: four independent runs hours apart, across substantial changes to the retrieval path, agree on 12 shared accuracy fields at every rung"
  - "SUPERSEDED: the 2026-09-07 run (qa-record-bm25.json, 24 items) predates commits 9ba9dfb6 and eb73abf8 and its verdict is retracted here; the 2026-09-08 ladder's LATENCY figures predate 62315399 and 00602af9 and are retracted too"
  - "triple-edit records triple-{gpt2-triples-dose-10s-lr005,gpt2-medium-triples,gpt2-large-triples,gpt2-xl-triples}.json at half depth and the same four with -rome-depth at ROME's 17/48 fraction; same 32 items, same corpus digest, dose 10 x 0.05 throughout, one card"
  - "the triple-edit arm has NO seed axis: the value search starts from zeros and is deterministic given the plan, so its accuracy differences carry no noise floor and only gpt2-xl's exact zero across thirteen edits is a statement"
  - "board tasks d3639e09 (weight injection), d3742672 (steering vectors), 1fc5afed (cartridges -- its closure result is WRONG, see the 23:08Z note on that task), ac5f88cb (scale ladder, done), a8f799c5 (retrieval methods, done), 3fc98ed6 (corpus representation, measured 2026-09-09, held open on the shallower-site caveat), 74dd514e (persona adapter, never started)"
  - "AKEW figures from wiki page wu-2024-akew-editing-in-the-wild in the personal wiki, read page by page"
  - "RETRACTED 2026-09-09, same day it was published: the claim that the cartridge beats lexical, dense and fused retrieval. The accuracy numbers in the table are unchanged and were not the error -- reading a 1.7-item and a 1.3-item margin out of 32 as a lead was. McNemar exact on 32 paired items cannot declare a net difference below 6 items significant under any arrangement of the discordant pairs, since the most favourable configuration for a net of d gives p = 2*(0.5)^d. The cartridge-vs-base rows (+8.7, +8.3 items) clear that floor and survive; the cartridge-vs-retrieval rows never did. The floor was never computed before publication, which is the defect, not the size of the number."
  - "RETRACTED 2026-09-09 in the same pass, and not part of the headline, which is why it needed a second reading: 'dense retrieval loses to plain BM25 at every rung'. The margins are 1, 3, 1 and 1 items of 32, all under the same floor. Dense was not shown to retrieve worse -- it was shown to be unresolvable from BM25 on this question set. The four rungs agreeing is not four pieces of evidence; it is one 32-item set scored four times. What still clears the floor in that comparison is the ORACLE GAP: fused retrieval trails the oracle by 6 to 8 items at every rung."
fact_checked: "2026-09-09"
confidence: high
hubs: [services]
---

# Attaching a corpus to a model

One question has been asked four ways over six days: **can a model be made
to answer questions about a private corpus without retrieving at query
time?** The arms were run as separate board tasks by separate sessions and
each produced its own page, so the program has never been written down as
one thing. A reader arriving at any single page cannot tell which of four
attempts they are looking at, or which of them are answered.

**The answer is yes from about 774M parameters — against the un-augmented
base, and only against that.** It was recorded as *no* for a day, on a 124M
measurement from an instrument that examined nine of the twelve pages it
trained on.

**Read the scope of that sentence carefully, because it was written wider on
2026-09-09 and had to be narrowed the same day.** A cartridge answers 8.7
more of 32 questions at 774M than the same model with nothing attached, and
that margin clears the smallest difference this question set can resolve.
Whether it beats RETRIEVING the same corpus at query time is a separate
question and is **not answered here**: the cartridge−BM25 margin is 1.7
items, and 32 paired items cannot resolve fewer than 6. The arithmetic is
under *The retrieval comparison, and why it is not a result*, below.

This page is the index. It carries no measurement of its own.

## The arms, and where each stands

| arm | mechanism | verdict |
|---|---|---|
| Weight injection | rank-one edit into the parameters | **No, and now measured rather than argued** — hand-curated triples clear the bar 13 times in 20, every edit lands at every scale, and no rung answers one more question |
| Steering vectors | add a direction to the residual stream | **No** — composition degrades to nothing |
| Cartridges | train a KV prefix, serve it ahead of the query | **Yes against the un-augmented base from ~774M** — +8.7 and +8.3 items of 32, clearing the instrument's floor. **NOT established against retrieval**: the cartridge−BM25 margin is 1.7 and 1.3 items, below anything 32 items can resolve (retracted 2026-09-09, see below). Cheaper per request than dense and fused everywhere, and than BM25 at 1.5B |
| Persona adapter | not started | untested |

The cartridge row said **No** until 2026-09-08. That verdict came from a
single 124M measurement on an instrument carrying two defects, and both the
defects and the missing rungs were found by questions the author of this
page had already dismissed. The retraction is documented below rather than
edited away, because the way it failed is more transferable than the result.

## Weight injection: the algebra works, the corpus does not fit it

The mechanism is sound and committed. A rank-one update composes, orients
correctly for GPT-2's transposed `c_proj`, applies, and moves every output by
exactly the predicted delta — agreement to 9.2e-07 — and restoring from the
snapshot returns the parameter byte-for-byte.

The blocker is the INPUT. Locate-then-edit methods take
subject-relation-object triples. AKEW measures those methods at **2.25-4.78%**
on real-world updates reached through automatic triplet extraction, against
**93-99%** on the same facts supplied as clean triples. Measured against this
wiki: **29 of 11,914 claims, 0.24%**, are already edit-shaped, median claim 27
words.

A cited wiki is prose, and prose is the regime these methods score single
digits in.

### And when the triples are supplied by hand, the method still buys nothing

That was the argument until 2026-09-09. It has now been TESTED rather than
argued, because "prose does not become a rank-one edit" is a claim about the
prose that only a curated representation can check.[^triples]

**The cost of a usable representation.** Twenty triples were curated by hand,
one per distinct answer in the same 32-item question set, each held to a
seven-clause mechanical gate: the source sentence must be one of the TRAINING
half's sentences, both ends of the association verbatim in it, object after
subject, no giveaway in the prompt, and the subject must be a corpus term by
the same extractor the question set uses to decide what it may ask about.

**13 of 20 clear it; 7 do not.** Six fail on the subject not being an entity
and two on the answer preceding it, and **zero** fail any clause a curator
controls — so the rejections are facts about the prose. Every one of the seven
appears in the training text as the FIRST entity in its sentence, leaving no
earlier entity to hang an association on. Three of the thirteen that pass are
grounded only in a filesystem path or a footnote, which is the ghost-term
problem in a new place.

**What the accepted thirteen do.** Written into the weights at half depth,
one dose, the same question set.[^triples]

| rung | base | after 1 edit | after 13 | gain | edit success | target NLL |
|---|---|---|---|---|---|---|
| gpt2 124M | 0.4375 | 0.4062 | 0.3438 | −0.094 | 1.00 | 23.3 → 15.9 |
| gpt2-medium 355M | 0.5625 | 0.5312 | 0.5938 | +0.031 | 1.00 | 21.7 → 19.3 |
| gpt2-large 774M | 0.5312 | 0.5000 | 0.4688 | −0.062 | 1.00 | 22.3 → 18.9 |
| gpt2-xl 1.5B | 0.5625 | 0.5625 | **0.5625** | +0.000 | 1.00 | 21.4 → 18.9 |

Every edit lands at every scale — success 1.00, target surprise down by 2.4 to
7.4 — and no rung answers more questions. **gpt2-xl's per-edit curve is 0.5625
thirteen times in a row**: not one of the 32 items moved while thirteen facts
were written into the parameters. That is AKEW's dissociation between edit
success and downstream answering, reproduced on this corpus at four scales.

**And it is not the site.** Half depth is not where the reference
implementation writes — ROME targets layer 17 of gpt2-xl's 48, about a third
down — so the whole ladder was re-run at that fraction (4/12, 8/24, 13/36, and
ROME's own 17/48), with dose, corpus, triples, fact token and question set
identical, so each rung differences against its half-depth twin on depth
alone.[^depths]

| rung | half depth | at ROME's fraction |
|---|---|---|
| gpt2 124M | L6 → 0.3438 (succ 1.00) | L4 → 0.3750 (succ 0.92) |
| gpt2-medium 355M | L12 → 0.5938 (succ 1.00) | L8 → 0.4688 (succ **0.23**) |
| gpt2-large 774M | L18 → 0.4688 (succ 1.00) | L13 → 0.5000 (succ 1.00) |
| gpt2-xl 1.5B | L24 → **0.5625** (succ 1.00) | L17 → **0.5625** (succ 1.00) |

Eight configurations, two depths, four scales, base unchanged within each row.
**Not one gains more than a single item out of 32.** gpt2-xl is flat at 0.5625
at BOTH depths — twenty-six edits across the two runs, every one landing, and
the model answers the same eighteen questions it always did.

The shallower site is also LESS stable, which is the one thing the second
ladder adds beyond a confirmation. Edit success is 1.00 at half depth
everywhere; at ROME's fraction it is 0.92, 0.23 and 1.00 — and medium's
shallower rung is the only configuration measured anywhere in this program
where the edits leave their own targets MORE surprising than they started
(21.7 → 23.0). Thirteen sequential rank-one writes at an early layer compound
into each other's captures, because everything downstream of the site has
already moved by the time the next edit reads it.

A gpt2-only dose curve had shown accuracy falling monotonically with every
unit of target likelihood bought, the only harmless dose being the one that
installed nothing. **The ladder retracts the generality of that**: the damage
is a 124M phenomenon and does not survive scale. What survives is the absence
of any benefit.

**What this arm cannot say.** Its differences are one to three items out of 32,
and unlike the cartridge arm it has NO seed axis — the value search starts from
zeros and is deterministic given the plan, so there is nothing to vary and no
noise floor to build from. Medium's +0.031 and large's −0.062 are one and two
items and neither is a finding. What IS a statement is xl's exact zero across
thirteen edits at each of two depths, and the fact that eight configurations
produced no gain anywhere.

Still untested: batched editing. Thirteen SEQUENTIAL rank-one writes are not
MEMIT, which solves for many associations at once and does not let each edit
read a model the previous ones have already moved — and the 0.23 success rate
at medium's shallow site is exactly the failure mode batching exists to avoid.

[^depths]: `services/Model-Trainer/src/model_trainer/core/services/model/editing/triple_edit_plans.py`
    § `TRIPLE_EDIT_PLANS` holds both ladders; the ROME-fraction rungs carry the
    `-rome-depth` suffix and `tests/test_triple_edit_plans.py` asserts each one
    differs from its half-depth twin in the layer alone. Commit `827f688c`.

[^triples]: `services/Model-Trainer/src/model_trainer/core/services/model/editing/grounding.py`
    is the gate, `editing/curated_triples.py` the twenty attempts with their
    source sentences, `editing/triple_edit_arm.py` the arm and
    `editing/triple_edit_plans.py` the dose curve and the ladder. Commits
    `f411fd19`, `cf344758`, `98438f67`; board task `3fc98ed6` carries the full
    trail including the instrument defect found and corrected mid-run.

## Steering vectors: bounded by published results, not by this stack

Closed on the literature rather than on a local measurement, because the
literature already answers it at a scale this machine cannot match. ASTEER
reports 10.69-23.18% steering success over 1.42M labelled generations.
Subbiah reports 15.7-40.1 points of trait expression lost at TWO composed
vectors, with nothing left at four.

The arm's own S0 gate concluded that building a local harness would replicate
known results below the state of the art. What shipped instead was the corpus
update — four papers into the personal wiki.

## Cartridges: the arm that beats its own base from 774M

The only arm carried to a full comparison on real data. Four model sizes, the
same 32 held-out questions about the 12 public me-wiki pages, one card, one
corpus digest, chance 0.25:

| model | base | + cartridge | gain | spread | + BM25 | + dense | + fused |
|---|---|---|---|---|---|---|---|
| gpt2 124M | 0.4375 | 0.6042 | +0.167 | 0.031 ✱ | **0.7500** | 0.7188 | **0.7812** |
| gpt2-medium 355M | 0.5625 | 0.7292 | +0.167 | 0.188 | **0.7812** | 0.6875 | 0.7500 |
| gpt2-large 774M | 0.5312 | **0.8021** | +0.271 | 0.125 ✱ | 0.7500 | 0.7188 | 0.7500 |
| gpt2-xl 1.5B | 0.5625 | **0.8229** | +0.260 | 0.031 ✱ | 0.7812 | 0.7500 | 0.7812 |

✱ the gain exceeds its own seed spread. Oracle retrieval scores 0.9688 at
gpt2 and 1.0000 at every larger rung.

Per-request cost, ms/item, with index builds excluded and reported
separately (BM25 ~4.6 ms, dense ~340 ms, both one-time):

| model | cartridge | BM25 | dense | fused |
|---|---|---|---|---|
| gpt2 124M | 69.8 | **64.3** | 81.9 | 82.7 |
| gpt2-medium 355M | 127.0 | **122.2** | 142.3 | 137.2 |
| gpt2-large 774M | 177.4 | **170.5** | 195.3 | 196.8 |
| gpt2-xl 1.5B | **250.7** | 270.5 | 299.4 | 309.5 |

~~**The cartridge overtakes every retriever on accuracy between 355M and
774M.**~~ **RETRACTED 2026-09-09. That sentence read a difference this
instrument cannot resolve.** What it is cheaper than survives — the latency
table is a per-request measurement, not a 32-item comparison — so the
cartridge is still cheaper per request than dense and fused at every rung,
and cheaper than BM25 at 1.5B.

### The retrieval comparison, and why it is not a result

**The arithmetic, because the retraction is a number and not a doubt.**
These arms are scored on the SAME 32 items, so the comparison is paired and
McNemar's exact test is the one that applies. It reads only the discordant
pairs — items one arm got right and the other wrong — and its most
favourable possible configuration for a net difference of *d* items is
*d* discordant pairs all falling one way, giving a two-sided
p of 2·(0.5)^*d*. That is ≤ 0.05 only from *d* = 6 upward
(*d* = 6 → p = 0.031; *d* = 5 → p = 0.063). **Six items of 32 is
therefore the smallest net difference this question set can EVER declare
significant, no matter how the pairs fall.**

Against that floor:

| comparison | rung | net difference | items of 32 | vs the 6-item floor |
|---|---|---|---|---|
| cartridge − base | 774M | +0.271 | +8.7 | clears it |
| cartridge − base | 1.5B | +0.260 | +8.3 | clears it |
| cartridge − BM25 | 774M | +0.052 | +1.7 | **below it** |
| cartridge − BM25 | 1.5B | +0.042 | +1.3 | **below it** |

So the two headline rows of the accuracy table were never a result. A margin
of one or two items cannot reach significance at this n under any
arrangement of the underlying pairs, which means the correct report is *not
measured*, not *smaller than we hoped*.

**Clearing the floor is necessary, not sufficient.** The cartridge−base rows
pass the test of "could this n ever show it"; whether they DO is a question
about the actual discordant split, which needs the per-item paired data and
is not settled by the accuracy means alone. The floor rules things out. It
does not rule them in.

~~**Dense retrieval is the surprise, and it is a negative one. It loses to
plain BM25 at every rung.**~~ **RETRACTED 2026-09-09 BY THE SAME FLOOR, and
this one was not in the headline, which is why it took a second pass to
catch.** Dense trails BM25 by 1, 3, 1 and 1 items across the four rungs —
every one of them under 6. **Dense retrieval was not shown to lose here.**
It was shown to be indistinguishable from BM25 on an instrument that cannot
tell them apart, four times, which is a different sentence and a much
duller one.

That the four rungs all lean the same way is not four pieces of evidence: it
is the same 32 items scored four times, so the agreement across rungs is
mostly the agreement of one question set with itself.

**What does survive is the oracle gap.** Fusing the two closes none of it —
fused reads 0.7812 / 0.7500 / 0.7500 / 0.7812 against an oracle at 0.9688
and then 1.0000, a shortfall of 6 to 8 items at every rung, which clears the
floor at every rung. Something is being missed by both retrievers, and the
prediction made before the measurement — by
`packages/wiki-search/src/fusion.ts` in the MCPs repo, that a proper-noun
query needs the lexical hit even where the vector arm ranks it nowhere —
remains the plausible reading. **Plausible, not measured.** This corpus is
project names, ClearGBM, NavProbe, TankpitBot, which is exactly what lexical
matching rewards and embeddings blur; but the number that would show the
dense arm contributing little is precisely the number the floor rejects.

The shape is what makes the cartridge column readable rather than the
endpoint. Cartridge accuracy rises monotonically — 0.6042, 0.7292, 0.8021,
0.8229 — while BM25 stays flat at 0.7500 / 0.7812 / 0.7500 / 0.7812.
**BM25's flatness is expected and is the control**: it puts the answer's own
sentence in the window, so the answer is nearly given and the reader's
capacity barely matters.

**And the floor picks out the same rung the eye did, which is worth noting
because it did not have to.** The cartridge−base gain is +5.3 items at 124M
and +5.3 at 355M — under the floor — and +8.7 and +8.3 at 774M and 1.5B,
over it. The "yes from ~774M" boundary in this page's headline was first
drawn by looking at the curve against its seed spread; recomputing it as
"the first rung whose gain a 32-item paired test could ever resolve" puts it
in the same place.

**What the ladder cannot say.** 32 items over 3 seeds on one corpus. Base
accuracy is not clean across rungs (0.4375, 0.5625, 0.5312, 0.5625), so
rung-to-rung differences of a few points are not readable. The crossing
point is an interpolation between two rungs, not a measurement of where it
happens. And `max_seq_len` is held at 896 with evidence truncated to fit, so
this tests MODEL SCALE and not context length — which is the axis the
technique is actually for.

## How the first verdict was wrong, which is the transferable part

Recorded because the failure generalises past cartridges: **an instrument
can be green, self-consistent, and measuring three-quarters of its corpus.**

The 2026-09-07 run reported base 0.5417, cartridge 0.6389 with its gain
inside the seed spread, and BM25 0.8333 — and concluded the arm was dead.
Two defects produced it, both found by asking why a plan permitting 120
items returned 24, rather than accepting 24 as the corpus's size.

1. **The held-out stride counted across the corpus, not within a page**
   (fixed in `9ba9dfb6`). Which pages got tested depended on where their
   windows landed in the concatenated sequence. Three of twelve pages held
   out nothing at all: trained on, never examined. Every arm was fitted to
   twelve pages and scored on nine, and nothing surfaced it because a short
   question set looks exactly like a short corpus.
2. **A term qualified an item from text the evidence could not cite**
   (fixed in `eb73abf8`). `build_items` read the raw training text;
   `evidence_for` reads sentences, which strip code fences, table rows and
   URLs. Nine terms of ninety-four lived only in stripped constructs. Three
   reached items no retrieval arm could answer — biasing **toward** the
   cartridge — and one of them crashed the run outright once page 6 was
   finally examined, which is how the second defect was found at all.

Item count on the real corpus: 24 → 35 after the first fix → 32 after the
second removed the unsupportable ones.

The instrument passed `make check` at 100% statements and branches
throughout. Coverage measures whether a line ran, never whether the thing it
computed was the thing intended.

## What the cartridge result does NOT cover

Stated because a positive reads broader than a negative did, and this one is
narrow in three ways that still matter.

1. **It is not the published Cartridges system.** Eyuboglu et al. need Llama
   or Qwen3, a two-stage pipeline with a synthesis server, and wandb for
   artifact loading. What was measured is this repo's own simpler thing:
   direct context distillation over corpus windows.
2. **It is still measured outside the regime cartridges are for.** They are a
   CONTEXT COMPRESSION technique — the claimed win is serving a very long
   context cheaply. The window here is 896 tokens with evidence truncated to
   fit. At that size there is nothing to compress, so the arm won on a
   question the technique is not designed around. That makes the win more
   surprising, not less caveated.
3. **No retrieval arm has been beaten, and this item has now been wrong
   twice in opposite directions.** The history is kept because the second
   error is the instructive one.

   * It first read *"only lexical retrieval was beaten"* — arguing that
     beating BM25 is not beating retrieval, because a dense retriever
     "would cost real milliseconds and would also retrieve better".
   * On 2026-09-09 it was rewritten to *"the cartridge now leads all three
     retrievers at 774M and 1.5B"*, on the strength of the accuracy table
     above. Board `a8f799c5` carries that measurement.
   * **That rewrite is retracted the same day.** The lead it claimed was
     1.7 and 1.3 items out of 32, and the smallest net difference this
     question set can ever resolve is 6. There was no lead to report.

   Of the original reasoning, the COST half survives and the RANKING half
   does not. A dense query does cost ~20 ms/item against BM25's ~1.5 —
   that is a per-request timing, not a 32-item comparison, and no floor
   applies to it. But "dense retrieves worse here" fails the same test the
   headline failed: the margins are 1, 3, 1 and 1 items. It was written
   into this page as a finding on 2026-09-09 and is struck above. The one
   ranking claim that clears the floor is the oracle gap, 6 to 8 items at
   every rung, which says both retrievers miss something and does not say
   which of them misses more.

   **What the sequence should be read for.** The first version was a caveat
   the author distrusted and tested. The second was the caveat's removal on
   the first evidence that pointed the desired way, WITHOUT asking whether
   the instrument could see a difference that size — and it went into a page
   marked `confidence: high` on the same day it was measured. The failure is
   not that the number was small. It is that no floor was computed before
   the caveat was struck.

4. **A stronger embedder is untested.** The dense arm ran `thenlper/gte-base`,
   chosen to match the family `wiki-search` deploys rather than to win. This
   result is about THIS corpus's proper-noun density, not about dense
   retrieval in general.

## The lever that was untested, and what testing it settled

Every arm above tested a METHOD. None tested the CORPUS REPRESENTATION, and
until 2026-09-09 this section argued that the omission mattered.

The argument was: AKEW's numbers say the format is the bottleneck — the same
facts score 93-99% as clean triples and 2.25-4.78% extracted from prose, and
this wiki is 0.24% triple-shaped — so "prose does not become a rank-one edit"
might be a fact about the prose rather than about the method. The naive test
was already known to fail, because having a model extract triples from prose
IS the automatic extraction AKEW measured in the single digits. What was
needed was high-precision triples behind a verification gate, which converts
the question from "does the method work" into "what does a usable
representation cost".

**That was built and run, and the answer is above.** A usable representation
costs a 0.35 reject rate on hand-curated triples, and it buys nothing: every
accepted association lands in the weights at every scale, and no rung answers
one more question. The two facts still sit next to each other, but the second
is now measured rather than inferred — **the method that tolerates prose
works, and the method that cannot is not rescued by being handed clean input.**

The shallower site ROME itself uses was the last caveat, and it was run rather
than left standing: it does not change the verdict. What remains untested is
BATCHED editing, and whether a corpus REWRITTEN into triple-shaped prose would
change what the CARTRIDGE arm can do — a different question, since that arm
reads windows rather than associations, and it is the arm that works.

## Reading order for the children

[[model-trainer-noise-floor-is-a-range]] first if you intend to compare any
two numbers here — it explains why the sweep's own floor is not a
significance test and cannot be compared across seed counts.
[[model-trainer-cartridge-question-set]] for how the question set is built and
why its accuracy arm is sensitive to the distractor policy.
[[model-trainer-composition-ceiling]] and
[[model-trainer-companioned-training-recipe]] for the composition arc, which
asks a different question — what a cartridge retains when another is
concatenated in front of it — and is not settled by anything above.
