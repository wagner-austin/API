---
title: Attaching a corpus to a model — four arms, and the one that works from 774M
tags: [ml, model-trainer, cartridges, retrieval, model-editing, research-program]
related:
  - "[[model-trainer-composition-ceiling]]"
  - "[[model-trainer-cartridge-question-set]]"
  - "[[model-trainer-noise-floor-is-a-range]]"
  - "[[model-trainer-companioned-training-recipe]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_retrieval.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_dense.py
  - services/Model-Trainer/src/model_trainer/core/services/model/corpus_cloze.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/apply.py
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py": 1e45bd37cf0cf4f191b62cd43485d892e1901e3c
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_retrieval.py": 36950df0ed6f7a4af8ae37e3550482f1203aa376
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_dense.py": 883e01b8afba699654da81e92e8e37eca608b234
  "services/Model-Trainer/src/model_trainer/core/services/model/corpus_cloze.py": 509132ebe55fc8717e973b3183bd3018d9b0ec58
  "services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py": 8f2fd9d682790c501b7255cba8c6187a33a7b81c
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/apply.py": 411c47f975fad5b700bcc6948da496f9af9c4c73
provenance:
  - "all arms measured 2026-09-08/09 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1, --controls none"
  - "records qa-svc-{gpt2,gpt2-medium,gpt2-large,gpt2-xl}.json, 32 items each, corpus digest e2f23c635583, one card, one determinism setting"
  - "DETERMINISM CERTIFICATE: four independent runs hours apart, across substantial changes to the retrieval path, agree on 12 shared accuracy fields at every rung"
  - "SUPERSEDED: the 2026-09-07 run (qa-record-bm25.json, 24 items) predates commits 9ba9dfb6 and eb73abf8 and its verdict is retracted here; the 2026-09-08 ladder's LATENCY figures predate 62315399 and 00602af9 and are retracted too"
  - "board tasks d3639e09 (weight injection), d3742672 (steering vectors), 1fc5afed (cartridges -- its closure result is WRONG, see the 23:08Z note on that task), ac5f88cb (scale ladder, done), a8f799c5 (retrieval methods, done), 3fc98ed6 (corpus representation, open), 74dd514e (persona adapter, never started)"
  - "AKEW figures from wiki page wu-2024-akew-editing-in-the-wild in the personal wiki, read page by page"
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

**The answer is yes, from about 774M parameters.** It was recorded as *no*
for a day, on a 124M measurement from an instrument that examined nine of
the twelve pages it trained on.

This page is the index. It carries no measurement of its own.

## The arms, and where each stands

| arm | mechanism | verdict |
|---|---|---|
| Weight injection | rank-one edit into the parameters | **No, on this corpus** — bounded by ingest, not algebra |
| Steering vectors | add a direction to the residual stream | **No** — composition degrades to nothing |
| Cartridges | train a KV prefix, serve it ahead of the query | **Yes from ~774M** — beats lexical, dense AND fused retrieval on accuracy; cheaper per request than dense and fused everywhere, and than BM25 at 1.5B |
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

## Steering vectors: bounded by published results, not by this stack

Closed on the literature rather than on a local measurement, because the
literature already answers it at a scale this machine cannot match. ASTEER
reports 10.69-23.18% steering success over 1.42M labelled generations.
Subbiah reports 15.7-40.1 points of trait expression lost at TWO composed
vectors, with nothing left at four.

The arm's own S0 gate concluded that building a local harness would replicate
known results below the state of the art. What shipped instead was the corpus
update — four papers into the personal wiki.

## Cartridges: the arm that crosses every retriever between 355M and 774M

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

**The cartridge overtakes every retriever on accuracy between 355M and
774M.** It is also cheaper per request than dense and fused at every rung,
and cheaper than BM25 at 1.5B.

**Dense retrieval is the surprise, and it is a negative one.** It loses to
plain BM25 at every rung, and fusing the two closes none of the gap to the
oracle. The reason was predicted before it was measured, by
`packages/wiki-search/src/fusion.ts` in the MCPs repo: a proper-noun query
needs the lexical hit even where the vector arm ranks it nowhere. This
corpus is project names — ClearGBM, NavProbe, TankpitBot — which is exactly
what lexical matching rewards and embeddings blur. The dense arm contributes
little, so reciprocal-rank fusion has little to fuse.

The shape is what makes it readable rather than the endpoint. Cartridge
accuracy rises monotonically — 0.6042, 0.7292, 0.8021, 0.8229 — while BM25
stays flat at 0.7500 / 0.7812 / 0.7500 / 0.7812. **BM25's flatness is
expected and is the control**: it puts the answer's own sentence in the
window, so the answer is nearly given and the reader's capacity barely
matters. A rising curve and a flat one, crossing once.

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
3. ~~**Only lexical retrieval was beaten.**~~ **ANSWERED 2026-09-09, and the
   answer inverted the caveat.** This page previously argued that beating
   BM25 is not beating retrieval, because a dense retriever "would cost real
   milliseconds and would also retrieve better". Half of that was right: a
   dense query costs ~20 ms/item against BM25's ~1.5. The other half was
   wrong — dense retrieves WORSE here, at every rung, and the fusion of the
   two closes none of the oracle gap. The cartridge now leads all three
   retrievers at 774M and 1.5B. Board `a8f799c5` carries the measurement.

   The caveat is struck rather than deleted because its REASONING was sound
   and is what made the test worth running; only its prediction failed.

4. **A stronger embedder is untested.** The dense arm ran `thenlper/gte-base`,
   chosen to match the family `wiki-search` deploys rather than to win. This
   result is about THIS corpus's proper-noun density, not about dense
   retrieval in general.

## The untested lever, which the cartridge result makes MORE interesting

Every arm above tested a METHOD. None tested the CORPUS REPRESENTATION.

This mattered when all three arms had failed; it matters more now that one
has not. The cartridge succeeds by reading prose windows directly, and it is
the only arm that never needed the corpus reshaped. Weight injection is
still blocked on exactly that, and the two facts sit next to each other:
the method that tolerates prose works, and the method that cannot is the one
nobody has given a usable input to.

AKEW's own numbers say the format is the bottleneck: the same facts score
93-99% as clean triples and 2.25-4.78% extracted from prose. This wiki is
0.24% triple-shaped. So "prose does not become a rank-one edit" may be a fact
about the prose rather than about the method — and nobody has tried changing
the prose.

**The naive version is already known to fail.** Having a model extract
triples from prose IS the automatic extraction AKEW measured in the single
digits. The version worth testing is high-precision triples — curated, or
generated behind a verification gate that discards anything not exactly
grounded in its source sentence. That converts the question from "does the
method work" into "what does a usable representation cost", which is a
different and better question.

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
