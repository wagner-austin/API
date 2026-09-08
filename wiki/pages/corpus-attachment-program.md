---
title: Attaching a corpus to a model — four arms, three answered, and the one nobody tested
tags: [ml, model-trainer, cartridges, retrieval, model-editing, research-program]
related:
  - "[[model-trainer-composition-ceiling]]"
  - "[[model-trainer-cartridge-question-set]]"
  - "[[model-trainer-noise-floor-is-a-range]]"
  - "[[model-trainer-companioned-training-recipe]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_retrieval.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/apply.py
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py": 2a14e9edc408d27d6324bb15ef4dcdb6af259a72
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_retrieval.py": 6fa2350991c7ed262a342326ba6846b4b6afa519
  "services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py": 8f2fd9d682790c501b7255cba8c6187a33a7b81c
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/apply.py": 411c47f975fad5b700bcc6948da496f9af9c4c73
provenance:
  - "QA arms measured 2026-09-07 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1, --controls none"
  - "record qa-record-bm25.json, plan gpt2-wiki-qa, 24 items over the 12 me-wiki pages carrying visibility: public"
  - "board tasks d3639e09 (weight injection), d3742672 (steering vectors), 1fc5afed (cartridges), 74dd514e (persona adapter, never started)"
  - "AKEW figures from wiki page wu-2024-akew-editing-in-the-wild in the personal wiki, read page by page"
fact_checked: "2026-09-08"
confidence: high
hubs: [services]
---

# Attaching a corpus to a model

One question has been asked four ways over five days: **can a model be made
to answer questions about a private corpus without retrieving at query
time?** The arms were run as separate board tasks by separate sessions and
each produced its own page, so the program has never been written down as
one thing. A reader arriving at any single page cannot tell which of four
attempts they are looking at, or that three of them are already answered.

This page is the index. It carries no measurement of its own.

## The arms, and where each stands

| arm | mechanism | verdict |
|---|---|---|
| Weight injection | rank-one edit into the parameters | **No, on this corpus** — bounded by ingest, not algebra |
| Steering vectors | add a direction to the residual stream | **No** — composition degrades to nothing |
| Cartridges | train a KV prefix, serve it ahead of the query | **No, at this scale** — loses to keyword search |
| Persona adapter | not started | untested |

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

## Cartridges: the arm that got a full measurement, and lost to BM25

The only arm carried to a three-way comparison on real data. Over 24 held-out
questions about the 12 public me-wiki pages, chance 0.25:

| arm | accuracy | serve latency |
|---|---|---|
| base model alone | 0.5417 | 62.98 ms/item |
| base + cartridge | 0.6389 (gain +0.097, **spread 0.125**) | 75.26 ms/item |
| base + BM25 retrieval | **0.8333** (gain +0.292, **p = 0.0156**) | **69.79 ms/item** |
| base + oracle retrieval | 1.0000 (knows the answer) | 70.90 ms/item |

The cartridge loses on BOTH axes to a lexical retriever with no model, no
embeddings and no weights, whose per-query search costs 1.84 ms/item. Its
accuracy gain does not separate from its own seed spread; BM25's does.

**The cartridge is not doing nothing**, and that is the interesting part. On
the same corpus it nearly halves the model's surprise at the correct answer —
summed NLL 18.46 to 10.68, better on 19 of 24 items, p = 0.0066. It raises
the likelihood of corpus vocabulary generally without sharpening the choice
between corpus terms, which is what the question actually asks for. Putting
the sentence in the window sharpens it.

## What this verdict does NOT cover

Stated because the result is narrow and reads broader than it is.

1. **It is not the published Cartridges system.** Eyuboglu et al. need Llama
   or Qwen3, a two-stage pipeline with a synthesis server, and wandb for
   artifact loading. What was measured is this repo's own simpler thing:
   direct context distillation over corpus windows, on a 124M model.
2. **It was measured outside the regime cartridges are for.** They are a
   CONTEXT COMPRESSION technique — the claimed win is serving a very long
   context cheaply. The window here is 896 tokens with evidence truncated to
   fit. At that size there is nothing to compress and retrieval is trivially
   cheap.
3. **Only lexical retrieval was tested.** A dense retriever costs real
   milliseconds per query, which is the one regime where the cartridge's
   latency story could survive.
4. **One model size.** gpt2, 124M. No ladder was run.

## The untested lever, and why it is the interesting one

Every arm above tested a METHOD. None tested the CORPUS REPRESENTATION.

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
