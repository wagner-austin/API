---
title: The cartridge's line-up gain at 124M lands on the instrument's own floor, and the run that first denied it skipped a quarter of the corpus
tags: [ml, model-trainer, cartridges, measurement, evaluation, retraction]
related:
  - "[[corpus-attachment-program]]"
  - "[[model-trainer-composition-ceiling]]"
  - "[[monorepo-discipline]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_arms.py
  - services/Model-Trainer/src/model_trainer/core/services/model/corpus_cloze.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_question_set.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cloze/identity.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/contracts/cloze.py
  - services/Model-Trainer/src/model_trainer/core/services/model/control_arms.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa.py": 3c262a7efce1dc2b3b9d25ef4787f03d420421a2
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_arms.py": e26a5e99699ac3f7b389df81b02fc04b44ef1637
  "services/Model-Trainer/src/model_trainer/core/services/model/corpus_cloze.py": 509132ebe55fc8717e973b3183bd3018d9b0ec58
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_question_set.py": 17f3a1278744c898b05c3511b8b89a51aa498fd6
  "services/Model-Trainer/src/model_trainer/core/services/model/cloze/identity.py": b128cc4752df995e95da34bf92d61e2c1bbc6b4c
  "services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py": 2f3ad497d7dcea1cb378720573de7243d6f85f01
  "services/Model-Trainer/src/model_trainer/core/contracts/cloze.py": c4e1e0ebaefc2fbb47a123d50d4c68ad4fa242ca
  "services/Model-Trainer/src/model_trainer/core/services/model/control_arms.py": d8d1e89ba5c1920464a501048a028d9b24b97acc
  "services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py": 8f2fd9d682790c501b7255cba8c6187a33a7b81c
provenance:
  - "CURRENT: record qa-svc-gpt2.json, measured 2026-09-08 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1, --controls none"
  - "gpt2 (12 layers, 12 heads, 1024 positions), 12 me-wiki pages carrying visibility: public, 128 slots, seeds 7/8/9, determinism pinned, corpus digest e2f23c635583"
  - "32 items generated mechanically from held-out windows, one per document; no item hand-written"
  - "RETRACTED: the 2026-09-04 run (qa-record.json, 24 items) and every number this page carried until 2026-09-09; see the retraction section for what was wrong and which way it moved"
  - "board task 1fc5afed-89a7-400e-b79e-378f322711c7 carries the original trail; ac5f88cb and a8f799c5 carry the corrected ladder"
  - "SCOPE CORRECTION 2026-09-09, later the same day: the accuracy gain was reported against its own SEED SPREAD, which measures wobble between initialisations and not whether 32 paired items can resolve two arms. The arms answer the same 32 items, so McNemar exact applies -- confirmed by this page's own p-values, where BM25's +10 items reports 0.0020 = 2*(0.5)^10 and the oracle's +17 items reports 0.0000153 = 2*(0.5)^17. Its floor is 6 items: 2*(0.5)^5 = 0.063 cannot reject under any arrangement of the pairs. The cartridge gain is 6/5/5 items across seeds -- one on the boundary, two under it -- and 'loses to every retriever at this scale' (cartridge-BM25 = 4.7 items) is struck for the same reason. The answer-NLL gain is continuous and not governed by this floor."
fact_checked: "2026-09-09"
confidence: medium
hubs: [services]
---

# The cartridge's line-up gain at 124M lands on the instrument's own floor

Every earlier cartridge number in this repository was a held-out **loss**, and
`core/contracts/cloze.py` already said why that is not enough: "a model can
memorise text word-by-word and still fail every question about it." The
question-set arm asks the other question. It first answered that a cartridge
halves the surprise on a fact and cannot choose between corpus terms; the
instrument behind that answer was defective, and the way it was defective is
the more useful finding.

**Two corrections have now been applied to this page and they point opposite
ways, so read the order.** The first (2026-09-08) fixed the instrument and
moved the accuracy gain up, from +0.0972 to +0.1667. The second
(2026-09-09, below) fixed how that gain was JUDGED, and moved the conclusion
back down: +0.1667 of 32 items is 6 / 5 / 5 items across the three seeds,
against a paired-test floor of 6, so **at 124M the accuracy result is at the
edge of what this question set can resolve and is not a finding on its own**.
What is NOT at the edge is the answer-token surprise: +7.8051 NLL against a
seed spread of 1.1491 is a continuous measurement, not a count of discordant
items, and the McNemar floor does not govern it.

Every number below is an observation of record `qa-svc-gpt2.json`, emitted by
`measure_qa_plan` under the names it gives them.[^rec]

| arm | accuracy on gpt2 | against base |
|---|---|---|
| chance | 0.2500 | — |
| base | 0.4375 | — |
| **cartridge** | **0.6042** (mean of three seeds) | gain **+0.1667**, seed spread **0.0313** |
| BM25 retrieval | 0.7500 | gain +0.3125, p = 0.0020 |
| dense retrieval | 0.7188 | — |
| RRF-fused | 0.7813 | — |
| retrieval (oracle) | 0.9688 | gain +0.5313, p = 0.0000153 |
| answer-token NLL | — | gain **+7.8051**, spread 1.1491 |

**THIS TABLE IS NO LONGER THE FULL SET OF ARMS, as of 2026-09-09 (`643757bd`).**
A long-context arm now runs beside these — the corpus handed to the model
whole, with no retrieval at all — and it is absent here because it did not
exist when these numbers were taken. Every comparison above is therefore
cartridge-versus-*searching*, and says nothing about cartridge-versus-*reading*,
which is the competitor a reader is most likely to assume was tested.

It is not a matter of re-running to fill the row. At this rung the corpus is
15,602 tokens against an 896-token budget, so the arm would carry about 6% of
the text and beating it would mean almost nothing; the new arm reports
`long_context_corpus_fraction` precisely so that number cannot be read without
its coverage. The row becomes meaningful at a corpus and window where that
fraction approaches 1.

The cartridge's accuracy gain is **five times its own seed spread** (0.1667
against 0.0313, per-seed gains 0.1875 / 0.1563 / 0.1563), and it moves in the
same direction as the surprise gain rather than against it. Both instruments
now say the same thing: reading the corpus made the model both less surprised
by the answer and better at choosing it.

**The seed spread is the wrong yardstick for that claim, and this page said
so on 2026-09-09 after saying the opposite.** Seed spread measures how much
the arm wobbles between initialisations. It says nothing about whether 32
paired items can resolve the difference between two arms — and that is the
question, because every arm here answers the SAME 32 items, which makes the
comparison paired and McNemar's exact test the one that applies.

**Read the p-values already in the table above and the instrument names
itself.** BM25's +0.3125 is 10 items of 32 and reports p = 0.0020; the
oracle's +0.5313 is 17 items and reports p = 0.0000153. Both are exactly
2·(0.5)^*k* for *k* = 10 and *k* = 17 — the two-sided exact probability of
*k* discordant pairs all falling one way. So the test behind this column is
McNemar exact, and its floor follows immediately: **a net difference of 5
items cannot reach p ≤ 0.05 under ANY arrangement of the pairs** (2·0.5^5 =
0.063), while 6 items can (0.031).

**That floor belongs to the EXACT test, and here the distinction is not
academic.** Under **mid-p** a 5:0 split gives 0.031 and rejects, putting that
floor at 5 — under which all three of the cartridge's seeds would clear
rather than one. **The exact floor is the one that applies**, because exact
is what this instrument computes and reports; a page does not get to pick the
variant that flatters it after the fact. Recorded because the temptation is
obvious and because `code-style` legitimately reports mid-p, so the same
count means different things in the two lanes and a floor quoted without its
variant will be applied to the wrong one.

The cartridge's gain is 6 / 5 / 5 items across the three seeds. **One seed
of three sits on the boundary and two sit under it**, which is why no p is
reported beside it while every retrieval row has one. The honest reading is
that this instrument is at the edge of being able to see this effect at
124M, not that it has seen it.

~~It still **loses to every retriever at this scale**, including the
weakest.~~ **STRUCK by the same arithmetic on 2026-09-09.** Cartridge−BM25
here is −0.1458, which is 4.7 items — under the floor. The cartridge was not
shown to lose to BM25 at 124M; the two are unresolvable on this question set.
Where the ladder sits at larger scales is on [[corpus-attachment-program]],
whose own retrieval headline was retracted the same day for the same reason.

## What was retracted, and which way it moved

The 2026-09-04 record asked **24** questions where the corrected instrument
asks **32**, over the same twelve pages, the same plan and the same corpus
digest. Two defects, both in item construction, both found by running the
measurement rather than by a failing test:

- **Held-out windows were chosen across the corpus rather than per document**
  (fixed in `9ba9dfb6`). The stride walked one concatenated sequence, so three
  of the twelve pages contributed training text and were never examined. The
  cartridge was graded on nine twelfths of what it read.
- **Terms qualified from raw document text** (fixed in `eb73abf8`, and exposed
  by the first fix). Nine of ninety-four terms existed only inside markdown
  table rows and URLs, which `sentences` strips, so an item could ask about a
  term the prose never contains. `corpus_cloze` now qualifies a term against
  the sentences of the training text — `terms_in(" ".join(sentences(training_text)))`
  — which is the same text the items are drawn from.

What that did to the numbers is the part worth keeping, reading the two
records side by side.[^pair]

| | retracted (24 items) | corrected (32 items) |
|---|---|---|
| base accuracy | 0.5417 | 0.4375 |
| cartridge gain | +0.0972 | **+0.1667** |
| cartridge seed spread | 0.1250 | **0.0313** |
| oracle accuracy | 1.0000 | 0.9688 |
| answer-NLL gain | +7.3790 | +7.8051 |

The spread fell by a factor of four. **The old verdict — a gain inside its own
noise — was a statement about the noise, and the noise was the instrument.**
An items list assembled from a quarter fewer documents, some of them asking
about terms that appear in no sentence, varies more between seeds for reasons
that have nothing to do with what the cartridge learned.

The retraction is recorded rather than the numbers deleted, because the
reasoning that produced the old verdict was sound: a gain inside its seed
spread genuinely is not a finding, and refusing to report it was right on the
evidence then available. Only the evidence was defective.

**And the old verdict was closer to right than this section credited it,
which is the third correction and the least comfortable one.** Fixing the
instrument moved the gain from 2.3 items of 24 to 5.3 of 32 — larger, and
still under the 6-item floor a 32-item paired test imposes. So "not a
finding" survived the very correction that was supposed to overturn it. The
sentence above says the old verdict fell to better evidence; what actually
happened is that better evidence moved the number toward the threshold
without carrying it across, and this page declared victory at the moment the
number moved rather than at the moment it cleared anything.

That is the transferable part, and it is not about cartridges. **A
correction that moves a number in the direction you wanted is the single
easiest place to stop checking.** The seed-spread comparison was available,
flattering, and already in the record; the floor had to be computed, was
not, and would have taken one line.

## The two runs were indistinguishable, and that is now fixed

`qa-record.json` and `qa-svc-gpt2.json` measure different question sets and
carried the **same** experiment, the **same** label — plan fields plus a
corpus digest — the same fingerprint, and the same empty payload digest. Every
identity `platform_core.run_record` has said they were the same measurement,
so `compare_runs` would subtract one from the other and report a change of
answer where there was a change of question.

The label cannot carry this and should not try: it names what was
**requested**, while the item set is what the code **derived**, and the two
move independently by construction. `cloze/identity.question_set_digest`
digests every item's full content in build order, and `qa_run_record` now
writes it into `payload_digest`, which the record shape has always had for
exactly this. The digest covers the **distractors** as well as the ids, for
the reason the next section gives.

## The distractor policy moved the answer more than the model did

The first item set repeated one distractor triple across nearly every item. On
it the base model scored 0.2500 — chance exactly — and the cartridge 0.5417 at
p = 0.006: a clean, significant, publishable-looking effect. Rotating
distractors per item moved the base to 0.5417 and the effect **vanished**.
Same corpus, same items, same models, opposite conclusions — and every
`item_id` was identical across the two, which is why a digest over identifiers
alone would not separate them.

Multiple-choice accuracy here is dominated by which wrong candidates are
offered, not by corpus knowledge. That is why `answer_nll` exists and why it
is reported beside the accuracy: scoring the answer's own tokens has no policy
knob to be sensitive to. `distractor_count` is part of the run label, so two
records built under different **counts** cannot be differenced; the digest is
what separates two built under different **choices**.

## What keeps the items honest

Items are **generated**, never written. A hand-written question set is a place
for a fact that is not in the corpus to enter the measurement, with no way to
tell from the numbers. So a term is chosen out of the corpus, the sentence it
occurs in is taken verbatim, and the term is blanked.

The memorisation trap is avoided by the split rather than by hope: the
cartridge trains on the **training** windows, items are built from the
**held-out** windows, and a term qualifies only if it *also* occurs in the
training text. So the answer is learnable from the text the cartridge trained
on, and is tested in a sentence absent from that text. `build_question_set`
now takes that stride **per document**, which is the fix above.

## Four things that were silently wrong

Each was found by running the measurement, and none surfaced as an error:

- **Token counts are not additive.** Locating the answer span as
  `len(encode(before))` is wrong wherever byte-pair encoding merges across the
  join — appending the answer `AI` to one item's prefix left the id count
  *unchanged* at 22. `answer_span` locates it by agreement with both contexts
  instead, which also covers the merged boundary token.
- **A causal model cannot score a sequence's first token.** An item whose
  template begins with the blank indexed `logits[0, -1 : ...]`, which Python
  reads as the *last* position and silently returns an empty selection.
- **Scoring inherited the caller's train/eval mode**, and training leaves the
  base in train mode, so dropout made two calls on one input differ.
- **`item_id` of document-plus-term repeats**, because a page names its own
  subject in several sentences. Arms are paired by id, so duplicates collapse
  in the lookup and pair one arm's outcome against a different question's.

## The numbers above were measured under one kernel posture

Both benchmarks require `--controls` (`none` / `split-k` / `attention` /
`both`, from `core/services/model/control_arms.py`), and the fingerprint
records which controls were applied.[^ctl] **Everything on this page was
measured under `none`** — the arm that applies neither control, which is what
the command was hardcoded to before the flag existed.

That is not a footnote. On the loss benchmark, running the same plan under all
four arms moved the noise floor from 0.0123 (`split-k`) to 0.0326 (`both`) and
flipped two of four separation verdicts, while the forward-only arm — no
optimiser, pure kernel arithmetic — moved by up to 3.07e-07 depending on which
control was applied.[^arms] No arm is the canonical one. A question-set record
and a loss record measured under different arms are different configurations,
which is why the posture is in the fingerprint rather than in a comment.

[^rec]: `services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py`
    § `measure_qa_plan`, which names every observation in the table, and
    `core/services/model/cartridge_qa_report.py` § `latency_observations` for
    the cost names it emits alongside them. The record itself is
    `qa-svc-gpt2.json`, listed under `provenance:` with the card, driver and
    determinism setting it was measured under.

[^pair]: The retracted column is record `qa-record.json` and the corrected one
    `qa-svc-gpt2.json`, both listed under `provenance:`. The two item-set
    defects are commits `9ba9dfb6` and `eb73abf8`; the split now lives in
    `core/services/model/cartridge_question_set.py` § `build_question_set` and
    the term qualification in `core/services/model/corpus_cloze.py`.

[^ctl]: `services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py`,
    `qa_run_record` and `main`; the arm table is
    `core/services/model/control_arms.py` § `CONTROL_ARMS`.

[^arms]: `services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py`
    § `cartridge_run_record`, whose docstring records the cross-card
    measurement the flag was added for. The four arms were run on a 3090 Ti on
    2026-09-04; board task `1fc5afed-89a7-400e-b79e-378f322711c7` carries the
    full table, and commit `f297331e` carries the code.

## Reading this beside the loss numbers

The loss arms and this one are deliberately different experiments
(`cartridge-capacity-and-composition` versus `cartridge-question-set`), so the
comparability layer refuses to subtract their records. That refusal is the
point: one says how surprising the prose was, the other whether the model
could use it. This page existed because those two came apart — and the arm
that made them come apart turned out to be measuring nine twelfths of a
corpus.
