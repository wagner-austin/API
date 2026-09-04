---
title: A cartridge halves the surprise on a fact and still cannot pick it out of a line-up
tags: [ml, model-trainer, cartridges, measurement, evaluation]
related:
  - "[[model-trainer-composition-ceiling]]"
  - "[[monorepo-discipline]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa.py
  - services/Model-Trainer/src/model_trainer/core/services/model/corpus_cloze.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/contracts/cloze.py
  - services/Model-Trainer/src/model_trainer/core/services/model/control_arms.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa.py": d6d94d33de70b0c09734a951773c3c38965da42c
  "services/Model-Trainer/src/model_trainer/core/services/model/corpus_cloze.py": e14dc465701df5af4e77c4facd8103aff35ab8ce
  "services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py": 67f057b3af4354a571abcf3d246951a122651c53
  "services/Model-Trainer/src/model_trainer/core/contracts/cloze.py": c4e1e0ebaefc2fbb47a123d50d4c68ad4fa242ca
  "services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py": 9aafe105357bb6ebb7462afbc8221aa16a0facd6
  "services/Model-Trainer/src/model_trainer/core/services/model/control_arms.py": d8d1e89ba5c1920464a501048a028d9b24b97acc
provenance:
  - "measured 2026-09-04 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1"
  - "gpt2 (12 layers, 12 heads, 1024 positions), 12 me-wiki pages carrying visibility: public, 128 slots, seeds 7/8/9, determinism pinned"
  - "24 items generated mechanically from held-out windows; no item hand-written"
  - "board task 1fc5afed-89a7-400e-b79e-378f322711c7 carries the full trail"
fact_checked: "2026-09-04"
confidence: high
hubs: [services]
---

# A cartridge halves the surprise on a fact and still cannot pick it out of a line-up

Every earlier cartridge number in this repository was a held-out **loss**, and
`core/contracts/cloze.py` already said why that is not enough: "a model can
memorise text word-by-word and still fail every question about it." The
question-set arm asks the other question, and the two instruments disagree.

| arm | accuracy | against base |
|---|---|---|
| base | 0.5417 | — |
| cartridge | 0.5833 | gain +0.0972, seed spread **0.1250**, p = 1.00 |
| retrieval (oracle) | 1.0000 | gain +0.4583, p = 0.00098 |
| answer-token NLL | 18.4569 → **10.6843** | gain +7.3790, spread 0.7788, better on 19/24, p = 0.0066 |

The cartridge nearly halves the model's surprise at the correct term — a gain
nine times its own seed spread — while its **accuracy** gain sits *inside*
that spread and is not a finding. It raises the likelihood of corpus
vocabulary generally without sharpening the choice *between* corpus terms.

Oracle retrieval answers all 24. That is the honest thing to lose to: it is
handed the training sentences that contain the answer, so it is an upper bound
on any real retrieval pipeline rather than a strawman. At this scale
**retrieval wins outright on accuracy**.

## The distractor policy moved the answer more than the model did

The first item set repeated one distractor triple across nearly every item. On
it the base model scored 0.2500 — chance exactly — and the cartridge 0.5417 at
p = 0.006: a clean, significant, publishable-looking effect. Rotating
distractors per item moved the base to 0.5417 and the effect **vanished**
(p = 1.00). Same corpus, same items, same models, opposite conclusions.

Multiple-choice accuracy here is dominated by which wrong candidates are
offered, not by corpus knowledge. That is why `answer_nll` exists and why its
numbers lead: scoring the answer's own tokens has no policy knob to be
sensitive to. `distractor_count` is part of the run label, so two records
built under different policies cannot be differenced.

## What keeps the items honest

Items are **generated**, never written. A hand-written question set is a place
for a fact that is not in the corpus to enter the measurement, with no way to
tell from the numbers. So a term is chosen out of the corpus, the sentence it
occurs in is taken verbatim, and the term is blanked.

The memorisation trap is avoided by the split rather than by hope: the
cartridge trains on the **training** windows, items are built from the
**held-out** windows, and a term qualifies only if it *also* occurs in the
training windows. So the answer is learnable from the text the cartridge
trained on, and is tested in a sentence absent from that text.

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

Both benchmarks now require `--controls` (`none` / `split-k` / `attention` /
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
could use it, and this page exists because those came apart.
