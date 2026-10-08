---
title: "The question-set instrument's axes existed as plans the power gate refused, and its 7B rung would have run out of memory"
tags: [ml, model-trainer, cartridges, measurement, hpc3, power]
related:
  - "[[model-trainer-cartridge-question-set]]"
  - "[[a-claim-four-times-below-its-instruments-floor]]"
  - "[[corpus-attachment-program]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_axes.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_plans.py
  - services/Model-Trainer/src/model_trainer/core/contracts/qa_plan.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_power.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_question_set.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_solo_seeds.py
  - tools/hpc3/specs/abl-image.json
  - tools/hpc3/runs/qa-corpus-wiki-full-ee6e6d12-stage.json
  - tools/hpc3/runs/qa-corpus-wiki-full-ee6e6d12-digests.txt
  - tools/hpc3/runs/qa-full-wiki-gpt2-v52.json
  - tools/hpc3/runs/qa-full-wiki-v56.json
  - tools/hpc3/runs/qa-full-wiki-pythia-6.9b-a30-v56.json
  - tools/hpc3/runs/hpc3-cartridge-qa.json
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_axes.py": 377f81f34295d45103954ec18b33764e530332a9
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_plans.py": 56c017ec9582b3271d6a05c164ad3c5f72b07c20
  "services/Model-Trainer/src/model_trainer/core/contracts/qa_plan.py": 11072ccc6494a0db30b6b4ab52e2989cdd446f76
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_power.py": 7a1566221a09ea1e36da9ff45c64c865323b059b
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_question_set.py": 0bfe2ff9a9e2e2bd310898af8e5a980d0514bb01
  "services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py": 0789465e4cc23a2ee2cc2ca735e0ee7e0697c096
  "services/Model-Trainer/src/model_trainer/cli/cartridge_solo_seeds.py": f77d37acd17b88451fc0264ec29417931035ebcc
  "tools/hpc3/specs/abl-image.json": b9d8948d2bd4ec6a6f51b001d9e419e556b23d15
  "tools/hpc3/runs/qa-corpus-wiki-full-ee6e6d12-stage.json": fb3cb4a487382c897508c8b1b1f4890c22babd4a
  "tools/hpc3/runs/qa-corpus-wiki-full-ee6e6d12-digests.txt": 27c0615da115d3c684e8d6e6ad96a0965589f03a
  "tools/hpc3/runs/qa-full-wiki-gpt2-v52.json": fa3fa9a9a37c9530a9b5e7648c0ba329f1feb913
  "tools/hpc3/runs/qa-full-wiki-v56.json": 2fe0e5e562b2da10a719a27f223b854fe61d80b0
  "tools/hpc3/runs/qa-full-wiki-pythia-6.9b-a30-v56.json": 56724d50894f22738093e4560e1e7221cc37aad0
  "tools/hpc3/runs/hpc3-cartridge-qa.json": 86660ee5bed1f6ec195f74c846b55fa8b8c5d0cb
provenance:
  - "re-read 2026-10-08 against API 38cbd5011 for MCPs board task 3d71a8e1: a1492d90e (board task 83c25b86) added the trait, repair, shard and style symbols to specs/abl-image.json and a 55th smoke command checking every trait plan's seed gate, while its git_commit still names v56's 0e9f177f until an image is built from a commit that carries them. So the spec at HEAD has one more smoke check than the 54 the v56 build below ran; the fiftieth entry [^image] cites is byte-identical to the pinned one, and the v56 image this page's runs use is unchanged"
  - "item counts measured 2026-10-04 on austinpc by running cartridge_question_set.build_question_set over ~/PROJECTS/wiki/pages at wiki commit ee6e6d12 (885 pages, `git status --porcelain -- pages` empty): gpt2 tokenizer, 1,634,715 tokens, 3,753 items at window 128, 3,827 at 256, 3,278 at 512; EleutherAI/pythia-6.9b tokenizer, 1,589,449 tokens, 3,793 items at window 256; corpus digest a756664dbb4e"
  - "the same builder over ~/PROJECTS/API/wiki/pages (38 pages, 104,524 gpt2 tokens) yields 275 items at window 256, against plans whose max_items was 240"
  - "HPC3 ledger rows for the earlier full-wiki submissions: 55898551 FAILED in 332s (image v50 lacked gpt2-full-wiki-qa, KeyError in the job's .err), 55901956 FAILED in 1454s (torch.OutOfMemoryError in the dense embedder's BERT forward, 22.86 GiB requested), 55914185 PREEMPTED at 1341s on hpc3-gpu-l54-04 while still building items"
  - "/pub/wagnera3/hf/hub on 2026-10-04 holds gpt2, gpt2-medium, gpt2-large, gpt2-xl, EleutherAI/pythia-6.9b and thenlper/gte-base, so every rung runs with HF_HUB_OFFLINE=1"
  - "image v55: build job 57747902 on free, from commit 04d46ff9 (spec commit c9b25ef0); the five first-party wheels' sha256 matched on both sides of the upload; its run documents were cancelled with no output"
  - "image v56: build job 57748479 COMPLETED on free in 48m15s, 54 of 54 smoke checks, /pub/wagnera3/images/v56/abl sha256 890e444560d4c5caa43860bfd3235849471eadc01acbe77b968ae30e6aa7d5be; its GIT_COMMIT file read 0e9f177f56d5e6895b6a148f6ae86901d18d4e89 over ssh on 2026-10-05"
  - "hpc3-preflight on 2026-10-05 against the A100 default: GPU_MODEL_EXHAUSTED, free-gpu A100 0/2 free, A30 10/28, V100 5/12; on A30 the array was admitted at 156.00 projected GPU-hours and the 7B run at 36.00, against the project's 200.00"
  - "submitted 2026-10-05 by hpc3-sweep and hpc3-submit: job array 57783112_[0-12] (qa-full-wiki-v56) and job 57783131 (qa-full-wiki-pythia-6.9b-a30-v56), both on free-gpu"
fact_checked: "2026-10-08"
confidence: medium
hubs: [services]
---

# The question-set instrument's axes existed as plans the power gate refused, and its 7B rung would have run out of memory

`cartridge_qa_benchmark` scores a cartridge against retrieval on a
multiple-choice question set built from held-out windows
([[model-trainer-cartridge-question-set]]). Its headline was withdrawn for
resting on 32 items ([[a-claim-four-times-below-its-instruments-floor]]), and
the repair was a gate that refuses a run whose realised item count cannot
resolve the effect its plan declares.[^gate] This page records what that gate
did to the axes added beside it, and what replaced them.

## Plans that could not produce a number

On 2026-09-09 the table gained a rung above the GPT-2 family
(`pythia-6.9b-api-wiki-qa`) and the first capacity axis
(`gpt2-large-api-wiki-qa-slots-{32,64,128,256}`), both on the api-codebase
wiki. Every plan declares 0.02 as its smallest effect of interest, which
needs 250 items under mid-p McNemar at alpha 0.05. Those plans capped the
question set at 240. The corpus yielded 237 items that day and 275 on
2026-10-04, so the cap bound either way, and the gate refused every one of
them before a model loaded.[^why]

So the ladder reached past its own ~774M crossing, and the capacity claim was
measured over a curve, only in the plan file. Neither could produce a
record.[^why]

## One base, one field per cell

The replacement lives on the whole personal wiki, which yields 3,278 to
3,827 items per cell, more than thirteen times the gate. Every cell is built
by `full_wiki_family` as the base plan `gpt2-full-wiki-qa` with one field
replaced:[^axes]

| axis | field it owns | cells |
|---|---|---|
| scale | `model_id`, and the `precision_selector` that base forces | gpt2-medium, gpt2-large, gpt2-xl, pythia-6.9b |
| capacity | `num_slots`, on gpt2-large, all at `max_seq_len` 768 | 32, 64, 128, 256 |
| window | `window` | 128, 512 (the base is 256) |
| epochs | `epochs` | 3, 6, 24 (the base is 12) |

Building rather than copying is the point. The written plans were nineteen-field
literals duplicated from a neighbour, and only a test that spread the base
could say a rung moved one field. A built cell cannot move a field its axis
does not own, and a field added to `QaPlan` later reaches every cell through
the base. `merged_plan_tables` refuses a name defined twice rather than
letting a dict merge keep whichever table came last.[^axes]

**The capacity cells share one 768-token budget.** Sizing each to
`1024 - num_slots` would give the smallest cartridge the most retrieved
evidence, and the axis would move two things in opposite directions at
once.[^budget]

**The window axis changes the questions.** Items come from held-out windows,
so a different `window` builds a different question set.[^items] Raw accuracy
across window cells is therefore not a curve. Each cell's within-cell gap
(cartridge against BM25, against long context) is paired on that cell's own
items, and that gap is what the axis compares. The slot and epoch cells keep
the base's items, so their accuracies compare directly.

## The 7B rung would have run out of memory

The benchmark handed the model loader `None` for every base until
2026-10-04, so pythia-6.9b would have loaded in fp32: 27.6GB of weights
against the 24GB A30 its rung was declared for. `QaPlan` now carries
`precision_selector`, and `measure_qa_plan` resolves it through
`resolve_precision` before reading the corpus, the same field and resolver
the companion sweeps use for the same base.[^prec] The 7B rung declares
`stored-bf16` (13.8GB), and its label carries `-storedbf16`. Every gpt2 plan
resolves to the policy fp32 load with an empty token, so their labels did not
change.

## The cluster path

The corpus is staged with `hpc3-stage` to
`/pub/wagnera3/mi/cartridge/corpus-wiki-full-ee6e6d12`. Each file's sha256 is
checked on both sides and against a digest listing read from the wiki tree
itself, not from the snapshot that was staged.[^stage] Image v56 is built from
`0e9f177f`, the published commit that declares every cell by a literal name.
Its fiftieth smoke check asserts the cells and the bf16 resolution inside the
image.[^image] v55, built from an unpublished worktree commit whose cell names
were assembled by an f-string, was cancelled before it produced anything.

The family runs as two documents, because a sweep's members share one GPU
profile. The 13 gpt2-family cells are one job array and the 7B rung is its own
2,160-minute run, all on A30s.[^runs] The project's default card is the A100,
but `free-gpu` carried two and neither was free on 2026-10-05, against 10 of
28 A30s, so `hpc3-preflight` refused the array. The 7B rung needs an A30 for
its weights anyway, so the whole ladder now times on one card model.

Three earlier full-wiki submissions never produced a number. The first image
predated the plan, the dense embedder ran out of memory on an unbatched
corpus, and the third was preempted while it was still building items. The
third's run document named a corpus copied by hand, with no stage manifest
behind it.[^v52]

[^gate]: `services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_power.py` § `require_resolvable_question_set`, called from `services/Model-Trainer/src/model_trainer/cli/cartridge_qa_benchmark.py` § `measure_qa_plan` against `len(items)`, never `max_items`.
[^axes]: `services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_axes.py` § `full_wiki_family`, `scale_rungs`, `slot_cells`, `window_cells`, `epoch_cells` and `merged_plan_tables`; the table that merges them is `services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_plans.py` § `QA_PLANS`.
[^items]: `services/Model-Trainer/src/model_trainer/core/services/model/cartridge_question_set.py` § `build_question_set`, which reads `plan["window"]` to find each held-out window.
[^prec]: `services/Model-Trainer/src/model_trainer/core/contracts/qa_plan.py` § `QaPlan.precision_selector`; `services/Model-Trainer/src/model_trainer/cli/cartridge_solo_seeds.py` § `resolve_precision`.
[^stage]: `tools/hpc3/runs/qa-corpus-wiki-full-ee6e6d12-stage.json` (destination, 885 files, provenance block) and `tools/hpc3/runs/qa-corpus-wiki-full-ee6e6d12-digests.txt` (the record `--expect-from` holds it to).
[^image]: `tools/hpc3/specs/abl-image.json`, `git_commit` and the fiftieth `smoke_commands` entry.
[^runs]: `tools/hpc3/runs/qa-full-wiki-v56.json` (13 members, `gpu` A30, `throttle` 6) and `tools/hpc3/runs/qa-full-wiki-pythia-6.9b-a30-v56.json` (`minutes` 2160, `gpu_pinned_because`); the project defaults they override are in `tools/hpc3/runs/hpc3-cartridge-qa.json`. The preflight reading and the job ids are under `provenance:`.
[^why]: `services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_axes.py`, the module docstring's "WHY THE FULL WIKI AND NOT THE API-CODEBASE WIKI" paragraph, which records the 237- and 275-item yields against the 240 cap; the item counts themselves are under `provenance:`.
[^budget]: `services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_axes.py` § `SLOT_AXIS_MAX_SEQ_LEN`, 768, which is 1024 less the largest of `SLOT_CELLS`.
[^v52]: `tools/hpc3/runs/qa-full-wiki-gpt2-v52.json`, whose `--corpus` is `/pub/wagnera3/mi/cartridge/corpus-me-wiki-full`; no `*-stage.json` under `tools/hpc3/runs/` names that destination. The three job ids and their outcomes are under `provenance:`.
