---
title: Model-Trainer's cloze scoring path — one scorer, four doors, and a digest over decisions not scores
tags: [services, model-trainer, cloze, evaluation, provenance, floor, determinism]
related:
  - "[[model-trainer-run-record-provenance]]"
  - "[[model-trainer-known-answer-registry]]"
  - "[[model-trainer-finetune-strategy-seam]]"
  - "[[model-trainer-service-architecture]]"
  - "[[a-claim-four-times-below-its-instruments-floor]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/core/services/model/cloze/score.py
  - services/Model-Trainer/src/model_trainer/core/contracts/cloze.py
  - services/Model-Trainer/src/model_trainer/cli/score_baseline.py
  - services/Model-Trainer/src/model_trainer/cli/score_run.py
  - services/Model-Trainer/src/model_trainer/worker/baseline_cloze_job.py
  - services/Model-Trainer/src/model_trainer/worker/cloze_job.py
  - docs/RESEARCH.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/core/services/model/cloze/score.py": 8c6dc9f6e0575ef110b3b7ddccb11ce778591fe5
  "services/Model-Trainer/src/model_trainer/core/contracts/cloze.py": c4e1e0ebaefc2fbb47a123d50d4c68ad4fa242ca
  "services/Model-Trainer/src/model_trainer/cli/score_baseline.py": ce1944988c83f47c08378aa94536f5a2a409b098
  "services/Model-Trainer/src/model_trainer/cli/score_run.py": 7da2bc858f6ed213b0e19d54deabcd4915170697
  "services/Model-Trainer/src/model_trainer/worker/baseline_cloze_job.py": 4d2eedae3cb07a886317d9b1618bd005f7b49c43
  "services/Model-Trainer/src/model_trainer/worker/cloze_job.py": 5d914f43283c0c04cea638cfa73a77153f5a53d9
  "docs/RESEARCH.md": fa186d5df657233450dfac204982ad9eff07173d
provenance:
  - "read from code at API commit 2e0511ee7 on 2026-10-05"
  - "the scoring rule itself (total-NLL substitution, strict-minimum ties) is described from captured source in the tech wiki's page model-trainer-cloze-substitution-scoring, pinned at commit 46fb6bcc; this page does not restate it beyond one paragraph and covers the paths around it"
fact_checked: "2026-10-05"
confidence: high
hubs: [services]
---

# Model-Trainer's cloze scoring path

Every accuracy the extraction-ablation and cartridge programmes publish is a
cloze score read as **lift over an untrained model's floor** — gpt2 scores
1374/2627 = 0.523030072325847 on the fixed item set.[^research] So the floor
and the arm must be scored by code that cannot differ, and this page is how
that is arranged.

## The scorer, in one paragraph

`score_cloze_items` renders each item's template once per candidate, assigns
each rendering a **total** negative log-likelihood (a mean would reward
candidates that tokenise long), and counts the item correct only when the
answer, at index 0, is the **strict** minimum — a tie is a wrong answer.[^score][^tie]
Nothing is sampled, so no seed is needed. Renderings are truncated to
`max_seq_len`, and one that tokenises to fewer than two ids raises
`CLOZE_ITEM_UNSCOREABLE` rather than being skipped.[^score] The tech wiki's
`model-trainer-cloze-substitution-scoring` page covers the rule in depth.

## Four doors, one scorer

Two queue jobs and two CLIs reach the same `score_cloze_items`:[^jobs][^baseline][^run]

| door | model | where it runs |
|---|---|---|
| `POST /runs/baselines/cloze` → `baseline_cloze_job` | untrained hub model | the service (API + Redis + RQ) |
| `POST /runs/{id}/cloze` → `cloze_job` | a trained run | the service |
| `score_baseline` CLI | untrained hub model, items from a staged file | a compute node |
| `score_run` CLI | a trained run's artifact | a compute node |

All four pin determinism the same way, with both controls on —
`apply_determinism_hook(remove_split_k=True, math_attention=True)` — **before
the model loads**, because `CUBLAS_WORKSPACE_CONFIG` is read when the cuBLAS
handle is created and loading weights creates it.[^jobs][^baseline] A posture
that differed between the queue path and the CLI would make the same floor
disagree in its last bits for a reason nobody would look for.[^baseline] The
CLIs also take a required `--kernel` arm, applied after load and before the
first item.[^baseline]

**`score_run` must not be replaced by pointing `score_baseline` at a run
directory.** The baseline loader's contract is "nothing applied", so for a
LoRA or QLoRA arm it would load whatever base sits there, never attach the
adapter, and report the base model's accuracy as the arm's — plausibly, with
nothing in the record to notice it by. `score_run` instead rebuilds the base
under the run's own quantization and asks the strategy that trained it to
reattach the adapter ([[model-trainer-finetune-strategy-seam]]).[^run]

## What a score leaves behind

A `RunRecord` with observations `cloze_accuracy`, `cloze_chance`,
`cloze_correct`, `cloze_total`, the run's fingerprint
([[model-trainer-run-record-provenance]]), and a `payload_digest` from
`outcomes_digest`; plus every per-item outcome, scores included, written
beside it.[^baseline]

**The digest covers `[item_id, correct]` pairs in scoring order and
deliberately excludes the scores.** Until 2026-08-25 it included them, and the
first cross-card comparison — 1374 correct on both a 3090 Ti and an A100,
identical to fifteen digits — produced different digests, because raw NLLs
differ in their low bits between any two cards. A check that fires on every
comparison carries no information.[^baseline] With the decision-only digest,
the gpt2 floor has since been shown equal **item for item** (payload
`e964e46b…`) on every card it has been scored on — 3090 Ti, V100, A30, A100,
L40S, RTX PRO 6000 and RTX A2000 — and across a CUDA-stack boundary.[^research] The floors are registered as known
answers ([[model-trainer-known-answer-registry]]).

Two cautions a new arm should inherit: name the experiment for what it will be
compared against, since records from different experiments refuse to compare
([[model-trainer-run-record-provenance]]); and check the item set can resolve
the difference you expect before reading one —
[[a-claim-four-times-below-its-instruments-floor]] is what happens otherwise.[^research]

[^research]: `docs/RESEARCH.md` § "`floor` — cloze floor scoring" (the `Provenance:` bullet and the generation table).
[^score]: `services/Model-Trainer/src/model_trainer/core/services/model/cloze/score.py` / `score_cloze_items`, `sequence_nll` (`MIN_SCOREABLE_TOKENS`), and the module docstring.
[^tie]: `services/Model-Trainer/src/model_trainer/core/contracts/cloze.py` / `answer_wins_outright`.
[^jobs]: `services/Model-Trainer/src/model_trainer/worker/baseline_cloze_job.py` and `services/Model-Trainer/src/model_trainer/worker/cloze_job.py` — each calls `_test_hooks.apply_determinism_hook(remove_split_k=True, math_attention=True)`.
[^baseline]: `services/Model-Trainer/src/model_trainer/cli/score_baseline.py` / `score_with_outcomes` (docstring and the comment "Both controls on, matching the queue path exactly"), `outcomes_digest`, `encode_outcomes`.
[^run]: `services/Model-Trainer/src/model_trainer/cli/score_run.py` — module docstring ("WHY NOT JUST POINT `score_baseline` AT THE DIRECTORY").
