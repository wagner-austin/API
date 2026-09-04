---
title: A declared batch size trained at four times itself, and every record said four
tags: [model-trainer, reproducibility, provenance, training]
hubs: [services]
related: ["[[model-trainer-cartridge-question-set]]", "[[model-trainer-composition-ceiling]]"]
source_paths:
  - "services/Model-Trainer/src/model_trainer/worker/job_utils.py"
  - "services/Model-Trainer/src/model_trainer/core/compute/device_selector.py"
  - "services/Model-Trainer/src/model_trainer/cluster/preflight.py"
  - "services/Model-Trainer/tests/test_device_auto.py"
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/worker/job_utils.py": "f9adcdedbf7f5933186b917008eda8024c39cf07"
  "services/Model-Trainer/src/model_trainer/core/compute/device_selector.py": "6777b63d027855332f2bb635776779b5e7cdbfae"
  "services/Model-Trainer/src/model_trainer/cluster/preflight.py": "5f8fb62bc7ba9d6bfee563da70a4ef72e8ea2db9"
  "services/Model-Trainer/tests/test_device_auto.py": "2d900f80d0c025b60b5bc8bd10a1d7d7269d8530"
provenance:
  - "job 55744675, code-style.qlora-qwen-v1 on an A30, 2026-09-04: CUDA OOM allocating 9.27 GiB, log line 'batch_size=16' against a payload declaring 4"
  - "job 55746427, the same document after the fix: 'batch_size=4', 1633 steps, 1h02m, exit 0"
  - "tools/hpc3/runs/code-style-qlora-v1.json -- the payload, which declares 4 in both runs"
fact_checked: 2026-09-04
confidence: high
---

# A declared batch size trained at four times itself, and every record said four

`build_cfg` handed every training request's batch size to
`recommended_batch_size_for`, which on CUDA rewrote any value of **4 or less**
to a family default: 16 for `hf_lm`, 32 for `gpt2`, 64 for `char_lstm`. Above
4 it passed through untouched[^1].

The rewritten value trained. The **declared** value is what the run record,
the training manifest and the hpc3 ledger row all carried[^2].

## Why this is not a performance note

Batch size decides the optimization trajectory. Two runs of one configuration
at 4 and at 16 are different experiments, not the same experiment at
different speeds — different gradient noise, different step count, different
effective learning rate per example.

So one payload described two experiments, and **which one you got depended on
the entry point**:

| path | reaches `build_cfg` | batch actually trained |
|---|---|---|
| `modeltrainer-cluster-train` → `process_train_job` | yes | 16 |
| a script calling `train_prepared_hf_lm` directly | **no** | 4 |

That is the reason it survived so long. The local sweep that established this
project's configuration called the trainer directly and got what it wrote;
the first time the same document went through the cluster entry, it did
not[^3].

## The OOM that exposed it was luck

Job 55744675 died allocating 9.27 GiB in `ForCausalLMLoss`, which is
`16 x 1024 x 151,936 x 4` bytes — batch sixteen, one thousand twenty-four
tokens, Qwen2.5's vocabulary, float32 logits. A 24 GiB A30 could not hold it.

**On a card with headroom it trains happily and reports the wrong number
forever.** Nothing in the run would have said so: not an error, not a
warning, not a slower step. The record is internally consistent and wrong.

## The fix is deletion, not correction

There is no correct silent rewrite of a declaration. A recommendation belongs
where a human chooses a configuration, not inside the worker that executes
it, so the function is gone rather than adjusted[^4].

The test that asserted the bump now asserts the opposite — that a declared 4
survives onto CUDA — which is the property worth pinning[^5].

## What it says about the class

A run is reproducible only if what it RECORDS is what it DID. Every silent
adjustment between the two is a place the record can be internally consistent
and false, and the adjustments are individually reasonable: resolving
`"auto"` to a concrete device is honest, because the payload asked for auto.
Rewriting `4` to `16` is not, because the payload asked for four.

The distinguishing question is whether the document DELEGATED the choice.
`device: "auto"` delegates. `batch_size: 4` does not.

## The neighbouring gap, closed at the same time

`cluster/preflight` already refused a corpus whose digest no certification
record named, on the reasoning that a corpus is the one input a run cannot
recover from getting wrong. A base model it cannot load is the same class and
had no check, so a run was allocated a GPU, verified every output root,
round-tripped the artifact store, certified its corpus, and died nine seconds
later on the first `from_pretrained`[^6].

`check_model_available` closes it by resolving the config — not the weights —
before training starts. What made that particular cache unreadable is
`tools/hpc3/wiki/pages/offline-model-staging.md` in the sibling wiki.

[^1]: `core/compute/device_selector.py` — the function's removal and the reasons are recorded in the module docstring in past tense.
[^2]: `worker/job_utils.py` section `build_cfg`, at the `batch_size` assignment.
[^3]: `worker/train_job.py` calls `build_cfg`; `core/services/model/backends/hf_lm/train.py` does not, and is what a script reaches directly.
[^4]: `core/compute/device_selector.py` module docstring.
[^5]: `tests/test_device_auto.py` section `test_build_cfg_resolves_auto_and_keeps_the_declared_batch_size`, whose assertion changed from 32 to 4 -- and whose NAME said `adjusts_batch_size` until the adjustment was removed.
[^6]: `cluster/preflight.py` section `check_model_available`, and `check_corpus_certified` above it for the argument it mirrors.
