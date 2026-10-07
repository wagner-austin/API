---
title: Model-Trainer's three ways in — service, compute node, and the measurement CLIs
tags: [services, model-trainer, architecture, rq, hpc3, cli, research]
related:
  - "[[model-trainer-finetune-strategy-seam]]"
  - "[[model-trainer-run-record-provenance]]"
  - "[[model-trainer-cloze-scoring-path]]"
  - "[[model-trainer-known-answer-registry]]"
  - "[[corpus-attachment-program]]"
  - "[[platform-workers-rq-pattern]]"
  - "[[service-port-map]]"
source_paths:
  - services/Model-Trainer/pyproject.toml
  - services/Model-Trainer/src/model_trainer/api/main.py
  - services/Model-Trainer/src/model_trainer/api/routes/runs.py
  - services/Model-Trainer/src/model_trainer/core/services/queue/rq_adapter.py
  - services/Model-Trainer/src/model_trainer/core/services/container.py
  - services/Model-Trainer/src/model_trainer/cluster/entry.py
  - docs/RESEARCH.md
source_git_blobs:
  "services/Model-Trainer/pyproject.toml": 0e3ca83a18d280bfc5f1ec8e4e153325dbfa4ef7
  "services/Model-Trainer/src/model_trainer/api/main.py": 19c915e2f19cd975e6deb3b31dfd3cecd52ed36a
  "services/Model-Trainer/src/model_trainer/api/routes/runs.py": 5499dda1eac9e3c5e034e9c636e4a2bf5bf60c27
  "services/Model-Trainer/src/model_trainer/core/services/queue/rq_adapter.py": 09a397ac1ad9036bde1c5889f8d1214c6d7b0957
  "services/Model-Trainer/src/model_trainer/core/services/container.py": e3165ede22b989b0679c5b8b3122a95785e107fc
  "services/Model-Trainer/src/model_trainer/cluster/entry.py": 67c8783d843179348bfdcd6b59641fa4cbab6a27
  "docs/RESEARCH.md": 501940c688722b74b5b1a49b1825b2a22e2322e4
provenance:
  - "docs/RESEARCH.md repinned 2026-10-07 from fa186d5d to 501940c6 on a mechanical argument rather than a re-reading: the diff is +17/-0, one paragraph appended to the turkic-lstm section by c0c80420b (board task bc12f18b); the mi section and its Runs bullet that this page cites are byte-identical. Check with: git diff fa186d5df657233450dfac204982ad9eff07173d 501940c688722b74b5b1a49b1825b2a22e2322e4"
  - "read from code at API commit 2e0511ee7 on 2026-10-05; the CLI module count is a listing of src/model_trainer/cli/ (42 .py files besides __init__, including report and hook modules that are not entry points)"
fact_checked: "2026-10-05"
confidence: high
hubs: [services]
---

# Model-Trainer's three ways in

Model-Trainer is a training service on port 8005 and also the `mi` research
project in `docs/RESEARCH.md` (with `floor`, `mi-cu128` and `cartridge-qa`
running its code too). The two roles share one core and enter it three ways:[^research][^scripts][^cluster]

| way in | composition root | dependencies it wires |
|---|---|---|
| HTTP service | `api/main.py::create_app` + `modeltrainer-rq-worker` | Redis, RQ, data-bank-api |
| one training job on a compute node | `modeltrainer-cluster-train` → `cluster/entry.py` | in-process store, staged corpus, local artifact dir |
| measurement CLIs | `python -m model_trainer.cli.<name>` (three also as scripts) | the job's own determinism pin and fingerprint |

Five console scripts exist: the RQ worker, the cluster trainer,
`modeltrainer-score-baseline`, `modeltrainer-score-run` and
`modeltrainer-continuations`.[^scripts] Every other CLI runs as a module.

## The service: API enqueues, worker executes

`create_app` mounts three routers — health, `/runs`, `/tokenizers` — each built
from the container; `/runs` and `/tokenizers` carry the API-key dependency.[^app] Long work under
`/runs` is enqueued through `RQEnqueuer`, one method per job kind, each naming
a function under `model_trainer.worker.*`: `train_job`, `eval_job`,
`tokenizer_worker`, `cloze_job`, `baseline_cloze_job`, `score_job`,
`generate_job`, `chat_job`.[^rq] The container refuses to start unless the RQ
queue is `TRAINER_QUEUE`.[^container]

The model registry names five families, and **two of them are stubs**:
`gpt2`, `char_lstm` and `hf_lm` have real backends; `llama` and `qwen` are
registered as `UnavailableBackend`.[^container] Llama- or Qwen-architecture
work goes through `hf_lm`, which loads any HuggingFace causal LM by hub id and
is the only backend that applies a fine-tuning strategy — see
[[model-trainer-finetune-strategy-seam]].

## The compute node: a second composition root, not a second trainer

`cluster/entry.py` hands the **same payload to the same `process_train_job`**
the RQ worker calls, so the training code cannot tell which one started it.[^cluster]
What a node changes is stated in its docstring and worth knowing before
touching it: progress has no reader and lives in memory; cancellation is
Slurm's `scancel`, so the cancel key is never set; the corpus must already be
staged (by `hpc3-stage`, digest-checked); and **resume is decided by the node,
not the payload** — Slurm may re-execute one payload file after preemption,
so `_resume_for_execution` supersedes the payload's `resume` with whether a
checkpoint exists.[^cluster] Jobs reach it only through `tools/hpc3`; the
hpc3 wiki's `submission-rules` and `preemption-and-campaigns` pages carry why.

## The CLIs: where the numbers come from

`src/model_trainer/cli/` holds the research entry points — GEMM/SDPA/forward
benchmarks and probes, `probe_ladder`, `train_benchmark`, `score_baseline`,
`score_run`, the known-answer pair, and the cartridge sweep family. The
authoritative list of which of them produce compared numbers is the `mi`
entry in `docs/RESEARCH.md`, and the `research-registration` guard fails
`make lint` on any entry point that builds a `RunRecord` without being listed
there.[^research] Every one of them emits the shared record shape —
[[model-trainer-run-record-provenance]] — and the scoring path they share is
[[model-trainer-cloze-scoring-path]]. The cartridge programme's results are
indexed from [[corpus-attachment-program]].

[^scripts]: `services/Model-Trainer/pyproject.toml:54-59` — `[tool.poetry.scripts]`.
[^app]: `services/Model-Trainer/src/model_trainer/api/main.py` / `create_app` — `api_key_dependency`, three `include_router` calls.
[^rq]: `services/Model-Trainer/src/model_trainer/core/services/queue/rq_adapter.py` / `RQEnqueuer.enqueue_train`, `enqueue_eval`, `enqueue_tokenizer`, `enqueue_cloze`, `enqueue_baseline_cloze`, `enqueue_score`, `enqueue_generate`, `enqueue_chat`.
[^container]: `services/Model-Trainer/src/model_trainer/core/services/container.py` / `_create_model_registry` (five registrations, `UnavailableBackend("llama")`, `UnavailableBackend("qwen")`), `_create_enqueuer` ("RQ queue must be trainer per platform alignment").
[^cluster]: `services/Model-Trainer/src/model_trainer/cluster/entry.py` — module docstring, and `_resume_for_execution`.
[^research]: `docs/RESEARCH.md` § "`mi` — Model-Trainer probes and benchmarks", the `Runs:` bullet.
