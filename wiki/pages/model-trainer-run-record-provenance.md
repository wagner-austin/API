---
title: RunRecord and RunFingerprint as Model-Trainer emits them — six axes, three verdicts, one refusal
tags: [services, model-trainer, provenance, run-record, comparability, determinism, platform-core]
related:
  - "[[model-trainer-service-architecture]]"
  - "[[model-trainer-cloze-scoring-path]]"
  - "[[model-trainer-known-answer-registry]]"
  - "[[covenant-radar-optuna-optimisation]]"
  - "[[determinism-env-read-once-at-library-load]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/core/run_fingerprint.py
  - libs/platform_core/src/platform_core/comparability.py
  - libs/platform_core/src/platform_core/run_record.py
  - docs/RESEARCH.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/core/run_fingerprint.py": 29b5d9b803fcb6b3cec894100d60d67a0a336bc6
  "libs/platform_core/src/platform_core/comparability.py": eec82a133f44877523c06c45a182bb6cf7a21741
  "libs/platform_core/src/platform_core/run_record.py": 7d3ca55d81674311342b179cbf476a3d61b14dd6
  "docs/RESEARCH.md": fa186d5df657233450dfac204982ad9eff07173d
provenance:
  - "read from code at API commit 2e0511ee7 on 2026-10-05"
fact_checked: "2026-10-05"
confidence: high
hubs: [services, libs]
---

# RunRecord and RunFingerprint as Model-Trainer emits them

`docs/RESEARCH.md` calls Model-Trainer's records "the only surface here that
carries all six axes".[^research] This page is what those axes are, where
Model-Trainer fills them, and what the comparability layer does with them —
the reference a new research entry point should copy.

## The record

`platform_core.run_record.RunRecord` is an experiment name, a label, named
`Observation`s, a `payload_digest`, and a `RunFingerprint`.[^rr] The
experiment pairs records; the label distinguishes them; the digest covers the
detail nobody subtracts (for cloze scoring, which items were right —
[[model-trainer-cloze-scoring-path]]).

## The six axes

`RunFingerprint` records what the run **resolved to, never what it
requested** — a lock file is intent, and on this project's published arms the
two disagreed.[^fp]

| axis | Model-Trainer's source |
|---|---|
| `image_digest` | the launcher's `IMAGE_DIGEST` variable; `""` when there is no image |
| `gpu_model` | the CUDA device name, read **only** for a cuda run |
| `driver_version` | the CUDA driver, likewise cuda-only |
| `determinism` | the `DeterminismRecord` the job's own pin returned |
| `host` | `capture_host_record` |
| `packages` | `numpy`, `torch`, `transformers` by default; wider for adapter or quantized runs |

Sources: `capture_run_fingerprint`.[^mt]

Four decisions in `capture_run_fingerprint` are the parts worth copying:[^mt]

- **The empty string is "unknown", and unknown is a difference, not a
  wildcard.** A cpu run records `""` for card and driver so it can never
  compare equal to a cuda run.[^mt][^fp]
- **Determinism is passed in, never applied here.** Pinning is a
  process-global side effect that belongs to the job, before any CUDA work;
  taking the pin's return value means the fingerprint can only claim a
  posture something actually applied.[^mt] (`covenant-radar`'s optimiser
  breaks exactly this rule — see [[covenant-radar-optuna-optimisation]].)
- **Packages are named, not enumerated.** All installed distributions would
  differ over a dev-dependency bump that cannot reach a matmul.[^mt]
- **`image_digest` is not the commit.** It held the commit while no image
  existed; a commit says which code, a digest says which environment, and two
  runs can share a commit and differ in torch.[^mt]

The module's own docstring records why it exists: the 52.3030% gpt2 floor that
every arm is read as lift over had been recorded with no device, driver,
torch version or timestamp.[^mt]

## Three verdicts

`compare_configurations(left, right, calibrations)` answers whether two
numbers may be **subtracted** — not whether they agree:[^cmp]

- `identical` — no axis differs;
- `offset` — every differing axis is covered by a measured `Calibration`,
  whose offsets are summed and applied;
- `uncalibrated` — at least one difference has no calibration, and the
  verdict names exactly which.

`compare_run_records` judges the configuration **first** and returns no
deltas at all on `uncalibrated` — "returning numbers beside a 'not comparable'
note is how a caller ends up using them".[^rr]

## One refusal

`compare_run_records` **raises** on two records with different experiment
names: no calibration bridges two different questions.[^rr] This has bitten:
the floor's 3090 Ti records were named `extraction-eval` and its cluster
records `wiki-corpus-extraction-ablation`, so their equality could only be read
by eye until the baseline was re-scored under the campaign's name.[^research]
**Choose the experiment name for what you will compare against, not for the
command that produced it.**

[^research]: `docs/RESEARCH.md` § "`mi` — Model-Trainer probes and benchmarks" (`Provenance:` bullet) and § "`floor` — cloze floor scoring" (the bullet beginning "The finding was real and not instrumented until 2026-09-13").
[^rr]: `libs/platform_core/src/platform_core/run_record.py` / `RunRecord`, `Observation`, `compare_run_records`.
[^fp]: `libs/platform_core/src/platform_core/comparability.py` / `RunFingerprint` (docstring).
[^mt]: `services/Model-Trainer/src/model_trainer/core/run_fingerprint.py` / `capture_run_fingerprint`, `FINGERPRINT_DISTRIBUTIONS`, `NO_GPU`, and the module docstring.
[^cmp]: `libs/platform_core/src/platform_core/comparability.py` / `compare_configurations`, `IdenticalVerdict`, `OffsetVerdict`, `UncalibratedVerdict`.
