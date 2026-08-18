---
title: MJX determinism holds within a backend and does not cross between them
tags: [mjx, determinism, measurement, cuda, portability, finding]
related: ["[[mjx-cuda-batched-step-reproduces]]", "[[mjx-batched-step-reproduces-on-cpu]]", "[[cpu-determinism-survives-os-and-version-change]]"]
source_paths:
  - "src/navprobe/crossprocess.py"
source_git_blobs:
  "src/navprobe/crossprocess.py": "2d5bfcb44bd8f37f3f2814edccb76f3b444775ee"
provenance:
  - "mujoco-mjx 3.11.0"
  - "jax 0.11.0"
fact_checked: 2026-08-13
confidence: high
measured_with:
  package: mujoco-mjx 3.11.0
  jax: 0.11.0 (both sides; CPU side also confirmed on 0.10.2)
  backend: cpu vs cuda
  device: NVIDIA GeForce RTX 3090 Ti under WSL2 Ubuntu; CPU side on the same machine
  model: falling sphere with a free joint on a plane, timestep 0.005
  seed: 7
  step_count: 200
  repetitions: 3
  world_counts: [1, 2, 4, 8, 16, 64]
  perturbation: 0.05
hubs: [determinism-measurement]
---

# MJX determinism holds within a backend and does not cross between them

A rollout is bit-reproducible on CPU, and bit-reproducible on CUDA, and the two do not agree with each other. They agree for a while and then part.

| left | right | match | first divergence | what differs |
|---|---|---|---|---|
| Windows CPU, jax 0.10.2 | WSL CPU, jax 0.11.0 | **true** | none | OS and library version |
| WSL CUDA | WSL CUDA (fresh process) | **true** | none | process only |
| WSL CPU | WSL CUDA | false | **57** | backend only |
| Windows CPU | WSL CUDA | false | **57** | everything |

The third row is the finding: OS and jax version held constant, only the backend varying.[^1]

## It is accumulate-then-diverge, not immediate

The two backends produce **identical bytes for the first 57 steps**, then differ. This is the failure mode the instrument was built to localise, and it is the one a coarse check misses: a comparison of final state alone reports "different" with no information, and a short rollout reports "identical" and concludes portability.

A test that ran 50 steps would pass. The same test at 60 steps would fail. Nothing about the setup changed.

## The divergence point depends on batch width, and then stops

| `nworld` | first divergence |
|---:|---:|
| 1 | 19 |
| 2 | 57 |
| 4 | 57 |
| 8 | 57 |
| 16 | 57 |
| 64 | 57 |

A single world parts from CPU at step 19; every batched width from 2 upward parts at 57, flat.[^2] The transition sits between 1 and 2 — that is, between not batching and batching at all, rather than anywhere along the width axis.

**Step 57 is now explained: it is the first contact solve.** The sphere penetrates its resting height at step 56 and the contact impulse first acts at 57, and MuJoCo-Warp — a completely different compiler and execution model — parts from its own CPU counterpart at the same step. The full account is in [[backend-divergence-begins-at-first-contact]]. Everything before 57 is free flight, which is a fixed-order integration that every backend reproduces exactly.

**Step 19 at width one is still not explained.** It falls well inside free flight, so the contact account does not cover it. The plausible reading remains that XLA selects a different kernel for a batch of one; that is a hypothesis this measurement does not test, and it is recorded as one.

## Why this matters more than the within-backend result

Both backends being individually deterministic is the reassuring half. The useful half is that "deterministic" turns out not to mean "portable":

- A golden digest captured from a CPU CI job cannot be compared against a GPU training run. It will diverge, and the divergence will look like a regression.
- The divergence point moves with batch width, so a CI job that changed its batch size would see the failure move and read as flaky.
- Because the disagreement takes 57 steps to appear, a short smoke test certifies agreement that longer runs do not have.

This is exactly the kind of unearned transfer the project was started to object to: a determinism result measured under one configuration, quietly assumed to hold under another.

## Conditions this does not cover

Only one model, one seed, and one GPU. 57 has since been shown to be a property of *this trajectory* — the step at which the ball lands — rather than of MJX, so the number will move with any scene that makes contact at a different time. The distribution of divergence points across seeds and models remains unmeasured.

[^1]: src/navprobe/crossprocess.py:82 `record_trial`, :136 `compare_recordings` — `[observed]` — each environment ran `record_trial` to its own directory; a separate process compared them with `compare_recordings(left, right, 0)`. Verdicts as tabulated. The CPU/CPU row compares Windows-native `jax` 0.10.2 against WSL2 `jax` 0.11.0.
[^2]: src/navprobe/crossprocess.py:136 `compare_recordings` — `[observed]` — recordings at `world_count` 1, 2, 4, 8, 16, 64 on both backends, each compared at repetition zero; `first_divergent_step` was 19 at width 1 and 57 at every other width. Each recording's `world_count` was read back from its trial summary and asserted to match the width requested.
