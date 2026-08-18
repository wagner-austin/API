---
title: MJX reproduces bit for bit on CUDA, including across fresh processes
tags: [mjx, determinism, measurement, cuda, batching]
related: ["[[mjx-batched-step-reproduces-on-cpu]]", "[[mjx-determinism-does-not-cross-backends]]", "[[jax-cuda-unavailable-on-windows]]"]
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
  jax: 0.11.0
  backend: cuda
  device: NVIDIA GeForce RTX 3090 Ti, driver 591.86, under WSL2 Ubuntu
  model: falling sphere with a free joint on a plane, timestep 0.005
  seed: 7
  step_count: 200
  repetitions: 5
  world_counts: [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096]
  perturbation: 0.05
hubs: [determinism-measurement]
---

# MJX reproduces bit for bit on CUDA, including across fresh processes

> **Read the conditions.** This was measured on a single falling sphere: one body, whose only contact is with the floor. GPU determinism is **not** general — it fails once six bodies touch *each other*, as [[warp-gpu-determinism-fails-on-coupled-bodies]] shows under a controlled sweep. This scene has no body-to-body contact at all, so it sits below that threshold by construction and must not be read as "the GPU is deterministic".

The GPU-batched, `jit`-compiled path is reproducible *for this scene*. Five repetitions at one seed agreed exactly at every batch width from 1 to 4096, with no divergence at any of 200 steps, and each width produced its own distinct reference digest.[^1]

This is the case that mattered. GPU reductions are where batched non-determinism is expected to arise at all, because reduction order across parallel lanes carries no ordering guarantee. The CPU result in [[mjx-batched-step-reproduces-on-cpu]] said nothing about it.

## Fresh-process reproducibility

Three independent interpreters each recorded a trial to disk; a fourth process read the recordings back and compared them. All three agreed.[^2]

That is a strictly stronger claim than repetition within one process. Repetitions inside one interpreter share module state, the JIT cache, and allocator history — agreement between them is consistent with a rollout that would not survive a restart. These three shared none of that, and the comparison was made by a process that never saw any of their memory.

| comparison | match | first divergence |
|---|---|---|
| `wsl-cuda` vs `wsl-cuda-2` | true | none |
| `wsl-cuda-2` vs `wsl-cuda-3` | true | none |

## Scale

4096 parallel worlds is 28,672 float32 values per step, digested every step for 200 steps, and it reproduced. The instrument is not measuring a degenerate batch that happens to be too small to expose a reduction-order effect.

## What is still not measured

The **rendered** observation stream. MJX-Warp's batch renderer is a raycaster over a per-step bounding-volume hierarchy, which is a different numerical path from the solver measured here, and it needs the Warp backend rather than only CUDA. That remains the project's open question.


## What this does not establish

One GPU, one model, one seed. The scene has no body-to-body contact at all, so it sits below the coupled-body threshold by construction and this result does not extend past it — see [[warp-gpu-determinism-fails-on-coupled-bodies]].

Nor does it say anything about agreement with any *other* backend: reproducing against itself and reproducing against CPU are different properties, and the second one fails ([[mjx-determinism-does-not-cross-backends]]).

[^1]: `[observed]` — sweep over the listed `world_counts` with `TrialSpec(seed=7, step_count=200, repetitions=5)` under `JAX_PLATFORMS=cuda`; every row reported `deterministic=true`, `first_divergent_step=none`, and a distinct `reference_digest`.
[^2]: `[observed]` — three runs of `record_trial` in separate interpreters writing to `wsl-cuda`, `wsl-cuda-2`, `wsl-cuda-3`, compared by a fourth process via `compare_recordings(..., 0)`; all reported `digests_match=true`. Each run independently reported `reference_digest=e7fd96eb84b03cb8df79d06deecc0af06e58fce48f74e9d415bd4c2a4dd69de9` at `world_count=4`.
