---
title: MJX's batched step reproduces bit for bit on CPU across batch widths
tags: [mjx, determinism, measurement, batching]
related: ["[[jax-cuda-unavailable-on-windows]]", "[[vmap-requires-every-leaf-batched]]"]
source_paths:
  - "src/navprobe/adapters/mjx.py"
  - "src/navprobe/sweep.py"
  - "tests/adapters/test_mjx.py"
source_git_blobs:
  "src/navprobe/adapters/mjx.py": "539e5fce071b6caf91eba05cf6f6e65bc8e0433d"
  "src/navprobe/sweep.py": "3fbecbbf2caeec07827f618c307ded4699414dfd"
  "tests/adapters/test_mjx.py": "1ea2dc9841cc9bbf54c49663d132a965f4650d3f"
provenance:
  - "mujoco-mjx 3.11.0"
fact_checked: 2026-08-13
confidence: high
measured_with:
  package: mujoco-mjx 3.11.0
  jax: 0.10.2
  backend: cpu
  model: falling sphere with a free joint on a plane, timestep 0.005
  seed: 7
  step_count: 200
  repetitions: 5
  world_counts: [1, 2, 4, 8, 16, 32, 64]
  perturbation: 0.05
hubs: [determinism-measurement]
---

# MJX's batched step reproduces bit for bit on CPU across batch widths

Five independent rollouts at one seed, compared step by step, agreed exactly at every batch width from 1 to 64. No divergence at any step.

| `nworld` | deterministic | first divergence | reference digest |
|---------:|:--------------|:-----------------|:-----------------|
| 1 | true | none | `eb5dfed73feae650e6551a42…` |
| 2 | true | none | `d7c99accf762692f57bcc6ba…` |
| 4 | true | none | `6b8e692c1a0c6e60ac74882d…` |
| 8 | true | none | `90d6b908261cc9683af44398…` |
| 16 | true | none | `1ac9829c2ad55fa2cc7c60c8…` |
| 32 | true | none | `2802691f4e9efdae3ccf3e6c…` |
| 64 | true | none | `07e704ce113f9bcb82d5171c…` |

Each repetition was driven by a freshly constructed simulator sharing one `jit`-compiled `vmap`-batched kernel, which is how a user runs MJX.[^1]

## What this does and does not establish

It establishes that **the contact-and-integration path is reproducible on the CPU backend**, under repetition within one process, at these widths.

It does not, on its own, establish anything about the CUDA backend, fresh processes, or rendering. Two of those three have since been measured separately, and one of them came out the other way:

- **The CUDA backend** — measured via WSL2 in [[mjx-cuda-batched-step-reproduces]]. Reproducible there too, but **not in agreement with this result**: see [[mjx-determinism-does-not-cross-backends]]. So this page's numbers are correct and specifically do not transfer to GPU.
- **Fresh-process reproducibility** — every repetition here ran in one interpreter, sharing module state, the JIT cache, and allocator history. Measured separately, and it holds: this exact digest is reproduced by a different OS on a different jax version in [[cpu-determinism-survives-os-and-version-change]].
- **The rendered observation stream** — still unmeasured. MJX-Warp's batch renderer is a raycaster over a per-step bounding-volume hierarchy, a different numerical path from the solver measured here, and it requires the Warp backend rather than only CUDA.

## Why an unsurprising result is worth recording

This reproduces the published Type 2 classification on the path that classification was measured over, which is the point: it is the instrument's **positive control**. An instrument that reported divergence here would be measuring itself rather than MJX, and every subsequent negative result would be uninterpretable.

The reference digest differs at every width, so the sweep is distinguishing its conditions rather than collapsing them — a sweep that returned one digest for all seven widths would be reporting that batch width does not reach the observation, which would be a bug in the adapter rather than a finding about MJX.[^2]

## Reproducing it

Build an `MjxSimulatorFactory` per width, hand it to `ProbeService`, and run a trial at a fixed `TrialSpec`.[^3] The trial protocol — one pinned seed, N freshly built simulators, every repetition compared against the first — is what makes the runs comparable; it is not something the caller assembles.

[^1]: `src/navprobe/adapters/mjx.py` L184-210, `__init__` of `MjxSimulatorFactory` (L167) — compiles MJCF, places the model, and builds `jit(vmap(step, in_axes=(None, 0)))` once; each `__call__` returns a simulator with its own batched state.
[^2]: `tests/adapters/test_mjx.py::TestTrialAgainstMjx::test_batch_width_changes_the_reference_digest` — pins that two widths produce different digests.
[^3]: src/navprobe/sweep.py:51 `run_scene_sweep` — `[observed]` — sweep over `world_counts` with `TrialSpec(seed=7, step_count=200, repetitions=5)`, `perturbation=0.05`, on `mujoco-mjx 3.11.0` / `jax 0.10.2` CPU; output is the table above.
