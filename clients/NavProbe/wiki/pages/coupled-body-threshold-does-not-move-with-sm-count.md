---
title: The coupled-body determinism threshold does not move with SM count
tags: [warp, determinism, measurement, finding, gpu, occupancy, negative-result]
related: [[warp-gpu-determinism-fails-on-coupled-bodies]], [[mjwarp-cannot-compile-under-warp-deterministic-mode]], [[the-numbers-are-scene-dependent-the-shapes-replicate]], [[measurement-fleet-is-reachable-by-ssh-alias]]
sources: [mujoco-warp 3.11.0, warp-lang 1.16.0]
fact_checked: 2026-08-16
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0
  devices:
    - NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, driver 13.1, host austinpc)
    - NVIDIA GeForce RTX 3070 Ti Laptop (sm_86, 46 SMs, driver 551.23, host sedona)
  harness: navprobe.sweep.run_scene_sweep over navprobe.scenes.row_scene
  adapter: navprobe.adapters.mjx_warp_state
  seed: 7
  step_count: 150
  repetitions: 12
  world_count: 2
  perturbation: 0.01
  constraint_capacity: 8192
  body_counts: [2, 4, 5, 6, 8, 32] touching / [2, 8, 16, 32] separated
  spacing: 0.055 (touching) and 0.070 (separated), radius 0.03, timestep 0.005
---

# The coupled-body determinism threshold does not move with SM count

[[warp-gpu-determinism-fails-on-coupled-bodies]] locates a boundary: a row of spheres
initialised in mutual lateral contact stops reproducing at five bodies, while a separated
row reproduces to at least thirty-two. The obvious next question is whether that boundary
is a property of the *solver* or of *how the work happened to be scheduled across the
device*. If it moves with the number of streaming multiprocessors, it is an occupancy
artefact and every published body count is device-specific.

**It does not move.** Two devices of the same architecture and nearly twice the width apart
place the boundary at the identical body count.[^1]

## The comparison

Same harness, same adapter, same `TrialSpec`, same seed, same scenes. The only variable is
the device.

| | RTX 3090 Ti | RTX 3070 Ti Laptop |
|---|---|---|
| SM count | 84 | **46** |
| driver | 13.1 | 551.23 |
| separated, 2 / 8 / 16 / 32 bodies | all reproduce | all reproduce |
| touching, 2 bodies | reproduces | reproduces |
| touching, 4 bodies | reproduces | reproduces |
| touching, 5 bodies | **fails** | **fails** |
| touching, 6 / 8 / 32 bodies | fails | fails |
| **threshold** | **5** | **5** |

A 1.83× difference in width and two different driver branches produce the same boundary in
the same place, with the separated family intact at every size on both.[^1][^2]

## Why this is worth having

It removes a live alternative explanation. Non-determinism from atomic accumulation is
plausibly a function of how many blocks run concurrently — more resident blocks, more
interleavings, more opportunity for a different summation order. Under that model a
device with 46 SMs should tolerate a larger coupled group than one with 84 before the
boundary is crossed. That prediction is falsified.

What survives is the explanation the original page argues for on structural grounds: the
threshold is set by **what the contacts connect**, not by how the resulting work is spread.
Once a chain of constraints shares degrees of freedom, accumulation order matters, and it
matters at the same group size regardless of how much parallel width is available to
disorder it.

Practically, it also means body counts published on this wiki are not device-specific
trivia. A reader on different Ampere hardware can expect the same boundary.

## What this does not establish

**Both devices are `sm_86`.** This isolates SM count *within* one architecture, which is
exactly what makes it a clean test of occupancy — and exactly why it says nothing about
codegen differences between architectures. Whether the boundary holds on Turing or Ada
remains the open question in [[open-questions-and-what-would-answer-them]], and no machine
in the fleet can answer it ([[measurement-fleet-is-reachable-by-ssh-alias]]).

It also does not establish that occupancy is irrelevant *everywhere* — only that it does
not move this threshold, on this scene family, at these sizes. A scene whose work does not
saturate 46 SMs in the first place would not test the question.

One observation recorded without interpretation: on the 46-SM device the failing
conditions diverge at step 0 (bodies 5, 8 and 32) or step 1 (bodies 6), not after
accumulation over many steps.[^1] No equivalent per-step figure was recorded on the 84-SM
device, so this is not offered as a comparison.

[^1]: `[observed]` — on host `sedona`, `PYTHONPATH=C:\navprobe\src C:\navprobe\.venv\Scripts\python.exe C:\navprobe\sm_occupancy_sweep.py`, driving `navprobe.sweep.run_scene_sweep` with the frontmatter conditions. Reported `arch 86, sm_count 46`. Separated family `deterministic=true` at 2/8/16/32. Touching family `deterministic=true` at 2 and 4, `false` at 5, 6, 8 and 32, with `first_divergent_step` 0, 1, 0 and 0 respectively. Computed `touching_threshold` 5.
[^2]: [[warp-gpu-determinism-fails-on-coupled-bodies]] footnote 3 — the 84-SM baseline through the identical harness: "The separated family (`spacing=0.070`) reported `deterministic=true` at every size and `first_irreproducible` of none; the touching family (`spacing=0.055`) reported true at 2 and 4 and false from 5 upward." Same `TrialSpec(seed=7, step_count=150, repetitions=12)`, `world_count=2`, `perturbation=0.01`.
