---
title: Cross-backend divergence begins at the first contact solve, not before
tags: [mjx, warp, determinism, measurement, finding, contact]
related: ["[[mjx-determinism-does-not-cross-backends]]", "[[warp-renderer-depth-is-not-device-portable]]"]
provenance:
  - "mujoco-mjx 3.11.0"
  - "mujoco-warp 3.11.0"
  - "jax 0.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-13
confidence: high
measured_with:
  package: mujoco-mjx 3.11.0 and mujoco-warp 3.11.0
  jax: 0.11.0
  warp: 1.16.0
  backend: cpu vs cuda, on both stacks
  device: NVIDIA GeForce RTX 3090 Ti
  model: falling sphere with a free joint on a plane, timestep 0.005, radius 0.1, initial height 0.5
  seed: 7 (MJX) and fixed offsets 0.013 / -0.027 (Warp)
  step_count: 200
  world_counts: [1, 2, 4, 8, 16, 64]
hubs: [determinism-measurement]
---

# Cross-backend divergence begins at the first contact solve, not before

Two entirely separate stacks — MJX compiled by XLA, and MuJoCo-Warp compiled by Warp — each first disagree with their own CPU counterpart at **step 57** of the same trajectory. Different compilers, different execution models, different initial offsets, same step.[^1]

That is not a coincidence about the backends. It is a property of the trajectory.

## Step 57 is when the ball hits the floor

Reading the height trace, the per-step change is a clean free-fall increment through step 56 and then abruptly halves:

| step | z | Δz |
|---:|---:|---:|
| 54 | 0.12231506 | −0.01348875 |
| 55 | 0.10858106 | −0.01373401 |
| 56 | 0.09460180 | −0.01397926 |
| **57** | 0.08793730 | **−0.00666450** |
| 58 | 0.08534671 | −0.00259060 |
| 59 | 0.08495498 | −0.00039173 |
| 60 | 0.08568716 | +0.00073218 |

The sphere's resting height is its radius, 0.1. It penetrates that at step 56, and at step 57 the increment stops following gravity: the contact impulse is acting.[^2]

## Why that is exactly where the backends part

Free flight is an integration — a handful of multiplies and adds per world, in a fixed order, with no reduction across lanes. Every backend gets the same bits, which is what the measurements show: **56 steps of bit-identical agreement** across CPU, CUDA, XLA and Warp alike.

Contact is an iterative constrained solve. It reduces across contacts and degrees of freedom, and floating-point addition is not associative, so a different summation order gives a different — equally valid — result. That is the first operation in this trajectory where the backends have any licence to differ, and they take it immediately.

So the divergence point is not a property of MJX, of JAX, of Warp, or of CUDA. It is the step at which the workload stops being trivially reproducible.

## What follows from that

- **A divergence point is a fact about the scene, not the stack.** Reporting "MJX diverges at step 57" without saying the ball lands at step 57 attributes to the simulator what belongs to the trajectory.
- **A contact-free benchmark proves nothing about portability.** Any rollout that never resolves a constraint will reproduce perfectly across backends and certify agreement the moment a contact appears.
- **The divergence is not a bug to be fixed.** Both answers satisfy the solver's tolerance. What is wrong is the assumption that a digest captured on one backend transfers to another.

## The one case this does not explain

At `nworld = 1`, MJX diverges from CPU at **step 19** — well inside free flight, thirty-eight steps before contact.[^3] So the batch-of-one path differs even in the trivial integration, which the contact account does not cover.

A plausible reading is that XLA selects a different kernel or vectorisation strategy for a batch of one, so a single world is computed by different code from a world inside a batch. That is a hypothesis this measurement does not test, and it is recorded as one. Every batched width from 2 to 64 diverges at 57, consistent with contact.

[^1]: `[observed]` — MJX CPU vs CUDA at widths 2-64 reported `first_divergent_step = 57` (see [[mjx-determinism-does-not-cross-backends]]). A 200-step MuJoCo-Warp rollout compared per step across `cuda:0` and `cpu` reported `first differing step: 57`, with 143 of 200 steps differing thereafter.
[^2]: `[observed]` — height trace of world 0 from the 200-step Warp rollout: Δz holds at roughly −0.0135 to −0.0140 through step 56, then −0.00666450 at step 57, −0.00259060 at 58, −0.00039173 at 59, and turns positive at 60. Minimum height 0.08495498 against a resting height of 0.1.
[^3]: src/navprobe/crossprocess.py:136 `compare_recordings` — `[observed]` — MJX recordings at `world_count = 1` compared across backends reported `first_divergent_step = 19`; at that step the height trace is still in free fall.
