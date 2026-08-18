---
title: GPU non-determinism in MuJoCo-Warp needs mutually-contacting bodies, not many contacts
tags: [warp, determinism, measurement, finding, contact, gpu]
related: ["[[backend-divergence-begins-at-first-contact]]", "[[warp-rendered-stream-is-reproducible-within-a-device]]", "[[mjx-cuda-batched-step-reproduces]]"]
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
  - "MJWarp documentation"
fact_checked: 2026-08-14
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0 and warp cpu
  device: NVIDIA GeForce RTX 3090 Ti
  model: single row of 0.03 m radius spheres dropped onto a plane between two walls, timestep 0.005
  spacing: 0.070 m (separated, gap 0.010) or 0.055 m (initialised in mutual lateral contact)
  step_count: 150
  world_count: 2
  repetitions: 12 per condition, same process
  harnesses: standalone script (identical worlds) and navprobe.sweep (per-world perturbation 0.01)
hubs: [determinism-measurement]
---

# GPU non-determinism in MuJoCo-Warp needs mutually-contacting bodies, not many contacts

The MJWarp documentation states that GPU results may differ between executions of the same code, and advises `wp.set_device("cpu")` for deterministic results.[^1] It does not say when. The answer is not "when there are many contacts" — it is **when contacts couple moving bodies to each other**, and only once enough of them do.

## The controlled sweep

A single row of spheres dropped onto a plane. No stacking anywhere, in any condition. The only variable is the spacing, which decides whether the row is initialised in mutual lateral contact (0.055 m, below one 0.06 m diameter) or never touches at all (0.070 m, every contact body-to-floor).

| bodies | spacing | mutually touching | peak contacts | GPU reproducible | CPU reproducible |
|---:|---:|---|---:|---|---|
| 2 | 0.070 | no | 4 | true | true |
| 8 | 0.070 | no | 16 | true | true |
| 16 | 0.070 | no | 32 | true | true |
| **32** | **0.070** | **no** | **64** | **true** | true |
| 2 | 0.055 | yes | 4 | true | true |
| 4 | 0.055 | yes | 10 | true | true |
| 5 | 0.055 | yes | 12 | true | true |
| **6** | **0.055** | **yes** | **14** | **false** | true |
| 8 | 0.055 | yes | 20 | false | true |
| 32 | 0.055 | yes | 122 | false | true |

The two bolded rows are the result. **64 contacts spread across 32 bodies that touch only the floor reproduce exactly. 14 contacts among 6 bodies that touch each other do not.**[^2]

Where it fails it fails completely: twelve runs of the same rollout in one process gave **twelve distinct** state digests. Where it holds, all twelve agreed.

Re-measured through this package's own sweep — same scene family, same trial design, but with the adapter's per-world seeded perturbation applied — the separated family still reproduces at every size up to 32, and the touching family's boundary sits one body lower.[^3]

## Contact count is not the variable

It cannot be — the reproducible condition has more than four times the contacts of the failing one. What separates them is what the contacts connect.

A body resting on the floor produces a constraint between that body's degrees of freedom and the world. Thirty-two such bodies produce thirty-two constraint groups that share nothing, and a solver can process them in any order and get the same answer. Two bodies resting against each other produce a constraint that writes into both, and once a chain of them shares degrees of freedom the accumulation order starts to matter.

That is consistent with the atomics the documentation names as the cause: contention requires two contacts writing to the same place, and body-to-world contacts never do.

## A threshold, not a gradient — but not a constant either

The transition is sharp. At each size, twelve runs either all agree or all differ; there is no band of sizes where disagreement is occasional.

**Where the transition sits depends on the setup.** Two harnesses measured this scene family and put the boundary one body apart:

| harness | initial conditions | last reproducing | first failing |
|---|---|---:|---:|
| standalone script | identical worlds, no perturbation | 5 | 6 |
| this package's sweep | per-world seeded perturbation of 0.01 | 4 | 5 |

Both are correct measurements of what they measured. The difference is the perturbation: the package's adapter offsets each world from the seed, so the worlds are not clones and the solve is not the same solve.

That makes "five" and "six" properties of a configuration rather than of MuJoCo-Warp, and this page does not claim a universal number. What survives both harnesses is the shape: a handful of mutually-touching bodies is enough, and it is far below the contact count that a floor-only scene sustains without ever failing.

Why the boundary sits in the region of five or six at all is not established. It is the sort of number that suggests a scheduling or block-size boundary rather than anything physical, but this measurement does not inspect a kernel and does not claim to know.

## What this corrects

An earlier version of this page concluded the variable was **stacking**, from a sweep that varied lattice width. That was wrong, and wrong in an instructive way: piling bodies up was the only way that scene generator had of making bodies touch each other, so "stacks" and "mutually-contacting clusters" moved together and the sweep could not tell them apart.

The flat-but-touching condition is what separated them — a single layer, nothing resting on anything, and it fails. The original comparison was controlled for contact count and body count, and still not controlled for the thing that mattered.

## Practical consequences

- **Flat-terrain locomotion will not see this.** Feet contact the ground, not each other.
- **Manipulation, clutter, granular media and any pile will.** Six mutually touching objects is a modest tabletop scene.
- **Scene size is not a useful proxy.** Thirty-two scattered objects reproduce; six touching ones do not.
- **CPU remains a valid oracle.** It reproduced in every condition tested — subject to the separate cross-backend problem in [[backend-divergence-begins-at-first-contact]].

## What this does not establish

One GPU, one solver configuration, one object shape, one timestep. Sphere-sphere contact only; whether meshes or boxes shift the boundary is unmeasured.

The connected-cluster size was controlled at initialisation, not verified at every step — spheres initialised at 0.055 m are slightly interpenetrating and push apart, so the row's contact graph during settling was not tracked. The measured quantity is the initial configuration and the peak contact count, and the claim is stated over those.

Constraint-island fields (`nisland`, `island_nefc`) were read and found to be zero on the GPU and uninitialised on the CPU, so island discovery is evidently not enabled in this configuration. No island quantity is reported here, because none was measured.

[^1]: MJWarp documentation, determinism section — "There may be ordering or _small_ numerical differences between results computed by different executions of the same code", with the advice to "Set device to CPU with `wp.set_device(\"cpu\")` for deterministic results".
[^2]: `[observed]` — for each row, twelve rollouts of 150 steps were run in one process and their per-step `qpos` digests compared. Peak contacts is the maximum of `Data.nacon` over the rollout, summed across the two worlds. The CPU column was produced by the same script under `wp.set_device("cpu")`.
[^3]: `[observed]` — `navprobe.sweep.run_scene_sweep` over `navprobe.scenes.row_scene` at body counts 2, 4, 5, 6, 8, 16 and 32, driven by `navprobe.adapters.mjx_warp_state`, with `TrialSpec(seed=7, step_count=150, repetitions=12)`, `world_count=2` and `perturbation=0.01`. The separated family (`spacing=0.070`) reported `deterministic=true` at every size and `first_irreproducible` of none; the touching family (`spacing=0.055`) reported true at 2 and 4 and false from 5 upward.
