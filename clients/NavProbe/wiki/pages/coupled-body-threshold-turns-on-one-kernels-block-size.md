---
title: The coupled-body determinism threshold turns on one kernel's block size
tags: [warp, determinism, measurement, finding, gpu, scheduling, codegen, block-dim]
related: ["[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[coupled-body-threshold-does-not-move-with-sm-count]]", "[[open-questions-and-what-would-answer-them]]", "[[the-numbers-are-scene-dependent-the-shapes-replicate]]"]
source_paths:
  - "src/navprobe/sweep.py"
  - "src/navprobe/adapters/mjx_warp_state.py"
  - "src/navprobe/experiment.py"
  - "scripts/gpu_deterministic_sweep.py"
  - "scripts/arguments.py"
source_git_blobs:
  "src/navprobe/sweep.py": "3fbecbbf2caeec07827f618c307ded4699414dfd"
  "src/navprobe/adapters/mjx_warp_state.py": "a0446a0cbccdfdbedf908da593cec2bab86bc1b8"
  "src/navprobe/experiment.py": "c9b57616cfed8af956f6f1956fdd6ecec1eeff92"
  "scripts/gpu_deterministic_sweep.py": "b11833275121df70433903d3753e70adc00ff35e"
  "scripts/arguments.py": "e12af5e9749ebfeb2e13ef9ce2e19bc010cce2da"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-25
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  mode: NOT_GUARANTEED
  backend: warp cuda:0
  devices:
    - NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, driver 591.86, host austinpc)
  harness: navprobe.sweep.run_scene_sweep over navprobe.scenes.row_scene
  adapter: navprobe.adapters.mjx_warp_state
  seed: 7
  step_count: 150
  repetitions: 12
  world_count: 2
  perturbation: 0.01
  constraint_capacity: 8192
  scene: "5 touching bodies, spacing 0.055, radius 0.03, timestep 0.005"
  independent_trials_per_cell: "8 to 20"
  block_dim_field: "mujoco_warp Model.block_dim.linesearch_iterative"
hubs: [determinism-measurement]
---

# The coupled-body determinism threshold turns on one kernel's block size

[[coupled-body-threshold-does-not-move-with-sm-count]] falsified the occupancy
explanation and concluded that the boundary "is set by **what the contacts connect**,
not by how the resulting work is spread". The first half stands. **The second half does
not.** Setting one kernel's CUDA block size from its default of 32 to 64 makes the
5-body scene — which never reproduces at defaults — reproduce most of the time, while
computing a bit-identical trajectory.[^1][^2]

The field is `Model.block_dim.linesearch_iterative`. It is necessary and sufficient on
its own.[^3]

## The measurement

The boundary scene, 20 independent trials per setting, fresh kernel cache each — block
size participates in codegen, so a shared cache would let one setting load another's
compiled kernels.[^1]

| all 23 `block_dim` fields set to | 5-body scene reproduced |
|---|---|
| default (a 32/64/128 mix) | 0 / 20 |
| 32 | 0 / 20 |
| **64** | **17 / 20** |
| 128 | 0 / 20 |

Non-monotonic: 32 and 128 both sit at zero. This is not "more parallel width, more
disorder".

## It is one field, and it is not the one you would guess

`Model.block_dim` is a `BlockDim` struct of 23 per-kernel fields, not a scalar.[^4]
Seven already default to 64, so they cannot be the cause; the search space is the other
16. Group-halving, then leave-one-out, with a no-change control at 0/8:[^3]

| set to 64 | reproduced |
|---|---|
| nothing (control) | 0 / 8 |
| the four 128-default fields, incl. `update_gradient_JTDAJ_dense` and `_sparse` | 0 / 8 |
| the twelve 32-default fields | 5 / 8 |
| …minus `linesearch_iterative` | **0 / 8** |
| `linesearch_iterative` alone | **11 / 12** |

**The `JTDAJ` accumulation kernels are innocent.** J^T·D·A·J is the textbook
order-dependent reduction and the obvious suspect; setting both to 64 does nothing. This
is recorded because it is where anyone would look first.

Two fields modulate without causing: `energy_vel_kinetic` **suppresses** the effect
(`linesearch_iterative` + `energy_vel_kinetic` falls to 5/12), and `actuator_velocity`
raises it (12/12 in combination).[^3] Group-halving briefly appeared to show a
combinatorial cause for exactly this reason — one half contained the suppressor.

## Only 64, not "more than one warp"

The natural mechanism is that the default 32 equals the CUDA warp size, so a
single-warp block takes a different reduction path. **That is falsified.**[^5]

| `linesearch_iterative` | reproduced |
|---|---|
| 32 (default) | 0 / 12 |
| **64** | **7 / 12** |
| 96 | 0 / 12 |
| 128 | 0 / 12 |
| 256 | 0 / 12 |

Exactly two warps, and nothing else. No mechanism is offered here; this page reports the
fact and leaves the generated CUDA to the reader — which is what question 5 of
[[open-questions-and-what-would-answer-them]] was waiting for.

## The trajectory does not change

A `deterministic` verdict means every repetition matched the reference,[^6] not that the
answer is right — a setting that truncated work could make twelve runs agree on a stable
wrong answer, the failure mode question 7 documents. Comparing `reference_digest`:[^2]

| bodies | default | `linesearch_iterative=64` |
|---|---|---|
| 2 | `3849c2aa…` reproduces | `3849c2aa…` reproduces |
| 4 | `08b7c665…` reproduces | `08b7c665…` reproduces |
| 5 | `71f581c9…` **fails** | `51605250…` **reproduces** |

Bit-identical where both reproduce, and no capacity or overflow warnings in either.[^2]
The 5-body row is the informative one: under defaults that reference digest is itself
unstable between runs — `51605250…` in one check and `71f581c9…` in another[^2] — which
is the non-determinism restated. At 64 it stops moving and settles on a value the
default also produces. The computation is the same; only the accumulation order stops
wandering.

## What this does not establish

**The rate is not a constant.** The isolated field measured 92%, then 58%, in
independent sessions of the identical configuration.[^3][^5] The effect is solid; the
magnitude is not. Quote a spread, never a point.

Cell sizes are 8–20 trials — enough to separate 0% from 60–90%, nowhere near enough to
rank the graded leave-one-out figures against each other.

One host, one card, one session; not replicated on `sedona`, and no sm_75 device has run
any of it. Only the 5-body scene was tested against the isolated field — 6 and 8 bodies
were not.

## Consequence for the cross-architecture question

Question 2 of [[open-questions-and-what-would-answer-them]] compares sm_75 against
sm_86. On this evidence, **`linesearch_iterative` must be pinned explicitly on both
devices.** A default differing by architecture or tile constraint would produce a
"threshold moved" result that is block size and nothing to do with the architecture.
That confound did not previously exist in the plan because the axis had never been
measured.

The instrument now closes it rather than leaving it to discipline. `--linesearch-block-dim`
pins the value, and it is carried in `DeviceRunConditions` — so a sweep report states
which block size produced it, and two reports that pinned different values are visibly
incomparable instead of quietly so. The banners carrying those conditions moved to `/2`
in the same change: a `/1` document has one fewer header line, and a decoder that read it
anyway would misattribute every field after the third.

[^1]: `[observed]` — on host `austinpc`. Reproduce with `python -m scripts.gpu_deterministic_sweep NOT_GUARANTEED <FRESH_CACHE> --linesearch-block-dim <N> --device cuda:0`, repeated and counting the `bodies=5 spacing=0.055` verdict; a fresh cache directory per setting is required, because block size participates in codegen and a shared cache would let one setting load another's compiled kernels. Reported `reproduced=0/20`, `0/20`, `17/20`, `0/20` over 20 trials for default, 32, 64 and 128. **The original measurement predates the flag**: it was taken with standalone scripts reaching past the factory into `_model.block_dim`, by the same precedent as the `sm_occupancy_sweep.py` run behind [[coupled-body-threshold-does-not-move-with-sm-count]]. `linesearch_block_dim` was added to the adapter and the flag to the sweep in the same change that published this page, so the figures are reproducible from a checkout rather than from a directory that no longer exists.
[^2]: `[observed]` — `scripts/gpu_deterministic_sweep.py`, the same sweep and invocation as [^1] — `reference_digest` read off the `TrialRecord` for the same scenes under `--linesearch-block-dim 64` and under no flag, with captured vendor stdout and stderr scanned for `overflow`, `warning`, `exceeded`, `truncat` and `capacity`. No marker appeared on any run. The two differing default 5-body digests come from two separate invocations, which is the instability itself rather than a discrepancy between them.
[^3]: `[observed]` — `scripts/gpu_deterministic_sweep.py` again, with individual `BlockDim` fields set to 64 and the rest left at their defaults, 8 trials per set. A control changing nothing reported 0/8. Group-halving reported 0/8 for the four 128-default fields and 5/8 for the twelve 32-default ones; leave-one-out over the resulting six reported 0/8 for the set omitting `linesearch_iterative` and non-zero for all five other omissions. Per-field selection is not exposed as a flag — only `linesearch_iterative` is, at `scripts/arguments.py:111` (`split_linesearch_block_dim`), because it is the only field this instrument has a measured reason to pin.
[^4]: `mujoco-warp 3.11.0` — `mujoco_warp._src.types.BlockDim`, read off a `put_model` result built by `src/navprobe/adapters/mjx_warp_state.py L206`, which `L208` then pins. 23 integer fields; defaults 32 for `actuator_velocity`, `cholesky_factorize`, `cholesky_factorize_solve`, `contact_jac_tiled`, `energy_vel_kinetic`, `linesearch_iterative`, `qderiv_actuator_dense`, `solve_beta_accumulate`, `solve_init_search_cg`, `solve_search_update_cg`, `update_gradient_cholesky_blocked`, `update_gradient_grad`; 64 for `cholesky_solve`, `contact_sort`, `convex_ccd`, `ray`, `render`, `small_cholesky`, `update_gradient_cholesky`; 128 for `segmented_sort`, `solve_LD_sparse_fused`, `update_gradient_JTDAJ_dense`, `update_gradient_JTDAJ_sparse`. The default is asserted as a literal by `tests/adapters/test_mjx_warp_state.py::TestFactory::test_leaves_the_vendor_default_alone_when_not_given_one`, so a vendor bump that moves it fails rather than silently re-baselining this page.
[^5]: `[observed]` — `--linesearch-block-dim <V>` for V in 32, 64, 96, 128 and 256, 12 trials each, fresh cache per value. Reported `0/12`, `7/12`, `0/12`, `0/12`, `0/12`. Values are rejected below one by `src/navprobe/adapters/mjx_warp_state.py` (`NP-WSTATE-004`) and by `scripts/arguments.py L111` `split_linesearch_block_dim`, so a nonsensical block size stops before a cold compile rather than during codegen.
[^6]: `src/navprobe/experiment.py L178-179` — `reference_digest=reference["digest"]` and `deterministic=all(comparison["digests_match"] for comparison in comparisons)`.
