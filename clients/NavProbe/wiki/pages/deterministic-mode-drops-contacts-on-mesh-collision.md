---
title: Deterministic mode drops every contact on a mesh-geom collision, and at adequate capacity it does so reproducibly
tags: [warp, determinism, contacts, finding, upstream, correctness]
related: ["[[tactile-alias-is-inert-with-live-taxels]]", "[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[a-determinism-verdict-needs-a-correctness-oracle]]", "[[gpu-nondeterminism-amplifies-to-macroscopic-scale]]"]
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-29
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, host austinpc)
  model: mujoco_warp/_src/sensor_test.py::test_tactile_sensor_geom_deduplication
  collision_path: mesh geom vs box, measured with multiccd BOTH enabled and disabled
  modes: [NOT_GUARANTEED, RUN_TO_RUN]
  deterministic_max_records: [64, 4096]
  step_count: 30
  repetitions: 8
  patch: the alias patch, applied (RUN_TO_RUN does not compile without it)
hubs: [determinism-measurement]
---

# Deterministic mode drops every contact on a mesh-geom collision, and at adequate capacity it does so reproducibly

Once [[tactile-alias-is-inert-with-live-taxels]] cleared the patch of changing any numbers, the tactile fixture became usable as a probe of the mode itself. Under `RUN_TO_RUN` on `cuda:0`, **MuJoCo-Warp stops generating contacts and the body falls through the geom it should rest on.** No exception is raised and the process exits 0.

## The measurement

Thirty steps, eight repetitions, contact count read per step and `qpos` read at the end — so "the physics diverged" is measured rather than inferred from a sensor reading:[^1]

| arm | physics digests | contacts over 30 steps | final z |
|---|---|---|---|
| `NOT_GUARANTEED` | 1 / 8 | 2 on every step (total 60) | 1.149779 |
| `RUN_TO_RUN`, `max_records = 64` | **4 / 8** | 0 on five reps; 3, 21, 28 on the others | 0.981754 / 1.054035 / 1.137341 / 1.126389 |
| `RUN_TO_RUN`, `max_records = 4096` | **1 / 8** | **0 on all eight reps** | **0.981754 on all eight** |

## Read the height, not the digest

The body starts at `z = 1.0`, interpenetrating the box. In the default mode it is pushed **up** out of that penetration and settles at 1.149779, in contact for all thirty steps. In the dead repetitions it ends at **0.981754 — below where it began**: it accelerated downward under gravity for thirty steps because collision detection produced nothing to stop it.

That is what makes this a correctness failure rather than a reproducibility one. A zeroed tactile reading is the *symptom*; the sensor was reporting a world with nothing in it, faithfully.

## Capacity converts a wrong answer into a reproducible wrong answer

The only difference between the second and third rows is `wp.config.deterministic_max_records`. Raising it from 64 to 4096 collapsed four distinct digests to one — and every one of those eight identical runs has zero contacts. **The capacity knob bought reproducibility and bought nothing else.**

So 64 — the value [[tactile-alias-patch-clears-warp-deterministic-compile]] found sufficient for the sphere-lattice family — is model-dependent and does not generalise, and a buffer too small for a given model does not announce itself here. Warp's own overflow guard is skipped while a CUDA stream is capturing,[^2] and MuJoCo-Warp captures.[^3] Neither arm raised the `RuntimeError` that the same overflow produces when it is caught.

## It is specific to this collision path

The failure does **not** generalise to the scene family this instrument normally measures. Run on NavProbe's own ten scenes — spheres on a plane inside walls — contacts survive the mode intact: contact totals are identical between default and `RUN_TO_RUN` on nine of ten scenes, and final heights agree to five decimals.[^4] That result is on its own page: [[a-determinism-verdict-needs-a-correctness-oracle]].

What differs here is the collision path: the tactile fixture is a **mesh** geom against a box, where the sphere family is primitive-vs-plane and primitive-vs-primitive.

## `multiccd` is not the cause

The fixture enables `<flag multiccd="enable"/>`, which made the multi-contact CCD routine the obvious suspect. It is innocent. Rebuilding the identical model with the flag removed changes nothing in either mode:[^5]

| `multiccd` | mode | contacts | final z | tactile |
|---|---|---:|---:|---:|
| off | default | 60 | 1.149779 | 0.005755194 |
| **off** | **`RUN_TO_RUN`** | **0** | **0.981754** | **0** |
| on | default | 60 | 1.149779 | 0.005755194 |
| on | `RUN_TO_RUN` | 0 | 0.981754 | 0 |

Both default-mode arms are identical to each other and both deterministic arms are identical to each other, to six decimal places, with one distinct digest from six repetitions in every cell. So the flag does not affect this scene at all, and removing it does not rescue the contacts. **The mesh collision path is what is implicated, and the bug report should say so rather than naming `multiccd`.**

## What this does not establish

Which kernel drops the contacts. The measurement locates the failure at contact *generation* — `d.nacon` is zero from the first step — but does not attribute it to a named routine or to the deterministic lowering of a specific atomic. The next cut is a build that leaves `collision_driver`'s kernels on ordinary atomics while the rest of the pipeline stays deterministic, which would localise it to a module.

Nor is `_preprocess_tactile_contacts` implicated, despite being the most fragile construct in the sensor pipeline — it consumes an `atomic_add` return value as a slot index, which routes through the two-pass counter-replay lowering rather than the simpler scatter path. It runs downstream of `d.nacon`, which is already zero, so it is a bystander here.

[^1]: `[observed]` — the vendor fixture (see [[tactile-alias-is-inert-with-live-taxels]] note 1), alias patch applied, `cuda:0`, 8 repetitions × 30 `mjw.step` calls, each repetition rebuilding `Data` from keyframe 0. Per step: `d.nacon` read, `d.qpos` digested via `digest_step`; per repetition folded with `digest_run`. Fresh `kernel_cache_dir` per arm.
[^2]: warp-lang 1.16.0, `warp/_src/deterministic.py` L2338 and L2438 — "if overflow_buf is not None and **not stream_is_capturing** and int(overflow_buf.numpy()[0]) != 0: raise RuntimeError(...)".
[^3]: mujoco-warp 3.11.0 — `wp.capture` / `ScopedCapture` appear in `mujoco_warp/_src/solver.py` and `mujoco_warp/_src/collision_driver.py`.
[^4]: `[observed]` — `navprobe.scenes.row_scene(n, spacing, 0.03, 0.005)` for (2,8,16,32)×0.070 and (2,4,5,6,8,32)×0.055, 3 repetitions × 60 steps, `cuda:0`, patched; default mode versus `RUN_TO_RUN` at `deterministic_max_records = 4096`.
[^5]: `[observed]` — the same fixture rebuilt with and without the `<option><flag multiccd="enable"/></option>` block, 6 repetitions × 30 steps per cell, `cuda:0`, patched, `deterministic_max_records = 4096`. `mjd.ncon` at the keyframe is 2 in both builds. All four cells report one distinct physics digest from six repetitions.
