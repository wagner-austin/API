---
title: Deterministic mode drops every contact in the convex narrowphase, and MuJoCo-Warp's own dispatch table predicts which pairs
tags: [warp, determinism, contacts, collision, finding, upstream, correctness]
related: ["[[tactile-alias-is-inert-with-live-taxels]]", "[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[a-determinism-verdict-needs-a-correctness-oracle]]", "[[gpu-nondeterminism-amplifies-to-macroscopic-scale]]"]
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-30
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, host austinpc)
  modes: [NOT_GUARANTEED, RUN_TO_RUN]
  deterministic_max_records: 4096
  step_count: 40 (20 for the overflow probe)
  repetitions: 4
  patch: the alias patch, applied -- RUN_TO_RUN does not compile without it
  geometry_pairs: [mesh_box, mesh_plane, prim_box, prim_plane, mesh_meshbox, box_box]
hubs: [determinism-measurement]
---

# Deterministic mode drops every contact in the convex narrowphase, and MuJoCo-Warp's own dispatch table predicts which pairs

Under `RUN_TO_RUN` on `cuda:0`, MuJoCo-Warp stops generating contacts for any geometry pair that routes to `CollisionType.CONVEX`, and bodies fall through the geometry they should rest on. No exception, no warning, exit 0. Pairs that route to `CollisionType.PRIMITIVE` are completely unaffected.

## The dispatch table is the predictor

`MJ_COLLISION_TABLE` sends each geom-type pair to one of two narrowphases.[^1] That column sorts every measured cell, six for six:[^2]

| pair | table entry | routes to | default | `RUN_TO_RUN` |
|---|---|---|---:|---:|
| sphere on plane | `(PLANE, SPHERE)` | `PRIMITIVE` | 40 | **40** |
| sphere on box | `(SPHERE, BOX)` | `PRIMITIVE` | 37 | **37** |
| mesh on plane | `(PLANE, MESH)` | `PRIMITIVE` | 58 | **58** |
| mesh on box | `(BOX, MESH)` | **`CONVEX`** | 62 | **0** |
| mesh on mesh | `(MESH, MESH)` | **`CONVEX`** | 41 | **0** |
| **box on box** | `(BOX, BOX)` | **`CONVEX`** | 160 | **0** |

Every `PRIMITIVE` row reproduces its default-mode trajectory exactly. Every `CONVEX` row goes to zero.

**It is not about meshes.** A mesh on a plane keeps all 58 contacts, because `(PLANE, MESH)` is a primitive pair. Two primitive boxes lose all 160, because `(BOX, BOX)` is a convex pair. Earlier revisions of this page blamed `multiccd`, then buffer capacity, then meshes; each was excluded by measurement before this one survived.

## The minimal repro carries no mesh at all

Box on box is the useful one for a bug report: no mesh asset, no plugin, no sensor. A falling `.3 .3 .3` box on a static `.7 .7 .3` box, started at `z = 0.58` so it seats in contact (`ncon = 4` at the keyframe), stepped 40 times, 4 repetitions. Default mode holds contact on every step and settles at `z = 0.598057`. `RUN_TO_RUN` records **zero contacts across all 40 steps** and ends at `0.547823` — below where it started, having accelerated under gravity through the box.

## Broadphase is fine; the failure is strictly narrowphase

Reading `d.ncollision` (the broadphase pair count) beside `d.nacon` separates the two stages:[^3]

| mode | broadphase pairs | contacts | `d.overflow` |
|---|---:|---:|---|
| default | 20 | 40 | none |
| `RUN_TO_RUN` | **20** | **0** | none |

Identical candidate pairs, zero contacts out. Broadphase does its job and the narrowphase discards everything it was handed.

## The mechanism, and why no diagnostic fires

`collision_convex.py` allocates CCD work with a **consumed-return** atomic — the return value is used as a slot index:[^4]

```python
ccdid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)
if ccdid >= naccdmax_in:
    ...
    wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.CCD))
    return
```

A consumed return routes through Warp's two-pass counter-replay lowering rather than the simpler scatter path — and the primitive narrowphase never uses that queue, which is exactly the split the table above measures. Warp's macro returns **zero** when a target's counter helper is absent:[^5]

```c
(out) = ((helper).count != nullptr) ? counter_add(...) : 0;
```

`ccdid = 0` does not trip `>= naccdmax_in`, so **no early return is taken and no `CCD` overflow bit is set** — confirmed above, `d.overflow` is clean in both modes. `nccd` simply never increments, and the CCD kernel is launched over zero pairs. The silence is structural: the only diagnostic on this path is attached to the branch that is not taken.

## What this does not establish

That the counter helper is genuinely absent rather than present-but-wrong — that reading is inferred from the macro plus the absent overflow bit, not measured inside the generated code. Nor is the fix known: whether the lowering can wire the helper for this target, or whether MJWarp should avoid a consumed-return atomic here, is upstream's call. Measured on one card, one Warp version, one MJWarp version. `GPU_TO_GPU` was measured on the mesh-on-box pair only, where it fails identically.

[^1]: mujoco-warp 3.11.0, `mujoco_warp/_src/collision_driver.py` L52-80 — `MJ_COLLISION_TABLE`, with `_narrowphase` at L866 splitting it into `convex_pairs` and `primitive_pairs`. `(BOX, BOX)` at L78 carries the comment "overwritten by NATIVECCD disable flag".
[^2]: `[observed]` — each pair built as MJCF and driven with `mjw.step` on `cuda:0`, alias patch applied, 4 repetitions x 40 steps, `deterministic_max_records = 4096`, fresh `kernel_cache_dir` per arm; `d.nacon` read per step and `d.qpos` at the end. Every pair carries a default-mode control, because a pair making no contact in EITHER mode indicts the scene rather than the mode — which caught two badly-built scenes during this work.
[^3]: `[observed]` — mesh-on-box, 20 steps, both modes: `d.ncollision` summed per step is 20 in each, `d.nacon` summed is 40 under default and 0 under `RUN_TO_RUN`, and `d.overflow` (`types.py` L2350, decoded against `OverflowType`) is 0 in both.
[^4]: mujoco-warp 3.11.0, `mujoco_warp/_src/collision_convex.py` L288 (and the same shape at L819).
[^5]: warp-lang 1.16.0, `warp/native/deterministic.h` L297-302 — the `WP_DET_COUNTER_OR_FALLBACK` macro's CUDA arm.
