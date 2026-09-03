---
title: Deterministic mode drops every contact in the convex narrowphase, and MuJoCo-Warp's own dispatch table predicts which pairs
tags: [warp, determinism, contacts, collision, finding, upstream, correctness]
related: ["[[tactile-alias-is-inert-with-live-taxels]]", "[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[a-determinism-verdict-needs-a-correctness-oracle]]", "[[gpu-nondeterminism-amplifies-to-macroscopic-scale]]", "[[kernel-split-fix-restores-convex-contacts-at-upstream-head]]", "[[heightfield-narrowphase-shares-the-contact-drop]]"]
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
  - "NVIDIA Warp 1.16 user guide, Deterministic Execution"
  - "google-deepmind/mujoco_warp PRs #1422, #1425, #1533 (all open, unmerged)"
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

Since this page was written the defect has been reproduced at upstream HEAD `3879591` on warp
1.15.0 — upstream's own version floor — and fixed there on a local branch
([[kernel-split-fix-restores-convex-contacts-at-upstream-head]]). The separate heightfield
CCD kernel shares the defect and the fix ([[heightfield-narrowphase-shares-the-contact-drop]]).
This page remains the record of the drop as shipped in 3.11.0/warp 1.16.

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

## Where the failure sits, and one falsified explanation

`collision_convex.py` allocates CCD work with a **consumed-return** atomic — the return value is used as a slot index:[^4]

```python
ccdid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)
if ccdid >= naccdmax_in:
    ...
    wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.CCD))
    return
```

That is Warp's "Pattern 2: Slot Allocation", and it routes through a two-pass counting-and-replay lowering rather than the simpler scatter path — a path the primitive narrowphase never uses, which is exactly the split the table above measures. So the convex/primitive divide is well explained by *which lowering runs*. What that lowering does wrong is not.

**A first explanation was published here and is now withdrawn.** It read Warp's macro — `(out) = ((helper).count != nullptr) ? counter_add(...) : 0;`[^5] — as meaning an unwired counter helper returns zero, so `ccdid = 0` never trips `>= naccdmax_in`, no overflow bit is set, and `nccd` never increments. It fit the evidence, and it is not supported by the vendor's own contract: Warp documents Pattern 2 as **supported** for precisely this form — an `int32` counter, a data-dependent index, and sliced counter views like `wp.atomic_add(counters[world], 0, 1)`.[^6] Reading a null-check branch as the normal path was speculation presented as a mechanism.

## The mechanism, measured at every link

The failure is a **contact-slot reservation whose condition depends on scratch-array writes that Warp suppresses during its counting pass** — the limitation Warp documents almost verbatim.[^6] Every stage was measured rather than reasoned about, and only the last one differs between modes:[^9]

| stage | what it reports | default | `RUN_TO_RUN` |
|---|---|---:|---:|
| broadphase | `d.ncollision` | 1 | **1** |
| CCD queueing | `nccd` (internal) | 1 @ slot 51 | **1 @ slot 51** |
| EPA result | `dist` at `write_contact` | −0.020000042 | **−0.020000042** |
| contact write | `d.nacon` | 4 | **0** |

Broadphase agrees. The CCD slot reservation agrees. **The EPA distance agrees bit for bit** — a real penetrating contact, well inside margin. And then no contact is written.

**Three separate gates make the reservation count scratch-dependent, and all three must be neutralised.** Each was found by probing, not by reading: a reservation planted at a line counts only if the counting pass reaches it, so walking a probe down the kernel locates the first gate exactly.[^12] In execution order:

1. **The kernel exits early on the EPA distance.** In `ccd_kernel`, a few lines after the `ccdid` reservation:[^4]
   ```python
   dist, ncollision, w1, w2, multiccd_idx = epa_phase(...)
   if dist >= gap and not is_collision_sensor:
     return
   ```
   `dist` is the EPA result. On the counting pass EPA runs against suppressed scratch, `dist` is meaningless, and the thread **leaves the kernel** — reaching neither of the gates below. This is the blocking one.
2. **The contact loop's trip count is the EPA contact count.** `for i in range(ncollision)` at the `write_contact` call site, where `ncollision` comes from `epa_phase` and `multicontact`.
3. **The writer's own reservation is conditional.** `write_contact` returns before `cid = wp.atomic_add(nacon_out, 0, 1)` when `detected` is false, and `detected` derives from the same `dist`.[^10]

A probe reservation planted immediately after the `ccdid` line counts correctly in **both** modes — `nacon` moves 4→5 in default and 0→1 under `RUN_TO_RUN`.[^12] So `nacon_out` reserves fine there, and a second consumed-return counter in one kernel is not the problem; the failure is purely that gate 1 stops the counting pass from ever arriving.

This explains every earlier observation, including the ones that killed the other hypotheses. `nccd` is correct because it is reserved *before* anything touches scratch. Primitive pairs are immune because their narrowphase computes `dist` in registers with closed-form geometry and never uses a scratch workspace. And nothing is raised because this is a correctness fault, not a capacity one — the buffers were never overrun.

## A prototype fix restores the contacts

Neutralising all three gates makes the reservation count scratch-independent and brings every convex pair back, at a measured cost in over-reservation, primitive-path side effects and throughput: [[prototype-fix-restores-convex-contacts-under-deterministic-mode]].

## Superseded candidates

**One documented limitation was excluded by measurement before the above was found.**

- **Deterministic kernels are unsupported inside conditional body graphs — EXCLUDED.** Warp states they "are not supported inside CUDA conditional body graphs, such as `wp.capture_while()` or `wp.capture_if()`",[^6] and MuJoCo-Warp's solver calls `wp.capture_while` directly.[^7] It is gated on `m.opt.graph_conditional`, whose else-branch is an ordinary Python loop over the same iteration kernel — so the limitation can simply be switched off. Switched off, **the contacts do not come back**: box-on-box under `RUN_TO_RUN` reports zero contacts and a final height of 0.547823 with the flag both on and off, while both default-mode arms hold 160 contacts either way.[^8] The conditional-graph path is not what breaks this.
Four further explanations were proposed and killed by measurement before the mechanism above was found, and they are recorded so nobody re-runs them: `multiccd` (removing the flag changes nothing), buffer capacity (4096 makes the wrong answer *reproducible*, not correct), meshes (mesh-on-plane is a primitive pair and is clean), and the CCD slot atomic itself (`nccd` is identical in both modes). A fifth — that Warp's Pattern 2 lowering is simply broken for this usage — was excluded by isolating it: a standalone kernel reproducing MuJoCo-Warp's exact shape, an early `return` before a consumed-return `wp.atomic_add` with a `wp.static(...)` index into a multi-element `int32` counter, allocates correctly in both modes.[^11]

The silence has a documented cause too, and it is not a defect: "Overflow checks are disabled during graph capture and replay to keep graph launches asynchronous."[^6] So a truncated or mis-scanned deterministic buffer under capture produces wrong numbers with no exception by design, and `d.overflow` reading clean proves less than it appears to.

## What this does not establish

**The fix.** The mechanism is measured; the remedy is not designed. Warp's guidance is to make the slot decision independent of suppressed writes — "use local variables or input arrays for that decision"[^6] — but in this pipeline the decision genuinely depends on a value EPA can only produce by iterating a workspace. Reserving unconditionally and discarding unused slots would decouple it at the cost of over-allocating `naconmax`; hoisting the detection into a prior launch would cost a kernel boundary. Which is right is MuJoCo-Warp's call, and neither has been prototyped here.

**Whether upstream considers this in scope at all.** MuJoCo-Warp is building its own `opt.deterministic` rather than relying on Warp's codegen mode ([[open-questions-and-what-would-answer-them]] question 1), so "Warp's deterministic mode silently drops contacts" may be answered with "that mode is not the supported path". The counter-argument is that nothing warns: a user who enables it gets bodies falling through geometry with exit 0.

**Generality.** One card, one Warp version, one MuJoCo-Warp version, and one convex pair traced end to end. The other two convex pairs were measured at the endpoints only.

Nor does it establish that this is a Warp *defect* at all. On the current reading MuJoCo-Warp may simply be using Warp's deterministic mode in configurations Warp documents as unsupported, which is a different report with a different audience. Upstream is in any case building its own `opt.deterministic` at the MuJoCo-Warp level rather than relying on Warp's codegen mode ([[open-questions-and-what-would-answer-them]] question 1), so the practical value of unblocking the codegen path is smaller than it first appeared.

Measured on one card, one Warp version, one MuJoCo-Warp version. `GPU_TO_GPU` was measured on the mesh-on-box pair only, where it fails identically.

[^1]: mujoco-warp 3.11.0, `mujoco_warp/_src/collision_driver.py` L52-80 — `MJ_COLLISION_TABLE`, with `_narrowphase` at L866 splitting it into `convex_pairs` and `primitive_pairs`. `(BOX, BOX)` at L78 carries the comment "overwritten by NATIVECCD disable flag".
[^2]: `[observed]` — each pair built as MJCF and driven with `mjw.step` on `cuda:0`, alias patch applied, 4 repetitions x 40 steps, `deterministic_max_records = 4096`, fresh `kernel_cache_dir` per arm; `d.nacon` read per step and `d.qpos` at the end. Every pair carries a default-mode control, because a pair making no contact in EITHER mode indicts the scene rather than the mode — which caught two badly-built scenes during this work. The sweep is now a repo script rather than a session recipe: `scripts/collision_pair_probe.py` (blob `26d6b854`, commit `32f1af0f`), run as `python -m scripts.collision_pair_probe <MODE> <CACHE_DIR> [MAX_RECORDS]`.
[^3]: `[observed]` — mesh-on-box, 20 steps, both modes: `d.ncollision` summed per step is 20 in each, `d.nacon` summed is 40 under default and 0 under `RUN_TO_RUN`, and `d.overflow` (`types.py` L2350, decoded against `OverflowType`) is 0 in both.
[^4]: mujoco-warp 3.11.0, `mujoco_warp/_src/collision_convex.py` **L819**, inside `ccd_kernel` (built by `ccd_kernel_builder`, L720) — with the EPA early-return a few lines below it. **An earlier revision of this page cited L288, which is wrong**: that site is in `ccd_hfield_kernel` (L176), the heightfield path, which never executes for a box/box pair. Caught by falsification rather than review — a probe reservation planted at L288 left default-mode `nacon` unchanged at 4, proving the line never ran. The executing module is `ccd_kernel_builder__locals__ccd_kernel_903e8811`, visible in Warp's compile log from the first run. Note also that L819 indexes its counter with a runtime `geomgeomid`, where L288 uses `wp.static(...)`.
[^5]: warp-lang 1.16.0, `warp/native/deterministic.h` L297-302 — the `WP_DET_COUNTER_OR_FALLBACK` macro's CUDA arm. Cited for the withdrawn explanation, not for a live claim.
[^6]: NVIDIA Warp 1.16 user guide, "Deterministic Execution" (`https://nvidia.github.io/warp/v1.16/user_guide/deterministic_execution.html`), sections "What Warp Supports" → Pattern 2: Slot Allocation ("`slot = wp.atomic_add(counter, index, value)`; counter must be an int32 array; index may be constant or data-dependent; Sliced counter views"), "Limitations" → "Side effects in the counting pass", and "CUDA Graph Capture" ("Deterministic kernels are not supported inside CUDA conditional body graphs" and "Overflow checks are disabled during graph capture and replay").
[^7]: mujoco-warp 3.11.0, `mujoco_warp/_src/solver.py` L3984-3998 — `if m.opt.iterations != 0 and m.opt.graph_conditional:` guarding `wp.capture_while(nsolving, while_body=_solver_iteration, ...)`, with an ordinary `for _ in range(m.opt.iterations)` loop as the else-branch.
[^8]: `[observed]` — box-on-box at `start_z = 0.58`, `cuda:0`, alias patch applied, 3 repetitions x 40 steps, `deterministic_max_records = 4096`, with `m.opt` replaced via `dataclasses.replace(m.opt, graph_conditional=...)` after `put_model`. Default mode: 160 contacts and final z 0.598057 with the flag on AND off. `RUN_TO_RUN`: 0 contacts and final z 0.547823 with the flag on AND off. **The branch flip was verified rather than assumed** — wrapping `wp.capture_while` in a counting spy over 10 steps records 10 calls with the flag on and 0 with it off, so the null result is a real negative and not a silently ineffective setting. That check exists because an earlier arm in this investigation set a module option that never applied and produced a clean-looking null.
[^9]: `[observed]` — box-on-box at `start_z = 0.58`, `cuda:0`, alias patch applied, 3 steps, `deterministic_max_records = 4096`. `d.ncollision` and `d.nacon` read directly. `nccd` is a local inside `convex_narrowphase()` and is not surfaced on `Data`, so it was exposed by a reversible append to that function capturing `nccd.numpy()`; it reads `1` at counter index 51 in both modes. The EPA distance was exposed by a one-line reversible insert in `write_contact` writing `dist_in` into `contact_dist_out[0]` before the reservation — slot 0 being free precisely because nothing is written in the failing mode. Both patches reverted and the venv verified canonical afterwards.
[^10]: mujoco-warp 3.11.0, `mujoco_warp/_src/collision_core.py` L214-268 — `write_contact`, whose `detected` guard returns before `cid = wp.atomic_add(nacon_out, 0, 1)`. Called from the convex path at `collision_convex.py` L512, L569, L624, L680, L921 and L1085, and from `collision_primitive.py` L318 and L389 — the same writer both narrowphases use, which is why a writer-level defect is excluded: primitive pairs write through it correctly.
[^11]: `[observed]` 2026-08-30 on warp-lang 1.16.0, `cuda:0`, RTX 3090 Ti, host `austinpc` — **a throwaway probe, not retained as a file**, stated so rather than cited to a path that does not exist; the kernel shapes below are the reproduction instructions. A standalone Warp program, no MuJoCo, four kernels isolating each feature of the CCD reservation: the documented example, an early `return` before the atomic, a `wp.static(...)` index into an 8-element `int32` counter, and both together. 1024 threads, half returning early. Under both `NOT_GUARANTEED` and `RUN_TO_RUN` every variant allocates exactly the expected count (1024, 512, 1024, 512) with matching non-zero outputs, so Pattern 2 is not broken for this shape in isolation. The caveat that keeps this from being a full exclusion: the isolated kernels sit alone in their own modules, while MuJoCo-Warp's sits in a large module beside many other deterministic targets.

