---
title: A determinism verdict cannot tell "reproducible" from "reproducibly broken", and the ten-scene table was checked against that
tags: [instrument-design, determinism, methodology, finding]
related: ["[[deterministic-mode-drops-contacts-on-mesh-collision]]", "[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[passing-test-can-miss-its-own-premise]]", "[[bit-equality-is-a-leading-indicator]]"]
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
  scenes: row_scene(n, spacing, 0.03, 0.005) for (2,8,16,32)x0.070 and (2,4,5,6,8,32)x0.055
  modes: [NOT_GUARANTEED, RUN_TO_RUN]
  deterministic_max_records: 4096
  step_count: 60
  repetitions: 3
  patch: the alias patch, applied
hubs: [determinism-measurement, instrument-design]
---

# A determinism verdict cannot tell "reproducible" from "reproducibly broken", and the ten-scene table was checked against that

This instrument answers one question: given a fixed seed and a fixed action sequence, do repeated rollouts produce the same bytes? `comparison.py` folds two run records into agree-or-disagree; `experiment.py` reports whether every repetition matched the first. **Nothing in that chain ever asks whether the trajectory was physically right.**

That is a design choice, and mostly a good one — an instrument that needed a physics oracle would need a reference implementation and would stop being vendor-agnostic. But it has a failure mode with teeth, and 2026-08-29 produced a live instance of it: on the vendor tactile fixture, `RUN_TO_RUN` at adequate buffer capacity returns **one distinct digest from eight repetitions while generating zero contacts**, the body falling through the geom every time, identically ([[deterministic-mode-drops-contacts-on-mesh-collision]]).

Scored by this instrument, that run is `deterministic: true`. It is also completely wrong. **Reproducible and correct are different properties, and only the first one is measured.**

## Which puts a question to this wiki's own headline

[[tactile-alias-patch-clears-warp-deterministic-compile]] reports ten scenes `deterministic: true` under `RUN_TO_RUN` and reads that as "GPU determinism is now a setting". Every one of those verdicts came from comparing repetitions to each other. A mode that silently dropped every contact would have produced exactly that table.

So the table was re-run with a contact check attached.

## The table survives, and is now better evidenced than when it was written

Same ten scenes, contact counts and final sphere heights recorded alongside the digest:[^1]

| bodies | spacing | default: det / contacts | `RUN_TO_RUN`: det / contacts |
|---:|---:|---|---|
| 2 | 0.070 | true / 66 | true / 66 |
| 8 | 0.070 | true / 264 | true / 264 |
| 16 | 0.070 | true / 528 | true / 528 |
| 32 | 0.070 | true / 1056 | true / 1056 |
| 2 | 0.055 | true / 77 | true / 77 |
| 4 | 0.055 | true / 168 | true / 168 |
| 5 | 0.055 | **false** / 213 | true / 213 |
| 6 | 0.055 | **false** / 266 | true / 266 |
| 8 | 0.055 | **false** / 388 | true / 388 |
| 32 | 0.055 | **false** / 2249 | true / 2262 |

Three things at once. The default-mode boundary replicates exactly as published — separated family reproducible throughout, touching family irreproducible from five bodies upward. `RUN_TO_RUN` is deterministic on all ten. And **the contacts survive**: totals identical between modes on nine of ten scenes, minimum heights agreeing to five decimals. The 32×0.055 row differs (2249 vs 2262) for the expected reason — default mode is irreproducible there, so its count is one sample of a spread rather than a fixed reference.

No retraction is owed. The claim stands, and now rests on a check its original run did not make.

## What to do with this

The lesson is not "add a physics oracle" — that would cost the vendor-agnosticism the design is built on. It is narrower and cheaper: **a determinism verdict should carry a liveness witness from the scene it was measured on.** Contact count is the obvious one here, costs one array read per step, and would have flagged the tactile fixture immediately while leaving the ten scenes untouched. A verdict of `deterministic: true` on a scene that recorded zero contacts is a result to distrust, and nothing currently records enough to notice.

## What this does not establish

That contact count is sufficient. It catches "the scene stopped interacting", which is the failure observed; it would not catch a solve that produced contacts of the wrong magnitude, or a rendered stream that reproduced while being wrong. Whether a general liveness witness exists, or whether it has to be chosen per scene family, is open.

[^1]: `[observed]` — `navprobe.scenes.row_scene(n, spacing, 0.03, 0.005)` built through `build_scene`, driven with `mjw.step` on `cuda:0`, alias patch applied, 3 repetitions × 60 steps per scene, `deterministic_max_records = 4096` in the deterministic arm. `d.nacon` read per step; per-body heights read from `qpos` at the end; per-step `qpos` digested with `digest_step` and folded with `digest_run`. Spheres drop from `DROP_HEIGHT` and land within roughly 31 steps, so 60 steps covers flight and rest.
