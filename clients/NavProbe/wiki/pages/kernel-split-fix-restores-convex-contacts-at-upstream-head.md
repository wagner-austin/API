---
title: A kernel split restores convex contacts exactly at upstream HEAD, and coupled scenes then reproduce bit for bit
tags: [warp, determinism, contacts, collision, fix, upstream]
related: ["[[deterministic-mode-drops-contacts-in-convex-narrowphase]]", "[[prototype-fix-restores-convex-contacts-under-deterministic-mode]]", "[[heightfield-narrowphase-shares-the-contact-drop]]", "[[warp-binds-determinism-mode-at-first-compile]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]"]
provenance:
  - "google-deepmind/mujoco_warp @ 3879591, branch deterministic-ccd (local, unpushed)"
  - "warp-lang 1.15.0 (the version floor in upstream's pyproject.toml)"
fact_checked: 2026-08-31
confidence: high
measured_with:
  package: mujoco_warp @ git 3879591 + working-tree fix
  warp: 1.15.0
  backend: warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, host austinpc)
  modes: [NOT_GUARANTEED, RUN_TO_RUN]
  deterministic_max_records: 64
  step_count: 40 (pair sweep), 150 (coupled pile)
  repetitions: 3-4 per arm; pile digests 6 rollouts across 2 processes
hubs: [determinism-measurement]
---

# A kernel split restores convex contacts exactly at upstream HEAD, and coupled scenes then reproduce bit for bit

[[deterministic-mode-drops-contacts-in-convex-narrowphase]] documented the drop on the shipped
3.11.0 package, and [[prototype-fix-restores-convex-contacts-under-deterministic-mode]] validated
the direction at a measured cost. This page is the production fix, built in a clone of upstream
`mujoco_warp` at HEAD `3879591` — where the defect reproduces on warp **1.15.0**, the minimum
version upstream's own dependency floor allows, so it is not an exotic configuration.[^1]

The mechanism, previously inferred, is confirmed from Warp's source: deterministic mode runs
slot-allocating kernels twice and suppresses **every** array store during the counting pass —
the guard is kernel-wide with no per-array opt-out.[^2] EPA keeps its polytope in global
scratch far too large for registers, so the counting pass reads back values it was forbidden
to write and reserves a contact count the replay pass disagrees with.

The fix stops fighting the counting pass and gives it nothing to miscount. `ccd_kernel_builder`
gains a `deterministic: bool` build argument (upstream's own `warn_overflow` is the precedent),
so the default path compiles to exactly the shipped kernel. When set, the CCD kernel stages each
candidate pair's solved result into buffers indexed by the atomic-free `collisionid`, and a new
second kernel reads those buffers and reserves the slots — every branch on the reservation path
reads memory written by a *previous launch*, which both passes read identically.[^3]

| pair | routes | default z / contacts | `RUN_TO_RUN` fixed z / contacts |
|---|---|---|---|
| sphere on plane | `PRIMITIVE` | 0.099572 / 25 | 0.099572 / 25 |
| sphere on box | `PRIMITIVE` | 0.299572 / 25 | 0.299572 / 25 |
| box on plane | `PRIMITIVE` | 0.099744 / 100 | 0.099744 / 100 |
| mesh on plane | `PRIMITIVE` | 0.099744 / 100 | 0.099744 / 100 |
| box on box | `CONVEX` | 0.299744 / 100 | 0.299744 / 100 |
| mesh on box | `CONVEX` | 0.299744 / 100 | 0.299744 / 100 |
| mesh on mesh | `CONVEX` | 0.299744 / 100 | 0.299744 / 100 |

Contact counts are **exact** — 100, not the prototype's 3100-4000 over-reservation — because the
second kernel reserves from real results rather than a worst-case bound. The primitive path is
untouched, unlike the prototype's `write_contact` change. The ~2-2.7× per-step slowdown under
`RUN_TO_RUN` appears on primitive pairs that never enter the new kernel, so it is Warp's
two-pass replay cost, not the split's.[^4]

**What the fix buys is the actual determinism verdict.** A six-box mutually-contacting pile —
the scene class [[warp-gpu-determinism-fails-on-coupled-bodies]] showed never reproduces by
default, and which before the fix could not run under `RUN_TO_RUN` at all — gives one digest
from six rollouts across two processes (`42c2ef639e147693`), and one digest at `nworld = 8`
(`c55ab24152593f09`), where default mode gives a different digest every run.[^5]

Upstream regression state: their suite passes 1334 with the fix, the sole failure being
`test_collision20`, which fails identically on the untouched `main`. A new
`test_deterministic_convex_narrowphase` passes with the fix and fails `0 != 4` without it —
the negative control [[warp-binds-determinism-mode-at-first-compile]] makes mandatory. The
in-flight determinism PRs upstream (#1422/#1425/#1533) touch none of `collision_convex.py`,
and issue #562 by a core MuJoCo developer is the open umbrella this addresses.[^6]

[^1]: `[observed]` — `uv run python repro_contact_drop.py RUN_TO_RUN box_box 64` at HEAD `3879591`, warp 1.15.0: `final_z=-4.579524 total_contacts=0 zero_contact_steps=200/200`, exit 0; default mode `final_z=0.299892 total_contacts=720`. Script: `C:\Users\Test\PROJECTS\upstream\mujoco_warp-repro\repro_contact_drop.py`.
[^2]: warp-lang 1.16.0 `warp/_src/codegen.py` L5288-5293 — "Deterministic two-pass mode must suppress normal array writes in phase 0 so the counting pass does not introduce side effects"; `warp/_src/deterministic.py` L696-703 — `needs_store_guard` returns true kernel-wide when the kernel has any counter.
[^3]: `mujoco_warp/_src/collision_convex.py` (branch `deterministic-ccd`), `ccd_kernel_builder` — staging via `ccd_ncollision/ccd_dist/ccd_witness1/ccd_witness2/ccd_dists` keyed by `collisionid`; `ccd_write_kernel` reserves via the unmodified shared `write_contact`.
[^4]: `[observed]` — `sweep_pairs.py` both modes, 40 steps after 5-step warmup: primitive pairs 6.5-6.7 → 13.8-17.5 ms/step, convex pairs 9.0-9.4 → 17.9-21.4 ms/step.
[^5]: `[observed]` — `repro_determinism.py`: default 4 reps → 4 digests; `RUN_TO_RUN` 3+3 reps in two processes → 1 digest. `repro_determinism_batch.py` nworld=8: default 3 reps → 3 digests; `RUN_TO_RUN` 3 reps → 1 digest.
[^6]: `[observed]` — `uv run pytest -n 8`: "1 failed, 1334 passed, 29 skipped"; same single failure on clean `main` before any edit. PR file lists via `gh pr view 1422|1425|1533 --json files`: zero hits for `collision_convex.py`.
