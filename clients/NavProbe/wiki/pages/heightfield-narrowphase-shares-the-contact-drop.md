---
title: The heightfield narrowphase shares the deterministic-mode contact drop, and the same split fixes it
tags: [warp, determinism, contacts, collision, heightfield, fix, upstream]
related: ["[[deterministic-mode-drops-contacts-in-convex-narrowphase]]", "[[kernel-split-fix-restores-convex-contacts-at-upstream-head]]"]
provenance:
  - "google-deepmind/mujoco_warp @ 3879591, branch deterministic-ccd (local, unpushed)"
  - "warp-lang 1.15.0"
fact_checked: 2026-08-31
confidence: high
measured_with:
  package: mujoco_warp @ git 3879591 + working-tree fix
  warp: 1.15.0
  backend: warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, host austinpc)
  modes: [NOT_GUARANTEED, RUN_TO_RUN]
  deterministic_max_records: 64
  step_count: 60
  scene: box dropped on a flat 4x4 heightfield
hubs: [determinism-measurement]
---

# The heightfield narrowphase shares the deterministic-mode contact drop, and the same split fixes it

MuJoCo-Warp has two CCD kernels: the general convex one and a separate heightfield one
(`ccd_hfield_kernel_builder`), which iterates the terrain's prisms, runs GJK/EPA per prism
from the same `epa_*[ccdid]` global scratch, and selects up to four contacts to write. The
scratch dependence that breaks the convex kernel under Warp's deterministic modes
([[deterministic-mode-drops-contacts-in-convex-narrowphase]]) is therefore present here by
construction — and it was measured before it was fixed, because this wiki has been wrong by
inference before.

| box on flat hfield | default | `RUN_TO_RUN` before fix | `RUN_TO_RUN` after fix |
|---|---|---|---|
| final z | 0.099233 | **−0.198807, sinking** | 0.099233 |
| contacts (60 steps) | 50 | **0** | 50 |
| zero-contact steps | 35/60 | 60/60 | 35/60 |

The failure is the same silent one: the box passes through the terrain and the process exits
0.[^1] The fix is the same staged-write split as
[[kernel-split-fix-restores-convex-contacts-at-upstream-head]], adapted to this kernel's
select-N structure: the prism loop and contact selection stay where they are (register-side
math), but under the `deterministic` build flag the up-to-four selected contacts are staged
per candidate pair instead of reserved in place, and a second kernel — whose every branch
reads prior-launch memory — recomputes the material parameters and does the reserving.[^2]
Post-fix, `RUN_TO_RUN` matches the default mode to the digit, including the per-step contact
profile.[^3]

One structural note with scope beyond heightfields: the flex-contact pipeline
(`collision_flex.py`) turns out to already be shaped this way — candidates are staged into
global arrays by earlier kernels and `_write_filtered_contacts` reserves from them — which is
why its reservation gate is sound under deterministic mode by the same argument that makes
the fix correct.[^4] Confirmed by measurement, not just inspection: an 8×8 flex grid dropped
on a plane produces 5888 contacts over 200 steps in both modes, one digest per mode from
three repetitions — the same digest, in fact, in both. The shipped flex fixtures cannot show
this because they disable collision (`contype="0" conaffinity="0"`) or pin their flexes; a
rope-fixture check ran first and scored reproducible **vacuously** at zero contacts, exactly
the false positive [[a-determinism-verdict-needs-a-correctness-oracle]] predicts.[^5]

[^1]: `[observed]` — `repro_hfield_drop.py RUN_TO_RUN 64 60` at HEAD `3879591` before the hfield fix: `final_z=-0.198807 total_contacts=0 zero_contact_steps=60/60`, exit 0; default mode `final_z=0.099233 total_contacts=50`. Script: `C:\Users\Test\PROJECTS\upstream\mujoco_warp-repro\repro_hfield_drop.py`.
[^2]: `mujoco_warp/_src/collision_convex.py` (branch `deterministic-ccd`), `ccd_hfield_kernel_builder` — staging arrays `hfield_ncon/hfield_cdist/hfield_cpos/hfield_cnormal` of shape `(naconmax, 4)`; `ccd_hfield_write_kernel` recomputes `contact_params` and replays `write_contact` with the original call indices.
[^3]: `[observed]` — same script, same arguments, after the fix: `final_z=0.099233 total_contacts=50 zero_contact_steps=35/60`, matching the default arm exactly.
[^4]: `mujoco_warp/_src/collision_flex.py` L2192-2340 — `_write_filtered_contacts` takes `cand_dist`/`cand_pos`/`cand_nrm` as array inputs written by the candidate-generation and filter kernels; its `cand_dist[i] >= margin` gate before the `nacon` atomic reads prior-launch memory.
[^5]: `[observed]` — `repro_flex_determinism.py <MODE> 64 3 dropgrid 200`: default `contacts=[5888,5888,5888]`, 1 digest `a70bb794c8c0453c`; `RUN_TO_RUN` identical including the digest. Prior vacuous run: `rope` fixture, `contacts=[0,0,0]`, `LIVE: False` both modes. Script: `C:\Users\Test\PROJECTS\upstream\mujoco_warp-repro\repro_flex_determinism.py`.
