---
title: The deterministic-mode cost peaks at small parallelism and falls with scale, reaching 3x at 4096 worlds
tags: [warp, determinism, measurement, cost, scaling]
related: ["[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[open-questions-and-what-would-answer-them]]"]
source_paths:
  - "scripts/world_scaling_sweep.py"
source_git_blobs:
  "scripts/world_scaling_sweep.py": "842b6cd60eed3c7a569dbe88f064f3b087946dfe"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-19
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti (idle baseline; trainer run had ended)
  scene: touching row of 8 spheres, spacing 0.055, radius 0.03, timestep 0.005
  trial: TrialSpec(seed=7, step_count=150, repetitions=12), perturbation 0.01
  world_counts: [2, 64, 512, 4096]
  constraint_capacity: 256 per world (right-sized; the canonical 8192 allocates njmax*nv*nworld sparse-Jacobian entries and does not fit at 4096 worlds)
  deterministic_max_records: 64 (RUN_TO_RUN arm)
  patch: scripts/apply_tactile_alias_patch.py applied for both arms, reverted after
  script_revision: >-
    this run predates two changes to the cited script, both made 2026-08-19 and neither touching
    a measurement parameter -- a --device flag (the run used the then-hardcoded first card), and
    a report format moved from hand-built JSON onto navprobe.codecs.scaling_run. The archived log
    in the local runs/ directory is therefore JSON where a rerun would emit the wire format; the
    figures below are unaffected, and the pinned blob is the current file, so the citation is the
    design rather than byte-for-byte the revision that ran.
hubs: [determinism-measurement]
---

# The deterministic-mode cost peaks at small parallelism and falls with scale, reaching 3x at 4096 worlds

The 5.07x cost measured at the instrument's canonical `world_count = 2` ([[tactile-alias-patch-clears-warp-deterministic-compile]]) carried a stated qualifier: two worlds under-occupy the card, and Warp's sort-and-reduce lowering should amortise as parallel atomic traffic grows. Measured on the idle GPU with one coupled scene (touching row of 8) laddered over world counts, that is what happens — after first getting worse:

| nworld | default (world-steps/s) | RUN_TO_RUN (world-steps/s) | cost ratio | default reproducible | RUN_TO_RUN reproducible |
|---:|---:|---:|---:|---|---|
| 2 | 260 | 58 | 4.5x | no | **yes** |
| 64 | 5,655 | 765 | **7.4x** | no | **yes** |
| 512 | 27,883 | 5,519 | 5.1x | no | **yes** |
| 4,096 | 41,919 | 14,023 | **3.0x** | no | **yes** |

Three facts carry the page:

- **Determinism holds at every scale.** All twelve repetitions bit-identical at every world count in the `RUN_TO_RUN` arm, while the default arm was irreproducible at every world count — the same split as the canonical sweep, now shown scale-invariant in both directions. `deterministic_max_records = 64` sufficed even at 4,096 worlds, consistent with it bounding records *per thread*: more worlds add threads, not records per thread.
- **The cost curve is non-monotonic.** 4.5x at 2 worlds, peaking at 7.4x at 64, then falling through 5.1x to 3.0x at 4,096 — still falling at the top of the measured range. The default path's throughput grows 161x across the ladder while the deterministic path grows 242x, so the gap closes as the sort work amortises over more parallel traffic, exactly the direction Warp's own high-contention benchmark points.
- **The absolute deterministic throughput is usable.** 14,023 world-steps/s at 4,096 worlds is training-scale simulation, bit-exact.

## Every wall clock here predates a known timing defect on this machine

Recorded 2026-08-20, after the fact and against this page's own figures.

Windows applies EcoQoS power throttling to a long-running console process a few seconds in, and a sibling project on this machine measured a **13x** slowdown from it — identical work, repeated in one process, going 0.547 s to 7.108 s and staying there until an explicit `SetProcessInformation` opt-out. It was not thermal: ninety seconds of idle did not restore speed, only the API call did. It cannot be detected by querying, because the throttled state reads back identically to "no preference"; it is only visible in timings. It applies to *any* measurement run from an agent-launched shell on these boxes.[^caveat]

Every ladder on this page was run that way, and none had the opt-out. So:

- **The determinism verdicts are unaffected.** They are bit-exact digest comparisons, immune to how fast the machine was going. `RUN_TO_RUN` reproduced at every rung and the default failed at every rung; that stands.
- **`deterministic_max_records = 64` holding at 4096 worlds is unaffected.** It is a capacity fact, not a timing one.
- **Every wall clock and every throughput figure below is suspect**, including the headline 14,023 world-steps/s.
- **The cost *ratios* are the interesting case and are not simply safe.** Both arms of each rung were measured in the same session, which would cancel a symmetric effect — but throttling is a one-way step change part-way through a run, so arms measured before the step keep the fast regime and the rest never see it again. The sibling project saw two attempts disagree by 9% on one arm for exactly this reason. Whether the non-monotonic shape reported here — 4.5x at 2 worlds, peaking 7.4x at 64, falling to 3.0x at 4096 — is the amortisation it is read as, or partly an artefact of where the step landed in each ladder, is **not settled by the data on this page**.

**What would settle it:** re-run both ladders with the opt-out in place. That is cheap, needs no new hardware, and should happen before this page's ratios are cited anywhere load-bearing. Until then read the *direction* (cost falls with scale) as supported by the determinism results being scale-invariant and the mechanism being Warp's documented sort-and-reduce amortisation, and read the *numbers* as provisional.

[^caveat]: Agent board, `opus-growth-strategy-0819`, 2026-08-20T03:49:15Z — root-caused and fixed for `covenant_ml`'s benchmark harness, with the before/after fit times, the ruled-out alternatives (RSS flat, thread count flat, idle does not restore) and the `GetProcessInformation` detection gotcha. Not measured on NavProbe's own ladders; what is claimed here is that the same machine and the same kind of shell were used, so the same exposure applies. Confirming or clearing it for these figures needs the re-run described above.

## What this does not establish

One scene, one GPU, one capacity setting. The ladder stops at 4,096 worlds with the ratio still falling, so where it bottoms out — and whether it crosses 1x at Warp's documented high-contention extreme — is unmeasured. Wall clocks include per-run model construction and cache loads, identically in both arms. Digests are not comparable across world counts (each world is seed-perturbed, so `nworld` changes the digested state); the per-count verdicts, not the digests, are the result. The `constraint_capacity` here is 256 per world rather than the canonical 8,192, right-sized so the 4,096-world allocation fits; capacity does not enter the physics, but the canonical-capacity digests at nworld 2 were not re-derived under 256.
