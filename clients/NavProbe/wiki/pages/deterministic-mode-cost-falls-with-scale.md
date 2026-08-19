---
title: The deterministic-mode cost peaks at small parallelism and falls with scale, reaching 3x at 4096 worlds
tags: [warp, determinism, measurement, cost, scaling]
related: ["[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[open-questions-and-what-would-answer-them]]"]
source_paths:
  - "scripts/world_scaling_sweep.py"
source_git_blobs:
  "scripts/world_scaling_sweep.py": "0d780d36b571e1d731998ce2f1581b0683b954b8"
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

## What this does not establish

One scene, one GPU, one capacity setting. The ladder stops at 4,096 worlds with the ratio still falling, so where it bottoms out — and whether it crosses 1x at Warp's documented high-contention extreme — is unmeasured. Wall clocks include per-run model construction and cache loads, identically in both arms. Digests are not comparable across world counts (each world is seed-perturbed, so `nworld` changes the digested state); the per-count verdicts, not the digests, are the result. The `constraint_capacity` here is 256 per world rather than the canonical 8,192, right-sized so the 4,096-world allocation fits; capacity does not enter the physics, but the canonical-capacity digests at nworld 2 were not re-derived under 256.
