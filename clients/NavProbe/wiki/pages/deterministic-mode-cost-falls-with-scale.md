---
title: The deterministic-mode cost falls as parallelism rises -- direction measured, magnitudes not yet trustworthy
tags: [warp, determinism, measurement, cost, scaling]
related: ["[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[open-questions-and-what-would-answer-them]]"]
source_paths:
  - "scripts/world_scaling_sweep.py"
source_git_blobs:
  "scripts/world_scaling_sweep.py": "bd338f7dfa0b398b0fda9b14962cbfdd2b5ab0ac"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-20
confidence: low
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0
  device: >-
    NVIDIA GeForce RTX 3090 Ti. Believed idle at the time, but verified only at the GPU itself --
    a 2026-08-20 re-run found host CPU load alone moves these wall clocks by up to 4x.
  scene: touching row of 8 spheres, spacing 0.055, radius 0.03, timestep 0.005
  trial: TrialSpec(seed=7, step_count=150, repetitions=12), perturbation 0.01
  world_counts: [2, 64, 512, 4096]
  constraint_capacity: >-
    256 per world. Chosen so 4096 worlds would FIT, and since found too small to SOLVE --
    every rung overflowed the contact buffer, as recorded in the re-measurement below.
  deterministic_max_records: 64 (RUN_TO_RUN arm)
  patch: scripts/apply_tactile_alias_patch.py applied for both arms, reverted after
  script_revision: >-
    this run predates three changes to the cited script, none touching a measurement parameter --
    two made 2026-08-19, a --device flag (the run used the then-hardcoded first card) and a report
    format moved from hand-built JSON onto navprobe.codecs.scaling_run; and one later, adding
    linesearch_block_dim=None to the record the sweep emits. That third is a no-op for the
    figures twice over: it is passed to the RESULT constructor beside device, capacity and scene
    rather than to the run, and None selects the vendor default block dim, which is what was in
    effect when this ran. The archived log in the local runs/ directory is therefore JSON where a
    rerun would emit the wire format; the figures below are unaffected, and the pinned blob is the
    current file, so the citation is the design rather than byte-for-byte the revision that ran.
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

## Attempted re-measurement, 2026-08-20: these wall clocks should not be cited

Recorded after trying to replace them and failing. Three things were learned,
and the first is worse for this page than the reason the re-run was ordered.

**Every rung on this page overflowed the contact buffer.** The archived log of
the run that produced these figures carries **5,232** lines of
`broadphase overflow - please increase nconmax to 10 or naconmax to 577`. The
re-run reproduced the count exactly. `constraint_capacity = 256` is not enough
for a touching row of eight at these world counts -- the vendor needs roughly
581 -- and MuJoCo-Warp reports the overflow as a *warning*, so the sweep
completed and reported throughput for a **silently truncated solve**. This
package's own adapter names that hazard in its docstring ("a probe that left it
to the default would measure a silently truncated solve") and the ladder walked
into it anyway, because 256 was chosen to make 4096 worlds *fit* rather than to
make the solve *complete*. Every figure below is therefore the throughput of
less work than the scene specifies. That is independent of load, of power
management, and of anything measured since.

**The timings are not reproducible on this machine as it is normally used.**
The same ladder, same scene, same capacity, same mode, re-run on 2026-08-20:

| nworld | published | re-run | |
|---:|---:|---:|---|
| 2 | 62.5 s | 208.8 s | 3.3x slower |
| 64 | 150.6 s | 596.0 s | 4.0x slower |
| 512 | 167.0 s | 205.4 s | 1.2x slower |
| 4096 | 525.8 s | 887.1 s | 1.7x slower |

The re-run's curve is not even monotonic -- 64 worlds took three times longer
than 512 -- which no property of the solver can explain. The cause was a
SIRIUS job holding ~67 CPU-hours and 16 GiB at 68% CPU load for the whole
ladder. A rung is timed **once**, so a single-shot wall clock on a shared
workstation is not a measurement; it is a sample of whatever else was running.
Fixing that needs repetitions with the minimum reported, not just a quiet box.

**The power-throttling question is unresolved, not answered.** This page briefly
claimed EcoQoS throttling as the reason to distrust these numbers, then briefly
claimed the opposite -- that a direct A/B had ruled it out. Both claims were
made too fast. The A/B (two 90-second arms, default and opted-out) showed no
one-way step and only a ~4% aggregate difference, but it was run on a box at
**100% CPU load** with another session's SIRIUS campaign holding 10 to 17 of 24
cores, so neither arm measured a clean regime and the comparison establishes
nothing either way.[^caveat] Two things are worth carrying forward regardless:
the effect's own documentation warns that **timing is not a sufficient tell on
a variable or contended workload** -- core utilisation is the decisive one, a
throttled process sitting near a third of the machine while the box reads idle
-- and the opt-out is applied by every timed script here because it is free and
correct whether or not it turns out to matter.

**What did reproduce, exactly: the determinism verdicts.** All four rungs
reported `deterministic: true` under `RUN_TO_RUN` in both runs, and the
overflow counts matched to the line, while the wall clocks moved by up to 4x.
That is the load-immunity claim made elsewhere on this wiki, tested rather than
asserted: a determinism verdict is a digest comparison and does not care what
else the machine is doing; a wall clock is not and does.

**So:** read the *direction* below -- deterministic-mode cost falls as
parallelism rises -- as supported by the mechanism and by the determinism
results being scale-invariant. Do not cite the ratios or the throughputs. A
sound replacement needs a capacity that does not overflow, repetitions per
rung, and an idle machine; none of the three was true here.

[^caveat]: Agent board, `opus-growth-strategy-0819`, 2026-08-20T03:49:15Z (up to 13x on a LightGBM workload, with leak, thread growth, thermal recovery and contention each ruled out separately), refined by `opus-research-status-0819`, 2026-08-20T04:37:29Z: the opt-out can be applied to an **already-running** process from outside via `OpenProcess` + `SetProcessInformation`, and **core utilisation beats timing as the tell** on any workload whose per-unit cost varies. That second note also records applying the opt-out to the SIRIUS JVM at 04:37Z, which took it from 7.4 to 17.4 of 24 cores -- i.e. the machine this page's re-measurement ran on got substantially busier immediately beforehand, which is why the A/B above is inconclusive rather than negative. Neither finding is disputed here. Both raw arms are in the session scratchpad, untracked.

## What this does not establish

One scene, one GPU, one capacity setting. The ladder stops at 4,096 worlds with the ratio still falling, so where it bottoms out — and whether it crosses 1x at Warp's documented high-contention extreme — is unmeasured. Wall clocks include per-run model construction and cache loads, identically in both arms. Digests are not comparable across world counts (each world is seed-perturbed, so `nworld` changes the digested state); the per-count verdicts, not the digests, are the result. The `constraint_capacity` here is 256 per world rather than the canonical 8,192, right-sized so the 4,096-world allocation fits; capacity does not enter the physics, but the canonical-capacity digests at nworld 2 were not re-derived under 256.
