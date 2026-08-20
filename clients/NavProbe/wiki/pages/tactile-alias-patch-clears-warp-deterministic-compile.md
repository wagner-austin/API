---
title: The alias patch makes GPU determinism a setting; 10/10 scenes bit-reproducible under RUN_TO_RUN
tags: [warp, determinism, finding, upstream, patch]
related: ["[[mjwarp-cannot-compile-under-warp-deterministic-mode]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[open-questions-and-what-would-answer-them]]"]
source_paths:
  - "scripts/apply_tactile_alias_patch.py"
  - "scripts/det_compile_test.py"
source_git_blobs:
  "scripts/apply_tactile_alias_patch.py": "a5c27614e81318fe95f5160e5049651dc313ebde"
  "scripts/det_compile_test.py": "17aa86fa60ab29175b7921ec5721ae7a7b42d95d"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
  - "google-deepmind/mujoco_warp PR #1591 (withdrawn)"
fact_checked: 2026-08-18
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cpu (the rejection is parse-time; no GPU required or involved)
  host: austinpc
  modes_attempted: [RUN_TO_RUN, GPU_TO_GPU]
  model: navprobe.scenes.row_scene(6, 0.055, 0.03, 0.005), nworld 2, nsensor = 0
  kernel_cache: fresh directory per run via wp.config.kernel_cache_dir (cold codegen, every module logged compiled)
  patch: scripts/apply_tactile_alias_patch.py (the PR #1591 alias-binding shape, applied to site-packages)
  script_revision: >-
    both cited scripts were reworked 2026-08-19 to satisfy this repo's guard rules (structured
    output through navprobe.codecs, no bare prints, no blind excepts), and the compile gate now
    drives the scene through navprobe.adapters.mjx_warp_state rather than hand-written
    put_model/put_data calls. The patch's three swap sites are byte-identical to the revision
    that ran, and its apply/revert round trip was re-verified against the installed sensor.py at
    the rework. The gate no longer catches a rejection -- a mode Warp refuses now raises the
    vendor's own error instead of being summarised into a FAIL document.
  gpu_sweep: cuda:0 RTX 3090 Ti, ten-scene family, TrialSpec(seed 7, 150 steps, 12 reps), deterministic_max_records 64, co-resident training load (timings not interpretable)
hubs: [determinism-measurement]
---

# The alias patch makes GPU determinism a setting; 10/10 scenes bit-reproducible under `RUN_TO_RUN`

[[mjwarp-cannot-compile-under-warp-deterministic-mode]] left open whether `_sensor_tactile` was the only blocker, since compilation stops at the first rejected function. Measured 2026-08-18: **it is the only one** — with the mixing removed, every module in the touching-row pipeline compiles cold and steps under both `RUN_TO_RUN` and `GPU_TO_GPU` — **and the unblocked mode delivers**: the full ten-scene sweep on `cuda:0` reports every scene bit-reproducible, including the coupled-body scenes where the default mode returns twelve distinct results from twelve runs.

## The patch under test is the withdrawn upstream fix, verbatim

GitHub user NY-WaKeUp filed the same defect upstream (issue #1590) and a fix (PR #1591) on the morning of 2026-08-18, then closed both within sixteen minutes when Google's CLA check failed; the PR was never merged and, as far as the record shows, never compile-tested in CI. The fix is three edits: a second kernel parameter `sensordata_max_out` that **aliases** `sensordata_out`, the `atomic_max` routed through the alias, and `d.sensordata` bound twice at the launch site. No second array is allocated, no arithmetic changes, and the channels write disjoint index ranges (channel `k` at `adr + k*dim + vertid`, `vertid < dim`), so the alias is semantically inert.

This experiment applied that shape to the installed mujoco-warp 3.11.0 via `scripts/apply_tactile_alias_patch.py` (round-trip byte-identical; the venv is left canonical between runs).

## What was measured

Baseline, unpatched, `RUN_TO_RUN`, cold cache: the pipeline's modules compile one by one — including the full constraint solver (`_JTDAJ_sparse`, blocked Cholesky, linesearch, `_update_constraint_efc`) — until `mujoco_warp._src.sensor` fails with the known `WarpCodegenError` at sensor.py:2307, byte-identical to the two-GPU reproduction on the earlier page. Wall clock 66.3 s to the failure.

Patched, same conditions: `sensor` compiles (19.1 s for the module), `forward` compiles, and two `mjw.step` calls complete on the CPU device. `GPU_TO_GPU` behaves identically (90.1 s wall). No further function was rejected in either mode.

Two readings of that result:

- **Warp's one-family rule keys on the array binding, not the underlying memory.** The checker accepts two parameters aliasing one array, each carrying one reduction family. Nothing in the captured Warp documentation promises this, so it is an implementation observation as of warp-lang 1.16.0, and a fair review question for the upstream PR.
- **The rest of the pipeline was already deterministic-mode-clean.** The solver kernels this wiki attributes the coupled-body non-determinism to all pass deterministic codegen; the only rejection in the whole pipeline was the sensor kernel that blocked everyone from finding that out.

## The GPU verdict: 10/10 scenes reproducible under RUN_TO_RUN, including every coupled-body scene

Ran the same evening on `cuda:0` (RTX 3090 Ti), same ten-scene family and trial design as [[warp-gpu-determinism-fails-on-coupled-bodies]] — TrialSpec(seed 7, 150 steps, 12 repetitions), world_count 2, perturbation 0.01 — via `scripts/gpu_deterministic_sweep.py` (raw log in the local `runs/` directory, which is deliberately untracked; the verdict rows are reproduced here in full). **Every scene reported `deterministic: true` with `first_divergent_step: null`, including touching rows at 5, 6, 8 and 32 bodies — the configurations that produce twelve distinct results from twelve runs under the default mode.** The GPU non-determinism this wiki characterises is now demonstrably a *setting*: three patched lines plus two config values convert it to bit-reproducibility.

| bodies | spacing | deterministic | reference digest (first 16) |
|---:|---:|---|---|
| 2 | 0.070 | true | 24e4b98b3cd5fdda |
| 8 | 0.070 | true | e2cc60ade06391fa |
| 16 | 0.070 | true | c9002ed3f667f1ae |
| 32 | 0.070 | true | 7bb8689c962bb13a |
| 2 | 0.055 | true | 3849c2aa47a799cf |
| 4 | 0.055 | true | d276b2f515cfa6ef |
| 5 | 0.055 | true | fa81afe0a180746a |
| 6 | 0.055 | true | d59366edf41d4acf |
| 8 | 0.055 | true | 2ab58e00b21fb1ce |
| 32 | 0.055 | true | ab082525f26ca3dc |

## A second wall stands behind the compile wall: the record-buffer capacity

The first GPU attempt died at the 32-body separated scene with `RuntimeError: Deterministic scatter buffer overflow in kernel '_M'. Increase 'deterministic_max_records' or reduce the per-thread atomic count.` Warp's deterministic lowering buffers each thread's atomic writes as records before sorting and reducing them; the default capacity (`wp.config.deterministic_max_records = 0`) uses a code-generated static lower bound, which the solver's data-dependent contact loops exceed once enough bodies interact. Setting `deterministic_max_records = 64` before `wp.init()` cleared it at every scene size tested. The failure is loud, not silent — but it means the deterministic path carries a model-dependent capacity knob: a configuration validated on small scenes can still fail at runtime on larger ones. Nobody could have reported this before, because nothing MJWarp-shaped ever compiled far enough to reach it.

## The cost: 3.3x to 7.1x at this instrument's scale, worst where the coupling is worst

Measured on the freed GPU (trainer finished, 0 % baseline utilisation): each mode run twice in sequence with a shared kernel cache, and the warm second runs compared per scene. Default mode: 12.6 to 20.3 s per scene. `RUN_TO_RUN`: 42 to 137 s. Ratios 3.3 to 5.2x on the separated family, 3.5 to 7.1x on the touching family — **the deterministic path is most expensive in exactly the coupled scenes it exists to fix** (6.4 to 7.1x at touching 6/8/32), consistent with record buffering and sorting scaling with contact coupling. Overall warm-total ratio 5.07x.

These figures ran at `constraint_capacity = 8192` and the archived logs carry **zero**
contact-buffer overflows, so unlike the world-count ladder
([[deterministic-mode-cost-falls-with-scale]]) they measure the full solve. What they
share with it is the weaker caveat: they are single-shot wall clocks taken on a shared
workstation whose idleness was verified at the GPU but not at the host CPU, and a
2026-08-20 re-measurement showed host load alone moving comparable timings by up to 4x.
The ratio is better protected than an absolute time would be, since both arms ran in one
session minutes apart, but it has not been repeated.

Two qualifiers keep this from being a universal number. The per-scene wall includes per-scene model construction and cache loads, not pure stepping. And `world_count = 2` barely occupies an RTX 3090 Ti: Warp's own published table shows the sort-and-reduce path *gaining* on atomics as contention rises, so at RL-scale world counts the ratio could move substantially in either direction. This is the instrument's cost, honestly bounded, not MJWarp's cost in general.

## The default mode replicated its own failure in the same session

The warm default-mode run, on the same patched build, same night, same card, reported the touching family irreproducible from 5 bodies upward and the separated family reproducible throughout — the exact boundary [[warp-gpu-determinism-fails-on-coupled-bodies]] published. The patch alters nothing under the default mode; the mode switch carries the entire effect.

## Cross-process, cross-load reproducibility: three processes, one digest set

Three separate `RUN_TO_RUN` processes — the original sweep under heavy co-resident training load, a cold fresh process on the idle GPU, and a warm process on the idle GPU — produced **10/10 identical digests pairwise across all three**. The deterministic path's output is invariant to process boundaries and to GPU load, which upgrades the claim from within-process repetition to the reproducibility users actually need.

## What this does not establish

The digests are RUN_TO_RUN-mode digests, comparable neither to default-mode GPU digests nor to the CPU reference — the deterministic lowering fixes a *different* summation order rather than reproducing either of the old ones. Cross-boot and cross-machine repetition of the RUN_TO_RUN digests is unmeasured, as is `GPU_TO_GPU` on a second architecture. The scene family declares `nsensor = 0`, so the aliased tactile writes were compiled but never executed with live taxel data; a model that uses tactile sensing should be checked for output equivalence against the unpatched build before the patch is trusted beyond compilation. One scene family, one pipeline configuration (sparse solver path, nworld 2), `deterministic_max_records` validated only up to 32 bodies.
