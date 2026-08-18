---
title: A three-line alias patch clears both Warp deterministic modes; _sensor_tactile was the only blocker
tags: [warp, determinism, finding, upstream, patch]
related: ["[[mjwarp-cannot-compile-under-warp-deterministic-mode]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[open-questions-and-what-would-answer-them]]"]
source_paths:
  - "scripts/apply_tactile_alias_patch.py"
  - "scripts/det_compile_test.py"
source_git_blobs:
  "scripts/apply_tactile_alias_patch.py": "b244ed872f85a1e71b69c2e686e33f81810915f7"
  "scripts/det_compile_test.py": "36ab97032303159839966c2c7afb356e78845488"
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
hubs: [determinism-measurement]
---

# A three-line alias patch clears both Warp deterministic modes; `_sensor_tactile` was the only blocker

[[mjwarp-cannot-compile-under-warp-deterministic-mode]] left open whether `_sensor_tactile` was the only blocker, since compilation stops at the first rejected function. Measured 2026-08-18: **it is the only one.** With the mixing removed, every module in the touching-row pipeline compiles cold and the simulation steps under both `RUN_TO_RUN` and `GPU_TO_GPU`.

## The patch under test is the withdrawn upstream fix, verbatim

GitHub user NY-WaKeUp filed the same defect upstream (issue #1590) and a fix (PR #1591) on the morning of 2026-08-18, then closed both within sixteen minutes when Google's CLA check failed; the PR was never merged and, as far as the record shows, never compile-tested in CI. The fix is three edits: a second kernel parameter `sensordata_max_out` that **aliases** `sensordata_out`, the `atomic_max` routed through the alias, and `d.sensordata` bound twice at the launch site. No second array is allocated, no arithmetic changes, and the channels write disjoint index ranges (channel `k` at `adr + k*dim + vertid`, `vertid < dim`), so the alias is semantically inert.

This experiment applied that shape to the installed mujoco-warp 3.11.0 via `scripts/apply_tactile_alias_patch.py` (round-trip byte-identical; the venv is left canonical between runs).

## What was measured

Baseline, unpatched, `RUN_TO_RUN`, cold cache: the pipeline's modules compile one by one — including the full constraint solver (`_JTDAJ_sparse`, blocked Cholesky, linesearch, `_update_constraint_efc`) — until `mujoco_warp._src.sensor` fails with the known `WarpCodegenError` at sensor.py:2307, byte-identical to the two-GPU reproduction on the earlier page. Wall clock 66.3 s to the failure.

Patched, same conditions: `sensor` compiles (19.1 s for the module), `forward` compiles, and two `mjw.step` calls complete on the CPU device. `GPU_TO_GPU` behaves identically (90.1 s wall). No further function was rejected in either mode.

Two readings of that result:

- **Warp's one-family rule keys on the array binding, not the underlying memory.** The checker accepts two parameters aliasing one array, each carrying one reduction family. Nothing in the captured Warp documentation promises this, so it is an implementation observation as of warp-lang 1.16.0, and a fair review question for the upstream PR.
- **The rest of the pipeline was already deterministic-mode-clean.** The solver kernels this wiki attributes the coupled-body non-determinism to all pass deterministic codegen; the only rejection in the whole pipeline was the sensor kernel that blocked everyone from finding that out.

## What this does not establish

The compile gate is cleared; the determinism verdict is not. Whether `RUN_TO_RUN` actually makes the coupled-body scenes reproducible on the GPU — and at what throughput cost — requires running [[warp-gpu-determinism-fails-on-coupled-bodies]]'s sweep on `cuda:0` under the patched build. That run was deliberately deferred: at measurement time the RTX 3090 Ti was at 98 % utilization under a Model-Trainer job with ~10 of 20 epochs remaining, and this wiki's own rule is not to measure under load someone else created. The scene here also declares `nsensor = 0`; a model that actually uses tactile sensing exercises the aliased writes at runtime and should be checked for output equivalence against the unpatched build before the patch is trusted beyond compilation.

One scene family, one pipeline configuration (sparse solver path, nworld 2). A different model could reach kernels this scene never compiled.
