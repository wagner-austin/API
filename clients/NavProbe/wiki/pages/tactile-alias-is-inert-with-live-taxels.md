---
title: The tactile alias patch is semantically inert once the taxels actually execute
tags: [warp, determinism, tactile, finding, upstream, patch]
related: ["[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[mjwarp-cannot-compile-under-warp-deterministic-mode]]", "[[deterministic-mode-drops-contacts-in-convex-narrowphase]]"]
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
  - "google-deepmind/mujoco_warp PR #1591 (withdrawn)"
fact_checked: 2026-08-29
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cpu and warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, host austinpc)
  model: mujoco_warp/_src/sensor_test.py::test_tactile_sensor_geom_deduplication
  nsensortaxel: 12
  mode: NOT_GUARANTEED
  step_count: 30
  repetitions: 3 (cpu), 8 (cuda:0)
  kernel_cache: fresh directory per arm, every module logged compiled
hubs: [determinism-measurement]
---

# The tactile alias patch is semantically inert once the taxels actually execute

[[tactile-alias-patch-clears-warp-deterministic-compile]] closed with a stated gap: its scene family declares `nsensor = 0`, so the patched `_sensor_tactile` compiled but its aliased writes **never ran**. That page recommended checking output equivalence on a model that actually uses tactile sensing before trusting the patch beyond compilation. Measured 2026-08-29: **the patch changes nothing.**

## The fixture is the vendor's own

MuJoCo-Warp ships a self-contained tactile model in its test suite — a builtin sphere mesh resting on a box, `multiccd` enabled, with a keyframe that seats it in contact.[^1] It needs no plugin and no external asset, and it drives live taxels: `nsensortaxel = 12`, two contacts, non-zero `sensordata`, agreeing with MuJoCo's own CPU reference to 2.81 × 10⁻⁸.[^2]

Using the vendor's fixture rather than a hand-authored one removes "is your scene even valid" as a review objection, and it is the same model whose assertion upstream already treats as ground truth.

## Patched and unpatched produce identical bytes, on both devices

Thirty `mjw.step` calls, each repetition rebuilding `Data` from the keyframe, digested through this package's own `digest_step` / `digest_run` so the numbers are comparable with every other result on this wiki:[^3]

| device | unpatched run digest | patched run digest | agree |
|---|---|---|---|
| warp cpu | `50f926422cfd75d1…` | `50f926422cfd75d1…` | **yes**, final `sensordata` max abs diff `0.0` |
| warp cuda:0 | `04446087e20d06dc…` | `04446087e20d06dc…` | **yes**, 8/8 repetitions each |

## It is not a stale-cache false pass

The two arms genuinely compiled different code. Warp logged `mujoco_warp._src.sensor` as module hash `b7b2442` unpatched and `c92620e` patched, each `(compiled)` rather than `(cached)`, into separate `kernel_cache_dir` directories.[^3] A warm cache skips codegen entirely and would turn an unchanged binary into a false agreement — the failure mode [[passing-test-can-miss-its-own-premise]] is about.

## The sensor path is also correct under deterministic mode

Holding the world fixed — every repetition rebuilding `Data` from the same keyframe and calling `sensor_acc` alone, with no `step`, no solver and no integrator, so the contact set is identical by construction — the patched kernel under `RUN_TO_RUN` on `cuda:0` returns one distinct digest from eight repetitions, at `0.199289039`, matching the MuJoCo CPU reference to 2.81 × 10⁻⁸.[^4]

That matters because the stepped runs under the same mode return zeros and disagree with each other. The sensor is not the cause: see [[deterministic-mode-drops-contacts-in-convex-narrowphase]], where the world itself is what varies.

## What this establishes for the upstream fix

PR #1591's alias shape is safe to file. The concern a reviewer would raise — that binding one array to two parameters changes what the kernel computes — is answered by measurement rather than by the argument that the channels write disjoint indices.

## What this does not establish

Equivalence was measured on one fixture with twelve taxels and two contacts. A model with many taxels, or with tactile geoms in mutual contact, is unmeasured. Nor does it say the mode the patch unlocks is *usable* — that is a separate question with a separate and worse answer.

[^1]: mujoco-warp 3.11.0, `mujoco_warp/_src/sensor_test.py::test_tactile_sensor_geom_deduplication` — builtin `sphere` mesh geom on a `.7 .7 .3` box, `<flag multiccd="enable"/>`, `<tactile geom="sensor_geom" mesh="sensor_mesh"/>`, keyframe `qpos="0 0 1 1 0 0 0"`.
[^2]: `[observed]` — the [^1] fixture (`mujoco_warp/_src/sensor_test.py::test_tactile_sensor_geom_deduplication`, mujoco-warp 3.11.0) built with `mujoco.MjModel.from_xml_string`, reset to keyframe 0, `mj_forward`, then `mjw.put_model` / `mjw.put_data`: `m.nsensortaxel` is 12, `mjd.ncon` is 2, and `abs(d.sensordata.numpy()[0] - mjd.sensordata).max()` is 2.81e-08 after `mjw.sensor_acc`.
[^3]: `[observed]` — 30 `mjw.step` calls per repetition, fresh `Data` from the keyframe each time, `wp.config.kernel_cache_dir` set to a fresh directory per arm, `wp.config.deterministic = NOT_GUARANTEED`. The patch applied and reverted with `python -m scripts.apply_tactile_alias_patch {apply,revert}`; the venv verified canonical (`grep -c sensordata_max_out` = 0) between arms.
[^4]: `[observed]` — same fixture, 8 repetitions, each rebuilding `Data` from the identical keyframe and calling `mjw.sensor_acc(m, d)` with no `step`; `wp.config.deterministic = RUN_TO_RUN`, `deterministic_max_records = 64`, `cuda:0`. State digest identical across all 8 (`nacon` = 2 every repetition), sensor digest identical across all 8.
