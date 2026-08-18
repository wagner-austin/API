---
title: MuJoCo-Warp cannot compile under either of Warp's deterministic modes
tags: [warp, determinism, finding, upstream, blocker]
related: [[warp-gpu-determinism-fails-on-coupled-bodies]], [[gpu-nondeterminism-amplifies-to-macroscopic-scale]], [[warp-renderer-depth-is-not-device-portable]]
sources: [mujoco-warp 3.11.0, warp-lang 1.16.0]
fact_checked: 2026-08-15
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0
  devices:
    - NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, driver 13.1, host austinpc)
    - NVIDIA GeForce RTX 3070 Ti Laptop (sm_86, 46 SMs, driver 551.23, host sedona)
  modes_attempted: [NOT_GUARANTEED, RUN_TO_RUN, GPU_TO_GPU]
  model: single row of spheres from navprobe.scenes, nsensor = 0
  replication: independent venv, kernel cache deleted before each run
---

# MuJoCo-Warp cannot compile under either of Warp's deterministic modes

Warp 1.16 ships an explicit determinism control, `warp.config.deterministic`, with three levels:

| level | value | meaning |
|---|---:|---|
| `NOT_GUARANTEED` | 0 | **the default** |
| `RUN_TO_RUN` | 1 | reproducible across runs on one device |
| `GPU_TO_GPU` | 2 | reproducible across devices |

It is a codegen option: Warp intercepts atomic calls and lowers them to an order-independent form, which is precisely the mechanism the non-determinism on this wiki was attributed to.

**Neither non-default level compiles MuJoCo-Warp 3.11.0.** Both fail identically:[^1]

```
warp._src.codegen.WarpCodegenError: Error while parsing function "_sensor_tactile"
  at mujoco_warp/_src/sensor.py:2307:
    wp.atomic_add(sensordata_out, worldid,
                  sensor_adr[sensor_id] + 1 * dim + vertid, forceT[1])
  Deterministic mode does not support mixing 'max' and 'add' reductions
  on array 'sensordata_out' in the same function or kernel.
```

## It is a property of the codegen, not of one machine

The failure reproduces byte-identically on a second GPU: same function, same line, same
message, on an RTX 3070 Ti Laptop with **46** streaming multiprocessors against the 3090
Ti's **84**, a different driver branch, and a separately built virtual environment.[^3]

That rules out the explanations a single-machine result leaves open — a driver bug, a
corrupted kernel cache, one bad install. Both runs deleted the Warp kernel cache before
`wp.init()` and confirmed every module logged `(compiled)` rather than `(cached)`, because
a warm cache skips codegen entirely and turns a failing mode into a false pass.[^3]

Since the rejection happens while *parsing* Python into Warp IR, before any device code is
generated, no GPU is required to trigger it and no GPU can avoid it. **Re-measuring this on
further hardware has no value.** What remains unknown is stated below, and it is not about
which card you run on.

## It cannot be avoided by choosing a simpler model

The scene used here declares **no sensors at all** — `nsensor` is zero — and it still fails.[^2]

Warp compiles a whole Python module into a kernel module, so `mujoco_warp._src.sensor` is built in full whether or not a given model uses tactile sensing. `_sensor_tactile` is in that module, so every MJWarp model on every scene hits the same wall.

## What this means for everything else on this wiki

Every measurement in this project ran under `NOT_GUARANTEED`, because that is the default and because it is the only level available. That was not a choice — it is the whole configuration space.

So the findings should be read as characterising the cost of the *only mode MJWarp users can currently run*:

- run-to-run reproducibility fails once a handful of bodies touch each other ([[warp-gpu-determinism-fails-on-coupled-bodies]]);
- the resulting difference amplifies chaotically to macroscopic scale ([[gpu-nondeterminism-amplifies-to-macroscopic-scale]]);
- depth output is not portable across devices ([[warp-renderer-depth-is-not-device-portable]]).

The MJWarp documentation's advice — set the device to CPU for deterministic results — is therefore not conservative guidance but the only remedy available, and it costs the GPU.

## Why this is the most actionable thing measured here

The other findings are characterisations: they describe behaviour and locate thresholds, and acting on them means changing what you measure or where you run it. This one is a **defect with an address**: one function, one line, one stated reason, and a fix that lies entirely inside MuJoCo-Warp rather than requiring anything of Warp or of the user.

If `_sensor_tactile` stopped mixing `max` and `add` reductions on one array — separate accumulators, or a second pass — then `RUN_TO_RUN` and `GPU_TO_GPU` become available, and most of what this wiki reports becomes a setting rather than a property.

## What this does not establish

Whether `_sensor_tactile` is the *only* blocker. Compilation stops at the first rejected function, so a fix there may reveal others; nothing here has compiled MJWarp all the way through under a deterministic mode.

Nor is the cost known. Deterministic atomics are generally slower, and no timing was taken because nothing compiled — so "how much throughput would the fix cost" is unmeasured and is a fair question to ask before assuming the fix is free.

[^1]: `[observed]` — `warp.config.deterministic` set to `DeterministicMode.RUN_TO_RUN` and then `GPU_TO_GPU` before `wp.init()`, driving `navprobe.adapters.mjx_warp_state` on `cuda:0`. Both raised the quoted `WarpCodegenError` while loading `mujoco_warp._src.sensor`. Under `NOT_GUARANTEED` the same script runs, reporting the separated row reproducible and the touching rows not.
[^2]: `[observed]` — `mujoco.MjModel.from_xml_string(build_scene(row_scene(8, 0.055, 0.03, 0.005))).nsensor` is `0`, and `_sensor_tactile` appears in `mujoco_warp/_src/sensor.py` at line 2181 with a reference at 2559.
[^3]: `[observed]` — on host `sedona`, `C:\navprobe\.venv\Scripts\python.exe C:\navprobe\determinism_check.py RUN_TO_RUN` and the same with `GPU_TO_GPU`. The script deletes `%LOCALAPPDATA%\NVIDIA\warp\Cache` before setting `wp.config.deterministic` and calling `wp.init()`. Both raised the identical `WarpCodegenError` quoted above. A prior `NOT_GUARANTEED` run on the same host compiled all 31 modules — every one logging `(compiled)` — and stepped successfully, reporting `cuda:0` as `sm_86, sm_count=46`.
