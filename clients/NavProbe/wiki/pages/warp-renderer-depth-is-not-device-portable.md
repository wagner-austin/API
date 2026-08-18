---
title: The MJX-Warp batch renderer's depth output is not portable across devices
tags: [warp, rendering, determinism, measurement, finding, depth]
related: ["[[backend-divergence-begins-at-first-contact]]", "[[mjx-determinism-does-not-cross-backends]]", "[[warp-rendered-stream-is-reproducible-within-a-device]]"]
source_paths:
  - "src/navprobe/adapters/mjx_warp_render.py"
  - "src/navprobe/scenes.py"
source_git_blobs:
  "src/navprobe/adapters/mjx_warp_render.py": "898c44998cd0d22b65901c73453c940b081c50ac"
  "src/navprobe/scenes.py": "4a05c692fbd2740bd717f015e7725fa8175fc207"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-13
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cpu vs warp cuda:0, same machine, same process image
  device: NVIDIA GeForce RTX 3090 Ti
  model: falling sphere with a free joint on a plane, timestep 0.005
  camera: one camera, 64x64, rgb and depth
  world_count: 2
  initial_offsets: [0.013, -0.027]
  step_count: 200
hubs: [rendered-observations]
---

# The MJX-Warp batch renderer's depth output is not portable across devices

This is the measurement the project was built to take. The published determinism results for GPU-batched simulators were taken with rendering disabled, and the papers name perception and sensor rendering as uncovered — so the rendered observation stream a navigation policy actually consumes had never been checked.

It does not reproduce across devices, and the reason is the renderer itself rather than the physics feeding it.

## Every input to the renderer is bit-identical; the depth output is not

Rendering one frame from the same state on Warp's CPU and CUDA devices:

| quantity | identical across devices? |
|---|---|
| `qpos` (generalised positions) | **yes** |
| `geom_xpos` (Cartesian geom positions) | **yes** |
| `geom_xmat`, `xpos`, `xquat` | **yes** |
| `rgb_data` (packed colour) | **yes** |
| `depth_data` | **no** |

The raycaster's inputs agree to the bit. Its depth output does not.[^1]

## How much it differs

| measure | value |
|---|---|
| differing pixels | 2,794 of 8,192 (**34.1 %**) |
| max absolute difference | 1.466 × 10⁻⁵ |
| mean absolute difference | 3.246 × 10⁻⁷ |
| max relative difference | 7.399 × 10⁻⁶ |
| max difference in float32 ULPs | 123 |
| non-finite values | none |

Depth minimum and maximum are identical to all printed digits; the disagreement is spread through the interior of the image, not at its extremes.[^2]

Re-measured through the package's own scene family and codecs — a different scene, so different pixels — the shape holds and the figures move: colour identical at **0 of 8,192**, depth differing on **3,492 of 8,192 (42.6 %)** at a maximum of 1.43 × 10⁻⁶.[^4] The fraction and the magnitude are properties of a scene and a camera; what replicates is that colour agrees exactly and depth does not, on a large fraction of pixels, at around one part in a million.

It does not accumulate. At step 1, 2,794 pixels differ with a maximum of 1.47 × 10⁻⁵; at steps 50, 100 and 200 it is 2,742 pixels at 1.43 × 10⁻⁶. The difference is a standing property of the raycaster, not a drift.[^3]

## Colour hides it, and that is the trap

`rgb_data` is quantised to eight bits per channel. A depth discrepancy of one part in 10⁶ is far below one quantisation step, so it rounds away and the colour buffer compares equal — at every step measured, including those where depth differs.

A team validating a rendered pipeline by comparing RGB frames would conclude the renderer is device-portable. It is not. The failure is invisible in exactly the channel people look at, and present in the channel depth-based navigation policies consume.

## This is separate from the contact divergence

The depth discrepancy is already present at **step 1**, thirty-eight steps before the solver's first cross-backend disagreement at step 57 ([[backend-divergence-begins-at-first-contact]]). At step 1 the physics state is bit-identical on both devices, so the renderer is the only candidate.

Two independent portability failures, then, with different causes and different onsets:

1. the **contact solver**, from step 57, common to MJX and Warp;
2. the **depth raycaster**, from the first frame, in Warp.

## Consequences

- A depth observation captured on GPU cannot be bit-compared against one captured on CPU, even with the physics pinned.
- Reference frames used as regression fixtures are only valid on the device that produced them.
- An RGB-only equality check certifies nothing about depth.

## What this does not establish

Two scenes, one camera geometry each, one resolution, one GPU. The differing-pixel fraction is demonstrably **not** scene-independent — it moved from 34.1 % to 42.6 % between the two scenes measured — so no fraction here should be read as characteristic of the renderer, as is the behaviour at other resolutions or with multiple cameras. The mechanism inside the raycaster, plausibly the order of BVH traversal or intersection accumulation differing between the CPU and CUDA kernels, remains a hypothesis this measurement cannot decide: it shows the inputs agree and the outputs do not, and nothing narrower.

[^1]: src/navprobe/adapters/mjx_warp_render.py:56 `MjWarpRenderSimulator` — `[observed]` — one step then one render on each device from identical written initial positions; `geom_xpos`, `geom_xmat`, `xpos`, `xquat` and `rgb_data` all compared equal by `numpy.array_equal`, `depth_data` did not.
[^2]: `[observed]` — element-wise comparison of the two 2×4096 float32 depth buffers. Depth min/max printed identically as 0.0000000000 and 6.5796294212 on both devices; the float64 sums were 18832.4648828506 (cuda) and 18832.4648711681 (cpu).
[^3]: `[observed]` — 200-step rollout on each device with depth captured at steps 1, 50, 100 and 200: 2794 pixels differing at max 1.466274e-05, then 2742 pixels at max 1.430511e-06 for the remaining three frames.
[^4]: src/navprobe/scenes.py:195 `row_scene` — `[observed]` — `navprobe.scenes.row_scene(1, 0.070, 0.03, 0.005)` rendered at 64x64 over two worlds, one step, seed 7; each device recorded its final observation in its own process via `navprobe.storage.save_observation_record`, and a third process compared them with `navprobe.divergence.compare_observations`. Separate processes are required because MuJoCo-Warp's device is global process state.
