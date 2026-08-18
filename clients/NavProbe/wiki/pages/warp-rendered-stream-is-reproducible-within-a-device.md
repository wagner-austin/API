---
title: The rendered stream reproduces exactly within a single device
tags: [warp, rendering, determinism, measurement]
related: ["[[warp-renderer-depth-is-not-device-portable]]", "[[mjx-cuda-batched-step-reproduces]]"]
source_paths:
  - "tests/adapters/test_mjx_warp_render.py"
source_git_blobs:
  "tests/adapters/test_mjx_warp_render.py": "2848fe1f9a8bcdbda294de393ead433ff06551b5"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-13
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0 and warp cpu, measured separately
  device: NVIDIA GeForce RTX 3090 Ti
  model: falling sphere with a free joint on a plane, timestep 0.005
  camera: one camera, 64x64
  seed: 7
  step_count: 60
  repetitions: 3
  world_counts: [1, 2, 4]
  channels: [rgb, depth, both]
hubs: [rendered-observations]
---

# The rendered stream reproduces exactly within a single device

> **Read the conditions.** The scene is a single falling sphere, whose only contact is with the floor. GPU determinism is scene-dependent and fails once six bodies touch each other ([[warp-gpu-determinism-fails-on-coupled-bodies]]); this scene has no body-to-body contact, so it sits below that threshold by construction. Whether the *renderer* would still reproduce under a state trajectory that is already irreproducible is a separate and unmeasured question — and worth answering, because it is the case a real manipulation scene would be in.

Pin the device and the batch renderer is bit-reproducible. Three independent rendered rollouts at one seed agreed exactly, at every batch width tried, on both Warp devices, in colour and in depth.[^1]

| device | `nworld` | deterministic | first divergence |
|---|---:|---|---|
| cuda:0 | 1 | true | none |
| cuda:0 | 2 | true | none |
| cuda:0 | 4 | true | none |
| cpu | 1 | true | none |
| cpu | 2 | true | none |
| cpu | 4 | true | none |

Each width produced its own distinct reference digest, so the sweep distinguishes its conditions rather than collapsing them.

## Why this is the necessary half of the result

Without it, [[warp-renderer-depth-is-not-device-portable]] would be uninterpretable. A renderer that varied run to run *within* a device would produce differing depth buffers across devices for the trivial reason that it produces differing depth buffers against itself, and the cross-device comparison would carry no information.

Because the renderer is exactly reproducible on each device taken alone, the cross-device disagreement can only be attributed to the device.

## What the suite pins

Three properties are asserted on every run, because each of them, if false, would make a determinism verdict meaningless while still reading as a pass:[^2]

- **The image changes between steps.** A renderer returning one fixed frame would reproduce perfectly and measure nothing.
- **Worlds within a batch differ from each other.** Identical worlds would make inter-world variability trivially zero.
- **The image is not uniform.** A camera pointed at nothing renders a constant, which also reproduces perfectly and also measures nothing.

Colour and depth are additionally measured on their own, not only together, so that a future divergence names a channel rather than only a step. That separation is what made the depth-specific finding legible when it appeared.


## What this does not establish

One camera, one resolution, three batch widths, on a scene with a single body and no body-to-body contact. Shadows, textures and multiple cameras were left at the adapter's settings and not varied.

The open case is a rendered rollout whose *physics* is irreproducible, which is what any scene above the coupled-body threshold would be. The raycaster itself is clean under a frozen state ([[the-raycaster-inherits-nondeterminism-it-does-not-create-it]]), but whether a rendered trial's divergence tracks the state divergence exactly has not been measured.

[^1]: `[observed]` — `record_trial` on each device at world counts 1, 2 and 4 with `TrialSpec(seed=7, step_count=60, repetitions=3)`, channel `both`; every recording reported `deterministic=true` and a distinct `reference_digest`.
[^2]: `tests/adapters/test_mjx_warp_render.py::TestRenderedContent` — `test_the_rendered_image_changes_as_the_scene_moves`, `test_worlds_within_a_batch_render_differently`, `test_the_image_is_not_uniform`; and `TestRenderedTrial::test_depth_alone_is_reproducible` / `::test_colour_alone_is_reproducible`.
