---
title: The raycaster inherits non-determinism, it does not create any
tags: [warp, rendering, determinism, measurement, gpu]
related: [[warp-renderer-depth-is-not-device-portable]], [[warp-gpu-determinism-fails-on-coupled-bodies]], [[warp-rendered-stream-is-reproducible-within-a-device]]
sources: [mujoco-warp 3.11.0, warp-lang 1.16.0]
fact_checked: 2026-08-14
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti
  model: rows of 1 and 16 spheres, separated (0.070 m) and mutually touching (0.055 m)
  camera: one camera, 64x64, rgb and depth
  world_count: 2
  settle_steps: 0 and 120
  repetitions: 12 renders of one frozen state, same process
---

# The raycaster inherits non-determinism, it does not create any

Every earlier rendered measurement stepped the physics and then rendered, so a varying image could have come from a varying state. Freezing the state and rendering it repeatedly separates the two.

Twelve renders of one fixed state produced **one** distinct colour digest and **one** distinct depth digest, in every condition — including a row of sixteen mutually-touching spheres settled for 120 steps, which is a scene whose *physics* is not reproducible at all.[^1]

| bodies | mutually touching | settle steps | distinct rgb | distinct depth |
|---:|---|---:|---:|---:|
| 1 | n/a | 0 | 1 | 1 |
| 1 | n/a | 120 | 1 | 1 |
| 16 | no | 0 | 1 | 1 |
| 16 | no | 120 | 1 | 1 |
| 16 | yes | 0 | 1 | 1 |
| 16 | yes | 120 | 1 | 1 |

A fresh render context was built for each repetition and the frozen state re-asserted before every render, so nothing the renderer allocates or caches carries between them.

## What this settles

The rendered observation stream has exactly one source of run-to-run variation on a fixed device, and it is upstream: the physics state. Given the same state, the raycaster returns the same pixels.

That is worth having because it makes the rendered pipeline diagnosable. A rendered rollout that fails to reproduce is a physics problem, and the fix — pin the device, or keep coupled-body counts below the threshold in [[warp-gpu-determinism-fails-on-coupled-bodies]] — is a physics fix. There is no second thing to chase.

## It does not extend across devices

Determinism on one device and portability between devices are different properties, and the renderer has the first without the second. From bit-identical state, the depth buffer differs between Warp's CPU and CUDA devices on about a third of pixels ([[warp-renderer-depth-is-not-device-portable]]).

So the complete picture for rendering:

| question | answer |
|---|---|
| same device, same state, repeated renders | identical |
| same device, repeated *rollouts* | identical only if the physics is |
| different devices, same state | colour identical, **depth differs** |

## What this does not establish

One camera, one resolution, one scene family, one GPU. Only the raycaster's own reproducibility was isolated; shadows, textures and multiple cameras were left at the adapter's settings and not varied. A renderer feature that introduced its own accumulation — an accumulating light pass, for instance — would not be covered by this measurement.

[^1]: `[observed]` — for each row, the state was advanced by the listed settle steps, its `qpos` copied to the host, and then re-asserted before each of twelve renders through a freshly created render context; colour and depth buffers were digested per render and the distinct-digest counts reported.
