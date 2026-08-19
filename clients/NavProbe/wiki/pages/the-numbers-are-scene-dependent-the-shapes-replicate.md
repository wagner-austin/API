---
title: The numbers here are scene-dependent; the shapes are what replicate
tags: [methodology, reproducibility, measurement-design]
related: ["[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[warp-renderer-depth-is-not-device-portable]]", "[[a-scene-is-a-value-not-a-string-literal]]"]
source_paths:
  - "wiki/log.md"
  - "src/navprobe/scenes.py"
source_git_blobs:
  "wiki/log.md": "39302407f39e0dafc6935a927d3a0171efdf51f2"
  "src/navprobe/scenes.py": "4a05c692fbd2740bd717f015e7725fa8175fc207"
fact_checked: 2026-08-14
confidence: high
hubs: [instrument-design]
---

# The numbers here are scene-dependent; the shapes are what replicate

Two findings on this wiki were measured twice — once by a standalone script and once through the package's own scene family and codecs. Both times the qualitative result held exactly and **every precise figure moved**.

| finding | first measurement | second measurement | what held |
|---|---|---|---|
| coupled-body threshold | reproduces at 5 bodies, fails at 6 | reproduces at 4, fails at 5 | separated rows reproduce at every size; touching rows fail at a handful |
| cross-device depth | 2,794 / 8,192 pixels (34.1 %), max 1.47 × 10⁻⁵ | 3,492 / 8,192 (42.6 %), max 1.43 × 10⁻⁶ | colour identical, depth differing on a large fraction at ~10⁻⁶ |

Neither difference was an error in either measurement. The scenes differed — a per-world seeded perturbation in one case, a different camera and body placement in the other — and the figures are properties of the scene, not of MuJoCo-Warp.

## How to read a figure on this wiki

Every measurement page carries a `measured_with` block. That block is not provenance decoration; it is **the scope of the number above it**. A figure quoted without its block is a figure quoted without its meaning.

Concretely:

- **A threshold** ("fails at six bodies") is a property of a configuration. Quote it with the perturbation and the spacing or do not quote it.
- **A fraction** ("34 % of depth pixels") is a property of a scene and a camera. It is not a property of the renderer.
- **A divergence step** ("step 57") is a property of a trajectory — in that case, when the ball lands.
- **A magnitude that saturates** (the chaotic spread reaching container scale) has no stable value at all; two sessions of the identical configuration gave 0.246 m and 0.518 m.

What replicates across all of it is the **shape**: which variable matters, which direction the effect runs, and roughly what order of magnitude it reaches.

## Why this kept being discoverable

Because both harnesses could be pointed at the same scene definition. While a scene lived in a string literal inside whichever script produced it, two measurements of "the same" family were not comparable and a moved boundary would have looked like a stable fact ([[a-scene-is-a-value-not-a-string-literal]]).

Making the scene a value did not make the numbers stable. It made their instability visible, which is the more useful outcome — a number that quietly depends on something unrecorded is worse than one known to depend on something recorded.

## The rule this implies

State the shape as the finding and the number as an instance of it. "GPU determinism fails once a handful of bodies touch each other, at six in this configuration" survives re-measurement; "GPU determinism fails at six bodies" does not.

Both pages above were rewritten to that form after the second measurement, and the `measured_with` blocks were extended to name the harness rather than only the parameters.
