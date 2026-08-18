---
title: The GPU's last-bit disagreement amplifies to macroscopic scale in contact-rich scenes
tags: [warp, determinism, measurement, finding, chaos, gpu]
related: [[warp-gpu-determinism-fails-on-coupled-bodies]], [[backend-divergence-begins-at-first-contact]]
sources: [mujoco-warp 3.11.0, warp-lang 1.16.0]
fact_checked: 2026-08-14
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0, with warp cpu as control
  device: NVIDIA GeForce RTX 3090 Ti
  model: single row of 32 spheres, radius 0.03 m, initialised in mutual lateral contact at 0.055 m spacing, dropped onto a plane between walls 1.96 m apart
  timestep: 0.005
  step_count: 400 (2.0 s simulated)
  world_count: 2
  repetitions: 8, same process
---

# The GPU's last-bit disagreement amplifies to macroscopic scale in contact-rich scenes

That twelve runs produce twelve digests says they differ. It does not say anyone would notice. In a contact-rich scene, they would: the difference starts at the last representable bit and grows until it is the size of the container.

## Growth of the spread across eight identical runs

Maximum spread of any body's position across eight runs of the same rollout, on the same GPU, in one process:

| step | simulated time | max spread |
|---:|---:|---:|
| 25 | 0.125 s | 4.47 × 10⁻⁸ m |
| 50 | 0.250 s | 1.14 × 10⁻⁷ m |
| 100 | 0.500 s | 1.14 × 10⁻⁴ m |
| 200 | 1.000 s | 8.17 × 10⁻² m |
| 300 | 1.500 s | 2.95 × 10⁻¹ m |
| 400 | 2.000 s | 5.18 × 10⁻¹ m |

Eight orders of magnitude in two seconds of simulated time, saturating at the scale of the 1.96 m box. The CPU control is **exactly zero** at every checkpoint.[^1]

This is ordinary chaotic amplification — a contact-rich pile is a chaotic system, and any perturbation grows exponentially until geometry bounds it. What makes it worth recording is the size of the perturbation being amplified: not a modelling choice, not a seed, but the order in which a GPU accumulated a sum.

## The magnitude is not a number to quote

Two measurement sessions of the same configuration gave 0.246 m and 0.518 m. That is not an inconsistency, it is the phenomenon: the final spread is itself a chaotic outcome and has no stable value. The reproducible facts are the *growth* and the *saturation scale*, not the endpoint.

Anyone quoting a single divergence figure for a contact-rich scene is quoting one sample from a distribution.

## The result is physically real, not a blown-up simulation

Checked on the final state, because a spread of half a metre would also be produced by an unstable integration:[^2]

- Every sphere rests at z = 0.02963 against a resting height of 0.03, i.e. normal soft-contact penetration.
- All bodies are inside the walls: max \|x\| of 0.927 against a wall at 0.980.
- All values finite, all bodies above the floor.
- Maximum speed 0.18 m/s — a pile still jostling, not one exploding.

The rollouts are well-behaved simulations that end in different places.

## Where the threshold and the significance separate

These are two different questions, and they have two different answers:

| bodies | bit-identical across runs? | spread at 400 steps |
|---:|---|---:|
| 6 | no | 1.75 × 10⁻⁸ m |
| 16 | no | 5.62 × 10⁻⁶ m |
| 32 | no | ~10⁻¹ m |

Six mutually-contacting bodies are enough to lose bit-reproducibility ([[warp-gpu-determinism-fails-on-coupled-bodies]]), but at six the divergence stays at the last bit for the whole rollout. Thirty-two bodies is where it becomes something a person could see.

So bit-level reproducibility is a *leading indicator*. It fails long before the failure is visible, which is exactly what makes it useful to measure — a digest comparison catches at six bodies what a position tolerance would not catch until thirty-two.

## Consequences

- **A tolerance-based comparison will pass and then suddenly fail.** Any threshold on positions is met easily for the first second and violated shortly after, with no change in the setup.
- **Evaluation results on contact-rich scenes are not reproducible on GPU.** Two runs of the same policy on the same seeds end with objects in different places, so any metric reading final configuration is a sample rather than a measurement.
- **Reporting a mean over GPU runs is fine; reporting a single run is not.** The single run is unrepeatable even by its own author.

## What this does not establish

One scene family, one GPU, one solver configuration. The growth rate is a property of this pile and would differ for other geometries; nothing here estimates a Lyapunov exponent, and the checkpoint table is one sample of eight runs, not a fitted rate.

Whether the same amplification occurs across *backends* rather than across runs on one backend is a separate question, and it is closely related to [[backend-divergence-begins-at-first-contact]] — that divergence also begins at the last bit, at the first contact solve.

[^1]: `[observed]` — eight rollouts of 400 steps in one process, positions sampled at the listed checkpoints; the reported figure is the largest bounding-box diagonal of any single body's positions across the eight runs. The same script under `wp.set_device("cpu")` gave 0.000000e+00 at 6, 16 and 32 bodies.
[^2]: `[observed]` — final state of run 0 at 32 bodies: z range 0.02963 to 0.02963, max \|x\| 0.92655 against a wall at 0.980, max speed 1.799981e-01 m/s, all values finite and all bodies above the floor.
