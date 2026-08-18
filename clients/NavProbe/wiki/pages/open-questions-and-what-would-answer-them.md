---
title: Open questions, and what would answer each one
tags: [methodology, roadmap, open-questions]
related: [[the-numbers-are-scene-dependent-the-shapes-replicate]], [[mjwarp-cannot-compile-under-warp-deterministic-mode]], [[warp-gpu-determinism-fails-on-coupled-bodies]]
sources: [wiki/log.md]
fact_checked: 2026-08-14
confidence: high
---

# Open questions, and what would answer each one

Every measurement page states its own limits. This is the consolidated view: what is not known, and specifically what would settle it. It exists so that "what is left" is a page rather than a memory.

Ordered by value, not by effort.

## 1. Is `_sensor_tactile` the only blocker?

Warp's deterministic modes would make most of this wiki's findings a *setting* rather than a property, and MuJoCo-Warp cannot compile under them ([[mjwarp-cannot-compile-under-warp-deterministic-mode]]). Compilation stops at the first rejected function, so a fix there may simply reveal the next one.

**What would answer it:** patch `_sensor_tactile` locally to stop mixing `max` and `add` reductions on one array — separate accumulators, or a second pass — and recompile. Either it builds, in which case the whole determinism story changes, or the next blocker names itself.

**Also unmeasured:** the throughput cost of deterministic atomics, because nothing compiled and so nothing was timed.

## 2. Do the findings hold on another GPU architecture?

Everything GPU-side was measured on one RTX 3090 Ti (Ampere, sm_86). A threshold that is a warp-scheduling artefact should move on a different architecture; one that is algorithmic should not.

**What would answer it:** run the same sweep on an Ada or Hopper card. The instrument needs no change — it already ran unmodified across Windows and WSL2 — and the machines do not need to share a process or even a filesystem: each records an observation or a trial, and a third process compares the files ([[the-numbers-are-scene-dependent-the-shapes-replicate]] explains why the figures rather than the shapes are what to watch).

**Cheaper partial answer available first:** MJWarp exposes `Model.block_dim`. Varying it on *this* hardware tests the scheduling hypothesis without another card, and has not been done.

## 3. Does the coupled-body threshold survive other geometry?

Sphere-sphere contact only, one radius, one spacing, one timestep. Whether meshes, boxes, or a real manipulation scene move the boundary — or remove it — is unmeasured.

**What would answer it:** new `SceneSpec` variants and a sweep. This is now cheap: the scene family is data and the sweep is a function, so it is new values rather than new code.

## 4. What is the distribution of divergence points?

The cross-backend divergence begins at the first contact solve ([[backend-divergence-begins-at-first-contact]]), which for this trajectory is step 57. That is one trajectory. Across seeds and scenes it is a distribution, and nothing here estimates it.

**What would answer it:** the same sweep over seeds rather than over scene sizes, recording `first_divergent_step` per seed.

## 5. Why six?

The coupled-body boundary sits at five or six bodies depending on the harness. Nothing here inspects a kernel, so the mechanism is a hypothesis: a scheduling or block-size boundary rather than anything physical.

**What would answer it:** reading the generated CUDA, or the `block_dim` sweep in question 2 — which is why that one is worth doing first.

## 6. Does the renderer stay clean under an irreproducible trajectory?

The raycaster adds no run-to-run variance of its own ([[the-raycaster-inherits-nondeterminism-it-does-not-create-it]]), measured by freezing a state and re-rendering it. It has not been measured while the *physics* is irreproducible — which is the case any real manipulation scene would be in.

**What would answer it:** a rendered trial on a coupled-body scene above the threshold, comparing whether the rendered divergence tracks the state divergence exactly or exceeds it.

## What is *not* open

Worth stating, so effort does not go here:

- **Whether the CPU is a valid reference.** It reproduced exactly in every condition tested, including the ones where the GPU produced twelve distinct results from twelve runs.
- **Whether the renderer or the solver causes rendered non-determinism.** Isolated: the solver.
- **Whether TF32 explains the cross-backend divergence.** Ruled out — `JAX_DEFAULT_MATMUL_PRECISION=highest` gives a byte-identical digest on this Ampere GPU.
- **Whether contact count drives the GPU failure.** Ruled out: 64 independent contacts reproduce, 14 coupled ones do not.
