---
title: Open questions, and what would answer each one
tags: [methodology, roadmap, open-questions]
related: ["[[the-numbers-are-scene-dependent-the-shapes-replicate]]", "[[mjwarp-cannot-compile-under-warp-deterministic-mode]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]"]
source_paths:
  - "wiki/log.md"
source_git_blobs:
  "wiki/log.md": "6a3636d6f8d3d1c4b0f4425d8c0a8764cd2953eb"
fact_checked: 2026-08-18
confidence: high
hubs: [instrument-design]
---

# Open questions, and what would answer each one

Every measurement page states its own limits. This is the consolidated view: what is not known, and specifically what would settle it. It exists so that "what is left" is a page rather than a memory.

Ordered by value, not by effort.

## 1. Is `_sensor_tactile` the only blocker? — ANSWERED 2026-08-18: yes

Warp's deterministic modes would make most of this wiki's findings a *setting* rather than a property, and MuJoCo-Warp cannot compile under them ([[mjwarp-cannot-compile-under-warp-deterministic-mode]]). Compilation stops at the first rejected function, so a fix there could have revealed the next one.

**It did not.** With the withdrawn upstream PR #1591's three-line alias patch applied, every module in the touching-row pipeline compiles cold and steps under BOTH `RUN_TO_RUN` and `GPU_TO_GPU` ([[tactile-alias-patch-clears-warp-deterministic-compile]]). The solver kernels this wiki attributes the non-determinism to were already deterministic-mode-clean. **And the GPU verdict is in, same day: all ten scenes bit-reproducible under `RUN_TO_RUN` on `cuda:0`, including every coupled-body scene, with `deterministic_max_records` raised to 64 to clear a runtime record-buffer overflow at 32 bodies.** What remains of this question is only the cost: the sweep ran under co-resident training load, so a clean-GPU throughput comparison (deterministic vs default) is the outstanding measurement, along with cross-process digest repetition.

**Upstream is answering a stronger version of this question by another route.** As of 2026-08-18 `google-deepmind/mujoco_warp` carries an unmerged three-PR stack (#1422, #1425, #1533, none merged) that builds an opt-in `opt.deterministic` stepping mode at the MJWarp level — sorted contacts, count-based row allocation, serial sensor scans — rather than through Warp's codegen mode. PR #1533's description enumerates every remaining order-sensitive accumulation in the pipeline, which is, in effect, upstream's own list of "the next blockers". An independent third party also filed and withdrew (CLA failure, 16 minutes) an issue + fix pair for exactly the `_sensor_tactile` mix (#1590/#1591) on 2026-08-18. The local-patch experiment is still worth running: it tests Warp's codegen mode, which the upstream stack deliberately bypasses.

**Also unmeasured:** the throughput cost of deterministic atomics, because nothing compiled and so nothing was timed. Warp 1.15's own documentation publishes a cost table (RTX 4090: deterministic sort-and-reduce is up to 7.6x *faster* than atomics at high contention, ~13x slower at low contention), which bounds the expectation but does not measure MJWarp.

## 2. Do the findings hold on another GPU architecture?

Everything GPU-side was measured on two Ampere sm_86 devices: the RTX 3090 Ti (84 SMs) and, since 2026-08-16, sedona's RTX 3070 Ti Laptop (46 SMs), where the coupled-body threshold did not move ([[coupled-body-threshold-does-not-move-with-sm-count]]). That falsifies the occupancy explanation *within* an architecture but says nothing about cross-architecture codegen: both devices compile to the same sm_86 target. A threshold that is a codegen or warp-scheduling artefact could still move on a different architecture; one that is algorithmic should not.

**What would answer it:** run the same sweep on a non-Ampere card (the pending sm_75 GTX 1630 purchase, or any Ada/Hopper device). The instrument needs no change — it already ran unmodified across Windows and WSL2 — and the machines do not need to share a process or even a filesystem: each records an observation or a trial, and a third process compares the files ([[the-numbers-are-scene-dependent-the-shapes-replicate]] explains why the figures rather than the shapes are what to watch).

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
