---
title: Open questions, and what would answer each one
tags: [methodology, roadmap, open-questions]
related: ["[[the-numbers-are-scene-dependent-the-shapes-replicate]]", "[[mjwarp-cannot-compile-under-warp-deterministic-mode]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]"]
source_paths:
  - "wiki/log.md"
source_git_blobs:
  "wiki/log.md": "cebbc930c6e6ace87cbff5328cca025f92811f7e"
fact_checked: 2026-08-19
confidence: high
hubs: [instrument-design]
---

# Open questions, and what would answer each one

Every measurement page states its own limits. This is the consolidated view: what is not known, and specifically what would settle it. It exists so that "what is left" is a page rather than a memory.

Ordered by value, not by effort.

## 1. Is `_sensor_tactile` the only blocker? — ANSWERED 2026-08-18: yes

Warp's deterministic modes would make most of this wiki's findings a *setting* rather than a property, and MuJoCo-Warp cannot compile under them ([[mjwarp-cannot-compile-under-warp-deterministic-mode]]). Compilation stops at the first rejected function, so a fix there could have revealed the next one.

**It did not.** With the withdrawn upstream PR #1591's three-line alias patch applied, every module in the touching-row pipeline compiles cold and steps under BOTH `RUN_TO_RUN` and `GPU_TO_GPU` ([[tactile-alias-patch-clears-warp-deterministic-compile]]). The solver kernels this wiki attributes the non-determinism to were already deterministic-mode-clean. **And the GPU verdict is in, same day: all ten scenes bit-reproducible under `RUN_TO_RUN` on `cuda:0`, including every coupled-body scene, with `deterministic_max_records` raised to 64 to clear a runtime record-buffer overflow at 32 bodies.** The cost and cross-process checks landed the same night on the freed GPU: warm-run cost 3.3 to 7.1x (overall 5.07x at world_count 2, worst in the most-coupled scenes), and three separate processes under different load produced 10/10 identical digests. This question is closed end to end; what it leaves behind is scale-dependence of the cost (world_count 2 under-occupies the card) and cross-machine digest repetition, both folded into question 2's hardware axis.

**Upstream is answering a stronger version of this question by another route.** As of 2026-08-18 `google-deepmind/mujoco_warp` carries an unmerged three-PR stack (#1422, #1425, #1533, none merged) that builds an opt-in `opt.deterministic` stepping mode at the MJWarp level — sorted contacts, count-based row allocation, serial sensor scans — rather than through Warp's codegen mode. PR #1533's description enumerates every remaining order-sensitive accumulation in the pipeline, which is, in effect, upstream's own list of "the next blockers". An independent third party also filed and withdrew (CLA failure, 16 minutes) an issue + fix pair for exactly the `_sensor_tactile` mix (#1590/#1591) on 2026-08-18. The local-patch experiment is still worth running: it tests Warp's codegen mode, which the upstream stack deliberately bypasses.

**Also unmeasured:** the throughput cost of deterministic atomics, because nothing compiled and so nothing was timed. Warp 1.15's own documentation publishes a cost table (RTX 4090: deterministic sort-and-reduce is up to 7.6x *faster* than atomics at high contention, ~13x slower at low contention), which bounds the expectation but does not measure MJWarp.

**The equivalence check this question deferred is now done, and it opened question 1b.** The patch was trusted for *compilation* only; whether the aliased writes changed any numbers was untested because the scene family has `nsensor = 0`. Measured 2026-08-29 on the vendor's own tactile fixture: they do not — patched and unpatched are bit-identical on both devices with 12 live taxels ([[tactile-alias-is-inert-with-live-taxels]]). The patch is safe to file. What the check found instead is below.

## 1b. Which kernel drops the contacts on a mesh/multiccd model under deterministic mode?

Under `RUN_TO_RUN` the vendor tactile fixture generates **zero contacts** and the body falls through the box it should rest on — reproducibly, at adequate buffer capacity, with no exception and exit 0 ([[deterministic-mode-drops-contacts-on-mesh-collision]]). The sphere family is unaffected, so this is specific to the mesh-vs-box collision path, not to deterministic mode in general.

This is the highest-value open item on this page, because it is a **correctness** failure rather than a reproducibility one, and because it decides how an upstream bug report is scoped.

**`multiccd` is ruled out, 2026-08-29.** It was the obvious suspect and the cheapest test, so it went first: rebuilding the identical fixture without `<flag multiccd="enable"/>` changes nothing in either mode — default keeps its 60 contacts and final z 1.149779, `RUN_TO_RUN` still drops to zero contacts and 0.981754, one distinct digest from six repetitions in all four cells. The flag does not affect this scene at all and removing it does not rescue the contacts.

**What would answer the rest, cheapest first:**

1. **Bisect the deterministic lowering by kernel.** Warp intercepts atomics per kernel; a build that leaves `collision_driver`'s kernels on ordinary atomics while the rest stays deterministic would locate the failure to a module. This is now the next experiment.
2. **Vary the geometry pair.** The sphere family (primitive-vs-plane, primitive-vs-primitive) is clean and this fixture (mesh-vs-box) is not. A mesh-vs-plane and a primitive-vs-box arm would say whether "mesh" or "box" is the operative half, which is one more axis the bug report can name.
3. **`_preprocess_tactile_contacts` is a bystander, not a suspect.** It consumes an `atomic_add` *return value* as a slot index — the two-pass counter-replay lowering, the most fragile construct in the pipeline. But it runs downstream of `d.nacon`, which is already zero, so it never had contacts to deduplicate. Noted here so the next reader does not re-open it.

**Also unresolved and worth stating:** whether `deterministic_max_records = 64` — the value question 1 adopted and this wiki still publishes — silently truncated anything in the runs that produced the ten-scene table. Those runs pass a contact check ([[a-determinism-verdict-needs-a-correctness-oracle]]), so nothing is known to be wrong; but the guard that would have said so is skipped under CUDA graph capture, so "no error was raised" is not evidence.

## 2. Do the findings hold on another GPU architecture?

Everything GPU-side was measured on two Ampere sm_86 devices: the RTX 3090 Ti (84 SMs) and, since 2026-08-16, sedona's RTX 3070 Ti Laptop (46 SMs), where the coupled-body threshold did not move ([[coupled-body-threshold-does-not-move-with-sm-count]]). That falsifies the occupancy explanation *within* an architecture but says nothing about cross-architecture codegen: both devices compile to the same sm_86 target. A threshold that is a codegen or warp-scheduling artefact could still move on a different architecture; one that is algorithmic should not.

**What would answer it:** run the same sweep on a non-Ampere card — the sm_75 GTX 1630, in the post as of 2026-08-19. The instrument needs no change of *design*; the sweep scripts gained a `--device` flag the same day so a host holding two cards can address each one, and the resolved device is recorded in every report ([[measurement-fleet-is-reachable-by-ssh-alias]] lists what each machine can host).

**The card goes in `austinpc`, beside the 3090 Ti — not in a second machine.** That is a measurement decision, not a convenience one. Question 2 asks to isolate *architecture*; putting the 1630 in the one box that already holds an Ampere card holds the OS, the driver, the CPU and the RAM fixed, leaving the architecture as the only variable. Housing it in `lavender` — the only other machine with a free slot — would introduce a second driver branch on top, which is exactly the confound [[coupled-body-threshold-does-not-move-with-sm-count]] already had to caveat when `sedona` ran driver 551.23 against this host's 13.1. The board is a full-ATX MSI PRO Z790-P WIFI. **Its second x16-length slot is not x16 electrical**, which the plan originally assumed: MSI's specification for this SKU gives `PCI_E1` PCIe 5.0 x16 from the CPU (where the 3090 Ti sits), `PCI_E2` PCIe 3.0 **x1**, `PCI_E3` PCIe 4.0 **x4**, and `PCI_E4` PCIe 3.0 x1 — so exactly one slot on the board carries meaningful bandwidth and it is already occupied. The 1630 measured Gen3 x16 (width 16/16) in `lavender`; in `austinpc` it would negotiate x1 or, in `PCI_E3`, x4.

**This does not invalidate the experiment.** Host-to-device bandwidth is transfer time; it does not touch codegen, block scheduling or accumulation order, which is what a determinism verdict and [[coupled-body-threshold-turns-on-one-kernels-block-size]] are about. A verdict taken over an x1 link is as valid as one taken over x16. It does mean the two cards pay unequal per-transfer tax, so **any timing figure from this task is not comparable across them, and `measured_with` must name each card's slot and negotiated link rather than just the device** — `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` is the query that produces it.

Physical checks before moving the card, in order: **PSU headroom — answered, 1000 W.** Whether the 3090 Ti's 3-slot cooler blocks `PCI_E2` — open. And if it does, check clearance to `PCI_E3` too, because at PCIe 4.0 x4 that is the better slot anyway; a blocked `PCI_E2` is not fatal to the plan. The 1630 is 2-slot and 75 W slot-powered, so nothing about cabling constrains where it lands. Lane counts here are vendor documentation, not a reading off this board — confirm with `pcie.link.*` once a card is actually seated, since a negotiated link is the only proof of what a slot delivered.

**The recipe**, once the card is in — one run per device, then compare the decoded records rather than the terminal output:

```
python -m scripts.gpu_deterministic_sweep RUN_TO_RUN <fresh-cache-0> 64 --device cuda:0 --linesearch-block-dim 32
python -m scripts.gpu_deterministic_sweep RUN_TO_RUN <fresh-cache-1> 64 --device cuda:1 --linesearch-block-dim 32
```

Each writes a `navprobe-sweep-run/2` document; `navprobe.codecs.sweep_run.decode_sweep_run` reads both, and the comparison is scene-by-scene on the verdicts. A fresh cache directory per device is not optional — a shared one would let the second run load the first's compiled kernels, which is the one thing a codegen question must not allow.

**`--linesearch-block-dim 32` is not decoration either.** It pins the one setting known to move the verdict ([[coupled-body-threshold-turns-on-one-kernels-block-size]]), and 32 specifically because that is the vendor default every published figure on this wiki was measured under — so the comparison extends the existing corpus rather than starting a new one. Passing it explicitly rather than relying on the default is what makes the report *say* which block size ran: the value is carried in the record, so two sweeps that pinned different values are visibly incomparable instead of quietly so. Omit it on one device and the whole experiment is uninterpretable, because a moved threshold could be the architecture or could be the block size.

**It also makes `GPU_TO_GPU` testable for the first time.** That mode compiles under the alias patch ([[tactile-alias-patch-clears-warp-deterministic-compile]]) but has never been swept, because a mode whose entire claim is *the same digest on different devices* cannot be tested with one architecture. Two architectures in one box turns it into a real experiment, and it absorbs the cross-machine digest repetition left over from question 1. If the digests match across sm_75 and sm_86 under `GPU_TO_GPU`, coupled-body reproducibility is portable rather than merely per-device — a stronger result than anything on this wiki so far.

**One constraint to plan around:** 4 GiB of VRAM. The scale ladder needed `constraint_capacity` right-sized to 256 to fit 4096 worlds on a 24 GiB card ([[deterministic-mode-cost-falls-with-scale]]), so the top rungs will cap lower here. Irrelevant to the threshold sweep, which runs at `world_count = 2`.

**The cheaper partial answer has now been taken, and it changes how this question must be run.** MJWarp exposes `Model.block_dim` — not a scalar but a 23-field struct — and varying it on this hardware moves the 5-body scene from 0/20 reproducing to 17/20, on a bit-identical trajectory. It localises to one field, `linesearch_iterative`, at one value, 64 ([[coupled-body-threshold-turns-on-one-kernels-block-size]]). **Consequence for the sweep below: `linesearch_iterative` must be pinned explicitly on BOTH devices and recorded in `measured_with`.** A default that differs by architecture or tile constraint would produce a "threshold moved" result that is block size and nothing to do with sm_75 versus sm_86.

**What is no longer needed for this question:** `emerald`. Its one job was the AVX-without-AVX2 rung, and that axis is closed by construction — MuJoCo's shipped binary will not import below AVX ([[mujoco-requires-avx-so-pre-avx-hosts-are-ineligible]]), so no processor exists that could extend it. It has no discrete GPU and no unmeasured CPU property, so it answers nothing here. Its pending Ubuntu reimage was a Windows-10-end-of-life plan, not a measurement need.

## 3. Does the coupled-body threshold survive other geometry?

Sphere-sphere contact only, one radius, one spacing, one timestep. Whether meshes, boxes, or a real manipulation scene move the boundary — or remove it — is unmeasured.

**What would answer it:** new `SceneSpec` variants and a sweep. This is now cheap: the scene family is data and the sweep is a function, so it is new values rather than new code.

## 4. What is the distribution of divergence points?

The cross-backend divergence begins at the first contact solve ([[backend-divergence-begins-at-first-contact]]), which for this trajectory is step 57. That is one trajectory. Across seeds and scenes it is a distribution, and nothing here estimates it.

**What would answer it:** the same sweep over seeds rather than over scene sizes, recording `first_divergent_step` per seed.

## 5. Why six?

The coupled-body boundary sits at five or six bodies depending on the harness. Nothing here inspects a kernel, so the mechanism is a hypothesis: a scheduling or block-size boundary rather than anything physical.

**What would answer it:** reading the generated CUDA. The `block_dim` sweep this question used to defer to has been run, and it narrows where to read: the boundary case reproduces when `linesearch_iterative`'s block size is 64 and not at 32, 96, 128 or 256, while the `JTDAJ` accumulation kernels — the obvious suspects — do nothing ([[coupled-body-threshold-turns-on-one-kernels-block-size]]). So the question is now specific: what does that kernel's generated CUDA do differently at exactly two warps? No mechanism is claimed; "only 64" is a fact awaiting one.

## 6. Does the renderer stay clean under an irreproducible trajectory?

The raycaster adds no run-to-run variance of its own ([[the-raycaster-inherits-nondeterminism-it-does-not-create-it]]), measured by freezing a state and re-rendering it. It has not been measured while the *physics* is irreproducible — which is the case any real manipulation scene would be in.

**What would answer it:** a rendered trial on a coupled-body scene above the threshold, comparing whether the rendered divergence tracks the state divergence exactly or exceeds it.

## 7. What does deterministic mode actually cost? (the published answer is unsound)

Listed last because it is a methodology gap rather than a physics unknown, but it
invalidates a published page, so it is not low priority.

[[deterministic-mode-cost-falls-with-scale]] reports a cost curve measured over
world counts. An attempted re-measurement on 2026-08-20 found three problems with
it, two of which are fatal to the numbers:

- **The solve was truncated.** Every rung ran at `constraint_capacity = 256`, and
  the archived log carries 5,232 `broadphase overflow` warnings asking for roughly
  581. MuJoCo-Warp reports the overflow as a warning, so the ladder completed and
  reported throughput for less work than the scene specifies. The capacity had been
  chosen so 4096 worlds would *fit*, not so the scene would *solve*.
- **The wall clocks are not reproducible.** The same ladder re-run came back 1.2x to
  4.0x slower per rung, with a non-monotonic curve (64 worlds slower than 512),
  because a SIRIUS job was holding 68% CPU throughout. Each rung is timed **once**.
- **Power throttling was not the cause**, despite this wiki briefly saying so. It did
  not reproduce when measured directly here.

The cost figures on [[tactile-alias-patch-clears-warp-deterministic-compile]] are in
better shape: they ran at capacity 8192 with **zero** overflows, so they measure the
full solve, and both arms ran minutes apart in one session, which protects a ratio
better than an absolute. They are still single-shot.

**What would answer it:** re-run the ladder with (a) a capacity that does not overflow
— roughly 1024, verified by grepping the run's own log for `broadphase overflow` rather
than by trusting the setting, (b) repetitions per rung with the **minimum** reported,
since the minimum is the sample least contaminated by whatever else the machine was
doing, and (c) an idle host, verified by total CPU load and not merely by GPU
utilisation, which is the check that missed SIRIUS. If (a) forces a lower ceiling than
4096 worlds on 24 GiB, report the lower ceiling: a ladder that stops at 512 with a
complete solve says more than one that reaches 4096 with a truncated one.

## What is *not* open

Worth stating, so effort does not go here:

- **Whether the CPU is a valid reference.** It reproduced exactly in every condition tested, including the ones where the GPU produced twelve distinct results from twelve runs.
- **Whether the renderer or the solver causes rendered non-determinism.** Isolated: the solver.
- **Whether TF32 explains the cross-backend divergence.** Ruled out — `JAX_DEFAULT_MATMUL_PRECISION=highest` gives a byte-identical digest on this Ampere GPU.
- **Whether contact count drives the GPU failure.** Ruled out: 64 independent contacts reproduce, 14 coupled ones do not.
