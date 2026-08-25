# NavProbe Wiki

**Read this first.** 5 topic hubs, 30 content pages. Follow the hub link for your topic; each hub lists its pages with one-line descriptions.

NavProbe is a reproducibility instrument for simulated navigation. This wiki records two things kept deliberately apart: **what the instrument was built to be** (design decisions and their reasoning) and **what it has measured** (results, with the conditions they were taken under).

**Context for all of it:** Warp 1.16 ships an explicit determinism control with a
`RUN_TO_RUN` and a `GPU_TO_GPU` mode, and MuJoCo-Warp 3.11.0 **cannot compile under
either as shipped** — one function mixes `max` and `add` reductions on one array, and it
blocks unconditionally, on models with no sensors at all. Every result below was measured
in the default mode, `NOT_GUARANTEED`, which is the only mode a MuJoCo-Warp user has
without patching their install.

**That is now a setting rather than a property, and it changes how to read everything
below.** A three-line alias patch clears the block, and under `RUN_TO_RUN` all ten scenes
— including every coupled-body scene that gives twelve different answers from twelve runs
by default — reproduce bit for bit ([[tactile-alias-patch-clears-warp-deterministic-compile]]).
So the failures below are real, reachable by anyone running the shipped package, and
**fixable**: they describe the default, not a limit of the hardware or the solver. What
that costs is [[deterministic-mode-cost-falls-with-scale]]: a few-fold slowdown that
*falls* as parallelism rises. Read the direction, not the numbers — that ladder's
figures were measured on a solve the contact buffer had silently truncated, and its
wall clocks move by up to 4x with host load.

**The headline results so far**, all in the default mode. Three separate reproducibility failures, none of which a simple benchmark would surface. Read the figures as instances rather than constants — every precise number below moved when re-measured on a different scene, while every shape held ([[the-numbers-are-scene-dependent-the-shapes-replicate]]):

1. **GPU run-to-run, once bodies touch each other.** Twelve runs of one rollout in one process give twelve different answers once six bodies are in mutual contact — while thirty-two bodies resting only on the floor, with *four times* the contacts, reproduce exactly. The CPU reproduces in every case. Contact count is not the variable; what the contacts connect is. And it is not last-bit trivia: at thirty-two touching bodies the disagreement amplifies chaotically from 10⁻⁸ m to the scale of the container within two simulated seconds, while the CPU stays at exactly zero. Every other measurement on this wiki is on a scene with no body-to-body contact, so all of them sit below the threshold by construction.
2. **The contact solver, across backends.** CPU and CUDA agree exactly through free flight and part at the step the scene's first contact is solved — step 57 here, for both MJX-on-XLA and MuJoCo-Warp, because that is when the ball lands. The divergence point is a fact about the trajectory, not the simulator.
3. **The depth raycaster, across devices.** From bit-identical inputs, 34% of depth pixels differ between Warp's CPU and CUDA devices, from the very first frame. Colour compares equal throughout, because 8-bit quantisation rounds the difference away — so the failure is invisible in the channel people check and present in the one depth-based policies consume.

## Hubs

[Determinism Measurement](hubs/determinism-measurement.md) -- what reproducible means here, and every physics trial this instrument has run (13 pages)
[Rendered Observations](hubs/rendered-observations.md) -- the batch renderer, and whether the pixel stream a policy consumes reproduces (3 pages)
[Instrument Design](hubs/instrument-design.md) -- canonical encoding, digest folding, record formats, and the injectivity obligations behind them (7 pages)
[Simulator Adapters](hubs/simulator-adapters.md) -- the vendor boundary: typing untyped APIs, keeping the declarations honest, and what each backend requires (3 pages)
[Platform Constraints](hubs/platform-constraints.md) -- which measurements are reachable from which machine, and why (4 pages)

**What is still open** is a page rather than a memory: see
[Open questions, and what would answer each one](pages/open-questions-and-what-would-answer-them.md),
which lists what is not known, ordered by value, with the experiment that would
settle each — and a short list of what is *not* open, so effort does not go there.

## How this works

**Three tiers:** this index (read every session) → hub pages (read when topic matches) → content pages (read when you need the facts). A content page can be linked from multiple hubs.

**Adding new content:** create the content page in `pages/`, add an inclusion link from the relevant hub(s), bump page counts here. If the topic needs a new hub, create it in `hubs/` and add one line above.

**The rules:** see `SCHEMA.md` — atomicity, frontmatter, citations, hub-link discipline. Two rules are specific to this wiki and easy to miss: a page reporting a measurement carries a `measured_with` block, and a claim about this package's behaviour is cited by the test that enforces it.
