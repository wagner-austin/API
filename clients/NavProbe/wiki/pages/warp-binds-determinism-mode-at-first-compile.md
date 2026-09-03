---
title: Warp binds determinism mode when a module first compiles, so an in-process mode flip tests nothing
tags: [warp, determinism, methodology, testing, false-positive]
related: ["[[kernel-split-fix-restores-convex-contacts-at-upstream-head]]", "[[a-determinism-verdict-needs-a-correctness-oracle]]", "[[tactile-alias-is-inert-with-live-taxels]]"]
provenance:
  - "warp-lang 1.15.0"
  - "google-deepmind/mujoco_warp @ 3879591"
fact_checked: 2026-08-31
confidence: high
measured_with:
  package: mujoco_warp @ git 3879591 + working-tree fix
  warp: 1.15.0
  backend: warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, host austinpc)
  modes: [NOT_GUARANTEED, RUN_TO_RUN]
  procedure: same test run with the fix enabled and force-disabled
hubs: [determinism-measurement]
---

# Warp binds determinism mode when a module first compiles, so an in-process mode flip tests nothing

Warp resolves `wp.config.deterministic` into a module's generated code at the moment that
module is first built for a device. Setting the config afterwards changes nothing for anything
already compiled: kernels keep the codegen they were born with, silently. There is no error,
no warning, and no recompile.[^1]

The consequence for measurement is severe. A test that runs a workload, flips
`wp.config.deterministic = RUN_TO_RUN`, and runs the workload again is comparing the same
compiled kernels against themselves. It passes whether or not the deterministic path works —
the exact shape of false positive [[a-determinism-verdict-needs-a-correctness-oracle]] warns
about, produced this time by the test harness rather than the metric.

This was not caught by reasoning; it was caught by the negative control. A first regression
test for the fix on [[kernel-split-fix-restores-convex-contacts-at-upstream-head]] compared
contact counts across an in-process mode flip and passed green in 6.6 seconds. Rerun with the
fix force-disabled — a build in which the deterministic path measurably drops every contact —
it **still passed**, in 0.61 seconds, because the flip never reached the compiled kernels.[^2]

Two rules follow, and every deterministic-mode measurement on this wiki now observes both:

- **Set the mode before the first import that can compile.** Every valid repro script sets
  `wp.config.deterministic` at the top of the file, before `mujoco_warp` is imported. Scripts
  that happened to do this earlier were correct by habit, not by design; this page makes it
  design.
- **A deterministic-vs-default comparison needs two processes.** The committed regression test
  runs its deterministic half in a `subprocess` with the mode set before import, then compares
  the reported contact count against the in-process default-mode count. With the fix present
  it passes (4 == 4); with the fix disabled it fails (`0 != 4`) — the same negative control
  that exposed the first test as fake.[^3]

The corollary for anyone testing this feature interactively: a Jupyter session or REPL that
has already stepped a model **cannot** be switched into deterministic mode by assignment. The
process must be restarted. Nothing in Warp says so at the point of failure.

[^1]: `[observed]` — with the fix's staged path force-disabled in a build (`deterministic = False` in `convex_narrowphase`, `mujoco_warp/_src/collision_convex.py`, branch `deterministic-ccd` over HEAD `3879591`), an in-process flip to `RUN_TO_RUN` after one default-mode `mjw.collision()` produced identical contact counts (4) in 0.61s with no recompile; the same flip done before import produces 0 contacts on the same build. warp-lang 1.15.0.
[^2]: `[observed]` — the first draft of `mujoco_warp/_src/collision_driver_test.py::test_deterministic_convex_narrowphase` (branch `deterministic-ccd`; the in-process-flip version, since replaced by the subprocess probe at [^3]): "1 passed ... in 6.57s" with fix, "1 passed ... in 0.61s" with fix disabled. A test that cannot fail is not a test.
[^3]: `mujoco_warp/_src/collision_driver_test.py::test_deterministic_convex_narrowphase` (branch `deterministic-ccd`) — subprocess probe sets `wp.config.deterministic` before `import mujoco_warp`; observed failing `AssertionError: 0 != 4` against the fix-disabled build in 464.62s (cold deterministic codegen), passing in 8.52s with the fix.
