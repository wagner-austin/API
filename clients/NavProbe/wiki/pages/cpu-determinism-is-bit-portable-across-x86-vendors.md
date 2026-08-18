---
title: The CPU control is bit-identical across x86 vendors and vector widths, not merely self-consistent on each
tags: [cpu, determinism, measurement, finding, control, portability, warp, avx]
related: ["[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[coupled-body-threshold-does-not-move-with-sm-count]]", "[[cpu-determinism-survives-os-and-version-change]]", "[[measurement-fleet-is-reachable-by-ssh-alias]]", "[[mujoco-requires-avx-so-pre-avx-hosts-are-ineligible]]"]
source_paths:
  - "src/navprobe/sweep.py"
source_git_blobs:
  "src/navprobe/sweep.py": "3fbecbbf2caeec07827f618c307ded4699414dfd"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
fact_checked: 2026-08-18
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cpu
  hosts:
    - lavender — Intel64 Family 6 Model 167 Stepping 1, GenuineIntel (i7-11700K, Rocket Lake); AVX2 + AVX512F
    - pendragon — AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD (Ryzen Z1 Extreme, Zen 4); AVX2 + AVX512F
    - sedona — Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (i7-12700H, Alder Lake); AVX2, **no AVX512F**
    - emerald — AMD64 Family 21 Model 48 Stepping 1, AuthenticAMD (A10-7800, Steamroller); AVX only, **no AVX2**
  harness: navprobe.sweep.run_scene_sweep over navprobe.scenes.row_scene
  adapter: navprobe.adapters.mjx_warp_state
  seed: 7
  step_count: 150
  repetitions: 12
  world_count: 2
  perturbation: 0.01
  constraint_capacity: 8192
  scenes: 10 (separated 2/8/16/32 at spacing 0.070; touching 2/4/5/6/8/32 at spacing 0.055)
hubs: [determinism-measurement]
---

# The CPU control is bit-identical across x86 vendors and vector widths, not merely self-consistent on each

Every GPU non-determinism figure on this wiki is stated against a CPU control that reproduces
exactly. That control carries more weight than any single measurement, because it is what
licenses the word *non*-determinism — without it, a disagreeing GPU could just be a
disagreeing simulator. It had only ever been exercised on one vendor's silicon at one vector
width, which makes it a narrower claim than it reads as.

Run on four processors spanning two vendors and **every vector width this software can
execute on**, the control holds and every digest matches.[^1][^2][^4][^5]

## Four processors, ten scenes, zero differences

| | Intel i7-11700K | AMD Ryzen Z1 Extreme | Intel i7-12700H | AMD A10-7800 |
|---|---|---|---|---|
| family | Family 6 Model 167 (Rocket Lake) | Family 25 Model 116 (Zen 4) | Family 6 Model 154 (Alder Lake) | Family 21 Model 48 (Steamroller) |
| vendor string | `GenuineIntel` | `AuthenticAMD` | `GenuineIntel` | `AuthenticAMD` |
| AVX | present | present | present | present |
| AVX2 | present | present | present | **absent** |
| AVX-512F | present | present | **absent** | **absent** |
| all ten scenes reproduce | yes | yes | yes | yes |
| **run digests** | **identical across all four columns, 10/10** | | | |

Every host reported `all_deterministic: true`, and mechanical comparison of the emitted
reports found **10 matching digests and 0 mismatches** in each pairing, scene for
scene.[^3][^4][^5]

## The width axis is closed, not merely sampled

Three widths are represented — AVX-512, AVX2-without-AVX-512, and AVX-without-AVX2 — and
that is not a convenience sample. It is the complete set: MuJoCo's shipped binary requires
AVX and will not import below it, so **no processor exists that could extend this axis
further** ([[mujoco-requires-avx-so-pre-avx-hosts-are-ineligible]]). The span measured here
is the span that can run the software at all.

Two hosts carry the weight. The Alder Lake part has AVX-512 fused off by Intel on consumer
12th-generation silicon, and the Steamroller part predates AVX2 by design — a 2014
microarchitecture executing the same workload through a two-generation-older vector unit
than the newest host here.[^4][^5] Neither moved a single bit.

This includes the touching family at 5, 6, 8 and 32 bodies — the exact configurations where
the GPU produces twelve different answers from twelve runs of the same rollout
([[warp-gpu-determinism-fails-on-coupled-bodies]]). The contrast is therefore measured on
the same scenes, same seed and same harness, not inferred across experiments.

## Why bit-identity is the interesting part

Self-consistency on each vendor would have been the weaker, expected result: it would say
only that each machine agrees with itself. Two failure modes remain open under that reading —
the vendors could each be stably wrong in different directions, and any comparison of a
result taken on one machine against a result taken on another would need a caveat.

Bit-identity closes both. Floating-point results are not automatically portable across
microarchitectures: FMA contraction, vector width, and library implementations of
transcendental functions are all places where two conforming implementations may legitimately
differ in the last bit, and a chaotic contact simulation amplifies exactly that
([[gpu-nondeterminism-amplifies-to-macroscopic-scale]]). None of it moved these digests.

The practical consequence is that a CPU reference digest is a **portable artefact**. A
rollout recorded on one machine can be compared against one recorded on another without
qualification, which is what makes the fleet usable as a fleet rather than as a set of
machines that can only be compared to themselves.

## What this does not establish

All four hosts ran the same OS family, the same Python 3.11, and the same package versions,
installed the same way. This is deliberate — holding them fixed is what isolates the
processor — but it means the result says nothing about compiler or libm differences, which
[[cpu-determinism-survives-os-and-version-change]] covers on a different axis and a
different backend.

The scene family is one family. Ten scenes spanning separated and mutually-contacting
configurations is a reasonable spread, and it deliberately includes the body counts where
the GPU fails, but it is not every workload MuJoCo can run.

Nothing here is claimed beyond x86-64. ARM is untested and is not reachable from any node
in the fleet — and unlike the vector-width axis, that gap is a matter of hardware access
rather than a closed question.

[^1]: `[observed]` — on host `pendragon`, `PYTHONPATH=C:\navprobe\src C:\navprobe\.venv\Scripts\python.exe C:\navprobe\cpu_control_sweep.py`. Reported `processor: AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD`, `device: cpu`, features `{SSE3, SSE4_1, SSE4_2, AVX, AVX2, AVX512F}` all true, and `all_deterministic: true` over all ten scenes with `first_divergent_step: null` throughout.
[^2]: src/navprobe/sweep.py:51 `run_scene_sweep` — `[observed]` — the identical command and script on host `lavender`. Reported `processor: Intel64 Family 6 Model 167 Stepping 1, GenuineIntel`, same feature set, same `all_deterministic: true`.
[^3]: `[observed]` — the emitted `REPORT` JSON documents parsed and compared pairwise by `(bodies, spacing)`: `MATCHING DIGESTS: 10/10   MISMATCHES: 0`. Separated family `f581f7d13e9f97a9`, `28da8310f3fae984`, `7ab7b42709d5e13c`, `7209271298759b49`; touching family `e834fb8482a2d99c`, `bcd8bd4a0cca6234`, `267590080dc41147`, `29588b5842782589`, `dbc9204ac108f3ab`, `f39840062e7615e4`.
[^4]: src/navprobe/sweep.py:51 `run_scene_sweep` — `[observed]` — the identical script on host `sedona`, reached over SSH from austinpc. Reported `processor: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`, features `{SSE3, SSE4_1, SSE4_2, AVX, AVX2}` true and **`AVX512F: false`**, `all_deterministic: true`, and all ten digests equal to those in [^3]. The host's `Win32_Processor.Name` is `12th Gen Intel(R) Core(TM) i7-12700H`.
[^5]: src/navprobe/sweep.py:51 `run_scene_sweep` — `[observed]` — the identical script on host `emerald`, reached over SSH from austinpc. Reported `processor: AMD64 Family 21 Model 48 Stepping 1, AuthenticAMD`, features `{SSE3, SSE4_1, SSE4_2, AVX}` true with **`AVX2: false`** and **`AVX512F: false`**, `all_deterministic: true`, and all ten digests equal to those in [^3]. The host's `Win32_Processor.Name` is `AMD A10-7800 Radeon R7, 12 Compute Cores 4C+8G`. This host also demonstrates the AVX floor from the other side: `import mujoco` succeeds here, where it fails on a processor lacking AVX entirely.
