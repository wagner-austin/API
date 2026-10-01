---
title: The coupled-body determinism threshold moves on Turing, and only by one scene
tags: [warp, determinism, measurement, finding, gpu, codegen, architecture, sm_75]
related: ["[[coupled-body-threshold-does-not-move-with-sm-count]]", "[[coupled-body-threshold-turns-on-one-kernels-block-size]]", "[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[warp-deterministic-modes-are-digest-portable-across-architectures]]", "[[open-questions-and-what-would-answer-them]]", "[[measurement-fleet-is-reachable-by-ssh-alias]]"]
source_paths:
  - "scripts/gpu_deterministic_sweep.py"
  - "src/navprobe/sweep.py"
  - "src/navprobe/experiment.py"
  - "src/navprobe/codecs/sweep_run.py"
source_git_blobs:
  "scripts/gpu_deterministic_sweep.py": "b11833275121df70433903d3753e70adc00ff35e"
  "src/navprobe/sweep.py": "3fbecbbf2caeec07827f618c307ded4699414dfd"
  "src/navprobe/experiment.py": "c9b57616cfed8af956f6f1956fdd6ecec1eeff92"
  "src/navprobe/codecs/sweep_run.py": "c7bdaf2c72d7cd014d977f97f27b9da1e5704a20"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
  - "board task a9a700c3 (2026-10-01 sweep, opus-navprobe-w1-1001)"
  - "records on austinpc, untracked: clients/NavProbe/runs/a9a700c3-2026-10-01/ (3090-ng, 1630-ng, 3090-ng-rate, 1630-ng-rate .txt; decode-comparison.txt; cache-and-hash-log.txt)"
fact_checked: 2026-10-01
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0 (CUDA Toolkit 12.9, driver CUDA 13.1)
  mode: NOT_GUARANTEED
  backend: warp cuda:0 on each host
  devices:
    - NVIDIA GeForce RTX 3090 Ti (sm_86, 24 GiB, driver 591.86, PCIe x16 in PCI_E1, host austinpc, i7-13700K)
    - NVIDIA GeForce GTX 1630 (sm_75 Turing, 4 GiB, driver 591.86, PCIe gen3 x16, host lavender, i7-11700K)
  os: Windows 11 10.0.26200 on both hosts
  harness: navprobe.sweep.run_scene_sweep over navprobe.scenes.row_scene
  adapter: navprobe.adapters.mjx_warp_state
  seed: 7
  step_count: 150
  repetitions: 12
  world_count: 2
  perturbation: 0.01
  constraint_capacity: 8192
  max_records: 64
  linesearch_block_dim: 32 (pinned explicitly on both devices)
  kernel_cache: fresh directory per device per run
  independent_trials_per_cell: 20 (touching rows of 4, 5 and 6 bodies)
hubs: [determinism-measurement]
---

# The coupled-body determinism threshold moves on Turing, and only by one scene

[[coupled-body-threshold-does-not-move-with-sm-count]] held the 5-body boundary fixed
across two Ampere devices and left open whether a different *architecture* would move it.
On the sm_75 GTX 1630 it does, partly. The 5-body touching row, which never reproduces on
the sm_86 3090 Ti, reproduces in **16 of 20** independent trials on Turing. The 6-body
row fails on both cards every time.[^1][^2]

So the boundary has a codegen component. It is not purely algorithmic, and the
published "five bodies" is an Ampere figure. The move is one scene wide: from 6 bodies
up the failure holds on both architectures.

## The rate, which is the comparison

The block size is pinned to the same value on both cards, so the block-size effect
cannot account for the move
([[coupled-body-threshold-turns-on-one-kernels-block-size]]). Driver 591.86 and the OS
build are identical too. Each trial is a separate `run_scene_sweep` call with a newly
built factory.[^2]

| touching row | 3090 Ti, sm_86 | GTX 1630, sm_75 |
|---|---|---|
| 4 bodies | 20 / 20 | 20 / 20 |
| **5 bodies** | **0 / 20** | **16 / 20** |
| 6 bodies | 0 / 20 | 0 / 20 |

The 3090 Ti column is a fresh control, not a quote. It re-measures the published 0/20
under the exact pin used on the 1630.

## The single sweep, decoded

One full ten-scene sweep per card, each emitting a `navprobe-sweep-run/2` record. Both
records were decoded with `decode_sweep_run` and compared entry by entry.[^1][^3]

| scene | 3090 Ti | GTX 1630 | reference digests equal |
|---|---|---|---|
| separated 2 / 8 / 16 / 32 | all reproduce | all reproduce | yes, all four |
| touching 2, 4 | reproduce | reproduce | yes, both |
| touching 5 | fails, first divergence step 0 | **reproduces** | no |
| touching 6 | fails, step 1 | fails, step 1 | no |
| touching 8 | fails, step 0 | fails, step 1 | no |
| touching 32 | fails, step 0 | fails, step 0 | no |

Wherever both cards reproduce, the two architectures compute **the same digest**. In
the default mode the disagreement is in accumulation order only: no scene that
reproduces on both cards reproduces to two different answers.

## What it does and does not say

- **The threshold read off a single sweep is the wrong unit here, as on Ampere.** At
  80% a single sweep usually reports "threshold 6" on Turing, and sometimes 5. Quote the
  rate, not a count.
- **Not the trajectory.** On the 1630, 19 of 20 trials of the 5-body row started from
  the same reference digest, `23e1ccc1...`, and the other started from `81de9385...`. The
  3090 Ti started from `71f581c9...` on all 20. This is the default mode, so neither
  value is a "correct" answer. [[warp-deterministic-modes-are-digest-portable-across-architectures]]
  covers what the two cards compute when ordering is fixed.
- **Not why.** Turing and Ampere differ in SM count, in warp scheduling and in the SASS
  emitted for the same PTX, and this measurement does not separate those. It does rule out
  SM count alone, because two Ampere devices at 46 and 84 SMs agreed.
- **Hosts differ.** The cards are in two machines, and the operator has not yet moved the
  1630 into `austinpc`. CPU, chipset and RAM therefore differ alongside the architecture.
  Driver and OS do not. The GPU computes the step and the host only compiles the model,
  and that compile is bit-portable across x86 parts
  ([[cpu-determinism-is-bit-portable-across-x86-vendors]]). So the host is not a
  plausible cause, but it is a named difference, not a controlled one.
- **n = 20 per cell.** That separates 0% from 80%. It does not pin 80% to better than
  roughly ±18 points.

[^1]: [observed] `python -m scripts.gpu_deterministic_sweep NOT_GUARANTEED <fresh-cache> 64 --device cuda:0 --linesearch-block-dim 32`, run 2026-10-01 on each host from a venv built from the same `poetry.lock`; Warp's init banner reads `"cuda:0" : "NVIDIA GeForce RTX 3090 Ti" (24 GiB, sm_86` on austinpc and `"cuda:0" : "NVIDIA GeForce GTX 1630" (4 GiB, sm_75` on lavender. Conditions are the module constants at `scripts/gpu_deterministic_sweep.py` L54-67.
[^2]: [observed] a loop calling `navprobe.sweep.run_scene_sweep` 20 times per scene with `row_scene(n, 0.055, 0.03, 0.005)` for n in 4, 5, 6, building each factory exactly as `scripts/gpu_deterministic_sweep.py` L182-198 does (same `PERTURBATION`, `CONSTRAINT_CAPACITY`, `TRIAL`, `WORLD_COUNT` imported from that module, `linesearch_block_dim` 32, `max_records` 64), fresh kernel cache; it emitted all 60 trials as one `navprobe-sweep-run/2` record, decoded to recount the rates above. The loop lived in the session scratchpad, unversioned; the rows are reproduced here in full.
[^3]: `src/navprobe/codecs/sweep_run.py` L63 (`decode_sweep_run`) is the decoder used; `deterministic` means every repetition's digest matched the first (`src/navprobe/experiment.py` L178-179).
