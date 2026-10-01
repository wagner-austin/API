---
title: Both Warp deterministic modes give the same digests on Ampere and Turing
tags: [warp, determinism, measurement, finding, gpu, gpu-to-gpu, run-to-run, architecture, portability]
related: ["[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[coupled-body-threshold-moves-on-turing]]", "[[tactile-alias-is-inert-with-live-taxels]]", "[[a-determinism-verdict-needs-a-correctness-oracle]]", "[[warp-binds-determinism-mode-at-first-compile]]", "[[open-questions-and-what-would-answer-them]]"]
source_paths:
  - "scripts/gpu_deterministic_sweep.py"
  - "scripts/apply_tactile_alias_patch.py"
  - "src/navprobe/codecs/sweep_run.py"
  - "src/navprobe/experiment.py"
source_git_blobs:
  "scripts/gpu_deterministic_sweep.py": "b11833275121df70433903d3753e70adc00ff35e"
  "scripts/apply_tactile_alias_patch.py": "a5c27614e81318fe95f5160e5049651dc313ebde"
  "src/navprobe/codecs/sweep_run.py": "c7bdaf2c72d7cd014d977f97f27b9da1e5704a20"
  "src/navprobe/experiment.py": "c9b57616cfed8af956f6f1956fdd6ecec1eeff92"
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
  - "board task a9a700c3 (2026-10-01 sweep, opus-navprobe-w1-1001)"
  - "records on austinpc, untracked: clients/NavProbe/runs/a9a700c3-2026-10-01/ (3090- and 1630-RUN_TO_RUN, 3090- and 1630-GPU_TO_GPU .txt; decode-comparison.txt; cache-and-hash-log.txt)"
fact_checked: 2026-10-01
confidence: high
measured_with:
  package: mujoco-warp 3.11.0 with the tactile alias patch applied, reverted after
  warp: 1.16.0 (CUDA Toolkit 12.9, driver CUDA 13.1)
  modes: [RUN_TO_RUN, GPU_TO_GPU]
  backend: warp cuda:0 on each host
  devices:
    - NVIDIA GeForce RTX 3090 Ti (sm_86, 24 GiB, driver 591.86, PCIe x16 in PCI_E1, host austinpc, i7-13700K)
    - NVIDIA GeForce GTX 1630 (sm_75 Turing, 4 GiB, driver 591.86, PCIe gen3 x16, host lavender, i7-11700K)
  os: Windows 11 10.0.26200 on both hosts
  harness: navprobe.sweep.run_scene_sweep over navprobe.scenes.row_scene (the ten-scene family)
  adapter: navprobe.adapters.mjx_warp_state
  seed: 7
  step_count: 150
  repetitions: 12
  world_count: 2
  perturbation: 0.01
  constraint_capacity: 8192
  max_records: 64
  linesearch_block_dim: 32 (pinned explicitly on both devices)
  kernel_cache: fresh directory per device per mode
hubs: [determinism-measurement]
---

# Both Warp deterministic modes give the same digests on Ampere and Turing

`GPU_TO_GPU` claims that the same program gives the same bits on different devices. It
had never been tested here, because the instrument had only ever run on one CUDA
architecture. With the alias patch applied
([[tactile-alias-patch-clears-warp-deterministic-compile]]), the ten-scene sweep was run
under it on the sm_86 RTX 3090 Ti and the sm_75 GTX 1630. **All ten reference digests are
identical across the two cards.** The coupled 5, 6, 8 and 32-body rows included.[^1][^2]

**`RUN_TO_RUN` does the same thing, which it does not promise.** Its ten digests also
match across the two cards, ten for ten.[^1][^2] On this scene family and this driver,
reproducibility under either mode is portable between architectures, not merely per
device.

## The digests, decoded from the records

Every scene reproduced on both cards in both modes. Prefixes of the reference digest from
each decoded `navprobe-sweep-run/2` record:[^2]

| scene | `RUN_TO_RUN`, both cards | `GPU_TO_GPU`, both cards |
|---|---|---|
| separated 2 | `24e4b98b3cd5fdda` | `24e4b98b3cd5fdda` |
| separated 8 | `e2cc60ade06391fa` | `e2cc60ade06391fa` |
| separated 16 | `c9002ed3f667f1ae` | `c9002ed3f667f1ae` |
| separated 32 | `7bb8689c962bb13a` | `7bb8689c962bb13a` |
| touching 2 | `3849c2aa47a799cf` | `3849c2aa47a799cf` |
| touching 4 | `d276b2f515cfa6ef` | `08b7c66574321268` |
| touching 5 | `fa81afe0a180746a` | `704100ce89a67b3d` |
| touching 6 | `d59366edf41d4acf` | `c93594f7cc2dc301` |
| touching 8 | `2ab58e00b21fb1ce` | `25a10a891749be76` |
| touching 32 | `ab082525f26ca3dc` | `10330a64599ae571` |

## The two modes are portable, and they disagree with each other

On every coupled scene, the `RUN_TO_RUN` digest differs from the `GPU_TO_GPU` digest, on
both cards. Each mode fixes *an* order and holds it across devices. It is not the same
order. On the separated rows and the 2-body row they agree, as do the default mode's
digests ([[coupled-body-threshold-moves-on-turing]]): with nothing to reorder, all three
compute one answer.[^2]

One coincidence worth recording: `GPU_TO_GPU`'s touching-4 digest, `08b7c665...`, is
the digest the default mode reproduces for that scene on both cards. `RUN_TO_RUN`'s
digest differs. So `GPU_TO_GPU` happens to choose the same reduction order as the
default kernels on that scene. It does not on the 5-body and larger rows, where the
default has no stable order to share.[^2]

Consequence: **a digest is comparable only between runs of the same mode.** A record that
does not state its mode cannot be compared with another. The sweep record already carries
`mode` in its header, which is why the comparison above could be decoded rather than
assumed.[^2]

## Limits

- **Two cards in two hosts.** The 1630 sits in `lavender`, not beside the 3090 Ti. A
  match across hosts is the *stronger* statement, because it also spans CPU and chipset.
  A mismatch would have been ambiguous; this result is not.
- **One driver.** Both cards ran 591.86. Portability across driver branches is untested.
- **One sweep per mode per card.** Each verdict is twelve repetitions agreeing. The
  cross-card match is a single comparison per scene, and it is bit-exact on all twenty.
- **Primitive geometry only.** Spheres route to MJWarp's primitive narrowphase. Under
  deterministic mode the convex narrowphase drops contacts
  ([[deterministic-mode-drops-contacts-in-convex-narrowphase]]), so a portable digest
  there could be a portable wrong answer.
- **`max_records` 64**, the value [[open-questions-and-what-would-answer-them]] question
  1c asks to check against 4096. Portability holds at 64; whether 64 truncates anything is
  still that question's to answer.
- **Venvs restored.** The patch was applied before the deterministic runs and reverted
  after, on both hosts. `sensor.py` returned to sha256 `921230eb...` each time, the same
  canonical hash both venvs started from.[^3]

[^1]: [observed] `python -m scripts.gpu_deterministic_sweep RUN_TO_RUN <fresh-cache> 64 --device cuda:0 --linesearch-block-dim 32`, then the same with `GPU_TO_GPU`, on each host, 2026-10-01, after `python -m scripts.apply_tactile_alias_patch apply`. The Warp init banner names the card: `"NVIDIA GeForce RTX 3090 Ti" (24 GiB, sm_86` and `"NVIDIA GeForce GTX 1630" (4 GiB, sm_75`. Conditions are the module constants at `scripts/gpu_deterministic_sweep.py` L54-67.
[^2]: `src/navprobe/codecs/sweep_run.py` L63 (`decode_sweep_run`) decoded all four records; the comparison was keyed on the decoded `scene` and read `trial.reference_digest` and `trial.deterministic`, which is all-repetitions-match (`src/navprobe/experiment.py` L178-179).
[^3]: [observed] `sha256sum` / `certutil -hashfile ... SHA256` of `mujoco_warp/_src/sensor.py` in each venv: `921230eb341cb3056cded2c2f0b3c3afd42cbe9cd76ff2cae82d1c7e6b4be779` before apply and after revert, `e51dc959a8e0e9f29f0878fca47968d0fd517c051f3039308985f3c3e3befddd` while patched, on both hosts. The script is `scripts/apply_tactile_alias_patch.py` L1-28.
