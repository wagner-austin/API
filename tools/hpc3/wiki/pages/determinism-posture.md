---
title: Determinism is declared, split between launcher and payload, and recorded
tags: [identity, determinism]
hubs: [images-and-staging]
related: ["[[known-answers]]", "[[run-documents]]"]
source_paths:
  - "README.md"
source_git_blobs:
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
provenance:
  - "measured: RTX 3090 Ti, torch 2.6.0+cu124, transformers 4.46.3"
  - "platform_core.determinism_env (libs/platform_core, outside this workspaceRoot)"
fact_checked: 2026-09-01
confidence: high
---

# Determinism is declared, split between launcher and payload, and recorded

`deterministic` is required on every project, and it is not a quality
setting — it **partitions results**. Measured on this exact stack (RTX 3090
Ti, torch 2.6.0+cu124, transformers 4.46.3): two same-seed runs of a 6-layer
model diverge at the sixth significant figure of the loss without the
controls, and the deterministic loss is a *different number* from the
nondeterministic one. Runs on either side form separate records, and
comparing across them measures the setting rather than the thing under test.

The posture travels with the run — into `--comment` as `det=on|off` and into
the ledger — so two arms that differ only in it can never be silently mixed.

## The split, because only one half is a submitter's to do

| half | who | why |
| --- | --- | --- |
| `CUBLAS_WORKSPACE_CONFIG` | **this tool**, in the batch script | cuBLAS reads it once when its handle is created; setting it after CUDA has started is accepted in silence and does nothing. Exported from the script it cannot be too late, and cannot be forgotten. |
| `torch.use_deterministic_algorithms(True)`, cuDNN and TF32 flags | **the payload** | they are torch calls in the payload's own process. This tool has no torch and does not pretend to make them. |

The payload reads `TRAIN_DETERMINISTIC` (`0` or `1`, always exported) and
applies its half. Model-Trainer's `setup_env` honours it: absent means on,
since determinism is the platform default and the local worker predates any
launcher; `0` declines it and logs `determinism declined`; anything else
raises rather than resolving to either posture. The name carries no cluster
prefix because a worker running locally should not be reading `HPC3_*`.

The record of what happened comes from the payload, not from here. This tool
*declares* a posture; only the process making the torch calls knows whether
they happened, and it logs the applied report either way — so "deterministic"
and "not" are never distinguished by a missing log line.

## Why the split is safe

PyTorch *enforces* the pairing: deterministic mode raises a `RuntimeError`
naming the missing variable, so a payload that does its half without the
launcher's half fails loudly rather than training quietly non-reproducible
numbers. The variable's name and value are defined once, in
`platform_core.determinism_env`, and imported by both the trainer and this
submitter — a duplicated literal would drift, nothing would fail, and the
runs would stop being comparable.
