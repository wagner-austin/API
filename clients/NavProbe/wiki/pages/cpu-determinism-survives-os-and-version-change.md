---
title: On CPU, a rollout survives a change of OS and of jax version
tags: [mjx, determinism, measurement, portability]
related: ["[[mjx-determinism-does-not-cross-backends]]", "[[mjx-batched-step-reproduces-on-cpu]]"]
source_paths:
  - "src/navprobe/crossprocess.py"
source_git_blobs:
  "src/navprobe/crossprocess.py": "2d5bfcb44bd8f37f3f2814edccb76f3b444775ee"
provenance:
  - "mujoco-mjx 3.11.0"
  - "jax 0.10.2"
  - "jax 0.11.0"
fact_checked: 2026-08-13
confidence: high
measured_with:
  package: mujoco-mjx 3.11.0 (both sides)
  jax: 0.10.2 (Windows) vs 0.11.0 (WSL2 Linux)
  python: 3.11.9 (Windows) vs 3.12.3 (WSL2 Linux)
  backend: cpu (both sides)
  model: falling sphere with a free joint on a plane, timestep 0.005
  seed: 7
  step_count: 200
  repetitions: 3
  world_count: 4
  perturbation: 0.05
hubs: [determinism-measurement]
---

# On CPU, a rollout survives a change of OS and of jax version

Two recordings — one from native Windows on Python 3.11.9 with `jax` 0.10.2, one from WSL2 Ubuntu on Python 3.12.3 with `jax` 0.11.0 — are **bit-identical** over 200 steps.[^1]

```
reference_digest = 6b8e692c1a0c6e60ac74882d08f5639aaa2befb99e2e44c65a51bee9b9134fd3
```

Same value from both. Different operating system, different Python minor version, different JAX minor version, same bytes.

## Why this is worth stating

It is the control that makes [[mjx-determinism-does-not-cross-backends]] interpretable. Without it, the CPU-versus-CUDA divergence has three candidate explanations — the backend, the library version, or the operating system — and no way to choose between them.

Holding this row fixed eliminates two of the three. The CPU/CUDA comparison was then run with OS and jax version held constant on one machine, leaving the backend as the only variable.

It also bounds how much the earlier CPU measurement can be trusted to travel: quite far, as it turns out, but along axes that do not include the backend.

## What this suggests about where the variance lives

Two minor releases of JAX and two operating systems produced identical bytes, which points at the numerical path being pinned by MuJoCo's own solver and by IEEE-754 semantics rather than by anything the surrounding stack chooses. The backend changes that, and evidently a version bump does not.

That is an inference from one model and one seed, not a general claim about JAX's release policy. It is the kind of thing worth re-checking on a major release rather than assuming.

## Method note

Neither side was compared by reading numbers off a terminal. Each environment wrote a recording; a third process loaded both and reached the verdict.[^2] That matters here specifically because the two runs could not have been in one process — they were on different operating systems — so file exchange is the only comparison available, and eyeballing truncated digest prefixes would not have been evidence.


## What this does not establish

Two minor JAX releases and two operating systems, on one model, one seed and one batch width. It says nothing about a *major* release, and nothing about any other CPU microarchitecture — both machines ran the same physical CPU, so "different OS" here does not mean "different hardware".

It also covers only the contact-sparse scene. A configuration above the coupled-body threshold is not bit-reproducible even against itself on GPU, and whether CPU portability survives there is unmeasured.

[^1]: `[observed]` — `record_trial` run in each environment writing to `win-cpu` and `wsl-cpu`; `compare_recordings(win-cpu, wsl-cpu, 0)` reported `digests_match=true`, `first_divergent_step=none`, `compared_step_count=200`. Both trial summaries carry the full 64-character digest quoted above.
[^2]: `src/navprobe/crossprocess.py` — `record_trial` persists every repetition and the summary; `compare_recordings` loads two recordings and compares a repetition. No simulator is constructed on the comparison side.
