---
title: JAX has no CUDA backend on native Windows
tags: [platform, jax, cuda, constraint]
related: [[mjx-batched-step-reproduces-on-cpu]], [[python-311-caps-scientific-stack]]
sources: [jax 0.10.2, PyPI]
fact_checked: 2026-08-13
confidence: high
---

# JAX has no CUDA backend on native Windows

`jax-cuda12-plugin` publishes no distribution installable on Windows.[^1] `jax` and `jaxlib` install cleanly and expose only `[CpuDevice(id=0)]`.[^2] The machine has a CUDA-capable GPU — an RTX 3090 Ti with 24 GB, driver 591.86 — so the constraint is the wheel, not the hardware.[^3]

## What this blocks, and what it does not

It blocks **JAX** on the GPU from native Windows. It does not block GPU work generally, and it does not block the measurements — both of which turned out to be true in ways worth stating precisely, because the obvious reading of the missing wheel is wider than the fact.

**Warp reaches the same GPU natively.** `warp-lang` 1.16.0 publishes a `win_amd64` wheel and ships its own CUDA runtime, so on native Windows it enumerates both `cpu` and `cuda:0` on the RTX 3090 Ti with no WSL2 involved.[^4] Every MuJoCo-Warp measurement in this wiki — including the renderer findings — was taken that way, and the Warp adapter's tests exercise the real GPU inside `make check`.

So the constraint is narrow: it is a JAX packaging fact, not a platform capability fact. MJX's own import reports `Failed to import warp: No module named 'warp'` when Warp is absent, which is the absence of a package rather than a platform rejection.[^5]

## The route around it, taken and confirmed

WSL2. JAX's CUDA wheels target Linux, and WSL2 exposes the host GPU: the same RTX 3090 Ti is visible from inside the Ubuntu distribution, and `jax` 0.11.0 there reports `[CudaDevice(id=0)]` with `default_backend() == "gpu"`.[^6]

Two things were predicted here before being tested, and both held:

- **The instrument needed no change.** It ran unmodified, with `PYTHONPATH` pointed at the same Windows source tree through `/mnt/c`. The adapter is the only layer that touches a vendor, and nothing above it knows which backend produced the numbers.
- **The CUDA measurements became reachable.** [[mjx-cuda-batched-step-reproduces]] and [[mjx-determinism-does-not-cross-backends]] were both taken this way.

One environment detail is worth recording because it is not obvious: the distribution had no `python3.12-venv` and `sudo` requires a password, so `python3 -m venv` fails and PEP 668 blocks `pip install --user`. The venv was created with `uv`, installed by its own no-sudo installer into `~/.local`. That is a local bootstrap for a measurement environment only — the package itself remains a poetry project, and switching it would fork the build toolchain across the four repos that share one guard rule set.

## Why this is recorded rather than worked around

A platform gap that shapes where measurements can run is a project fact. Left unrecorded it gets rediscovered — the natural next step after a clean CPU sweep is to try the GPU, and without this page that attempt starts by installing packages and reading errors.

It also bounds what the CPU results were ever allowed to claim. Presenting the CPU sweep as evidence about GPU batching would have been exactly the unearned transfer this project was started to object to — and, as it turned out, wrong: the two backends do not agree.

[^1]: `[observed]` — `pip index versions jax-cuda12-plugin` in the package venv returned `ERROR: No matching distribution found for jax-cuda12-plugin`.
[^2]: `[observed]` — `python -c "import jax; print(jax.devices())"` printed `[CpuDevice(id=0)]` with `jax` 0.10.2 / `jaxlib` 0.10.2 `cp311-win_amd64`.
[^3]: `[observed]` — `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` returned `NVIDIA GeForce RTX 3090 Ti, 591.86, 24564 MiB`.
[^4]: `[observed]` — on native Windows, `warp.init()` prints `Devices: "cpu" : "Intel64 ..."` and `"cuda:0" : "NVIDIA GeForce RTX 3090 Ti" (24 GiB, sm_86, mempool enabled)` with `CUDA Toolkit 12.9, Driver 13.1`; `warp.get_devices()` returns `['cpu', 'cuda:0']`. Both `warp-lang` 1.16.0 and `mujoco-warp` 3.11.0 install from PyPI into the Windows venv.
[^5]: `[observed]` — importing `mujoco.mjx` 3.11.0 emits `Failed to import warp: No module named 'warp'` and `Failed to import mujoco_warp: No module named 'warp'` to stderr, then proceeds on the JAX backend.
[^6]: `[observed]` — inside WSL2 Ubuntu: `nvidia-smi` reports the same RTX 3090 Ti, and `python -c "import jax; print(jax.__version__, jax.devices(), jax.default_backend())"` prints `0.11.0 [CudaDevice(id=0)] gpu`.
