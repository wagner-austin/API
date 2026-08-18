---
title: vmap requires every pytree leaf to carry a batch axis
tags: [jax, mjx, batching, adapters]
related: [[mujoco-untyped-requires-protocol-boundary]], [[mjx-batched-step-reproduces-on-cpu]]
sources: [mujoco-mjx 3.11.0, jax 0.10.2, src/navprobe/adapters/mjx.py]
fact_checked: 2026-08-13
confidence: high
---

# vmap requires every pytree leaf to carry a batch axis

An MJX `Data` is a JAX pytree. Batching it by replacing one field with a batched array does not work: `vmap` maps over the leading axis of **every** leaf, and the leaves that were not replaced still have rank 0.

## The failure

Building a batched state as `base_data.replace(qpos=<(nworld, nq) array>)` and stepping it under `vmap(step, in_axes=(None, 0))` raises:

```
ValueError: vmap was requested to map its argument along axis 0, which implies
that its rank should be at least 1, but is only 0 (its shape is ())
```

`qpos` gained its batch axis; `time`, `qvel`, and the rest did not.[^1]

The error is worth reading carefully, because it names a rank rather than a field: the diagnostic points at the transform, not at the leaf that was missed, so the natural first reading is that `in_axes` is wrong.

## What works instead

Vectorise the **construction**, not the field assignment. `vmap` broadcasts values a mapped function closes over, so mapping a builder that closes over `base_data` gives every untouched leaf its batch axis:

```python
def build_one(qpos: FlatArrayProtocol) -> MjxDataProtocol:
    return base_data.replace(qpos=qpos)

build_batched_state = state_transforms.vmap(build_one)
```

The builder takes one world's positions and returns one world's state; the vmapped form takes an `(nworld, nq)` array and returns a fully batched state.[^2]

## Why this shapes the adapter's types

The unbatched and batched states are genuinely different shapes, so they get different Protocols — `MjxDataProtocol.qpos` returns a flat array whose `tolist()` gives `list[float]`, while `BatchedMjxDataProtocol.qpos` gives `list[list[float]]`. The vmapped builder is what converts between them, and its type signature says so.[^3]

mypy enforced this before the runtime did: an early draft declared `replace` as taking a *batched* array, and the type checker rejected the flat call the builder actually makes. The Protocols encode the distinction that `vmap` enforces at runtime.

## Consequence for the observation

Because the batch axis is leading, `qpos.tolist()` on the batched state returns one row per world in world order, and the adapter flattens world-major.[^4] That order is part of the instrument's contract: a simulator that reordered its worlds between runs would register as non-determinism, which is the signal being measured rather than noise to normalise away.

[^1]: `[observed]` — driving `MjxSimulator.advance` against a state built by `base_data.replace(qpos=...)` on `mujoco-mjx` 3.11.0 / `jax` 0.10.2 raised the quoted `ValueError` from `jax/_src/api.py` `_get_axis_size`.
[^2]: `src/navprobe/adapters/mjx.py` L193-204 — `build_one` and `state_transforms.vmap(build_one)`.
[^3]: `src/navprobe/adapters/mjx_bindings.py` L35-56 `FlatArrayProtocol` / `BatchedArrayProtocol`, and L223-250 `StateBuilderProtocol` / `BatchedStateBuilderProtocol`.
[^4]: `tests/adapters/test_mjx.py::TestObservation::test_flattens_every_world_into_one_observation` — asserts the observation length is `world_count * nq`.
