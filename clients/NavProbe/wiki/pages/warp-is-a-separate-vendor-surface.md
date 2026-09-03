---
title: MuJoCo-Warp is a second vendor, not a second MJX
tags: [architecture, adapters, warp, typing]
related: ["[[mujoco-untyped-requires-protocol-boundary]]", "[[vmap-requires-every-leaf-batched]]"]
source_paths:
  - "src/navprobe/adapters/mjx_warp_bindings.py"
  - "src/navprobe/adapters/mujoco_bindings.py"
  - "src/navprobe/adapters/mjx_warp_render.py"
  - "tests/adapters/test_mjx_warp_bindings.py"
source_git_blobs:
  "src/navprobe/adapters/mjx_warp_bindings.py": "a2a8fdfa9ee65d6cac09fd9c15818d39111d9284"
  "src/navprobe/adapters/mujoco_bindings.py": "2665203e9f51764e84020b2e7fcca1d69a2ec5b1"
  "src/navprobe/adapters/mjx_warp_render.py": "898c44998cd0d22b65901c73453c940b081c50ac"
  "tests/adapters/test_mjx_warp_bindings.py": "b7bf8ede34c9e1e4e2d1b0b3350f8019ff517205"
provenance:
  - "mujoco-warp 3.11.0"
fact_checked: 2026-08-13
confidence: high
hubs: [simulator-adapters]
---

# MuJoCo-Warp is a second vendor, not a second MJX

Both drive MuJoCo physics and both batch across worlds, so the natural instinct is to extend the MJX bindings. That would have been wrong: the two APIs agree on almost nothing below the name.

| | MJX (on JAX) | MuJoCo-Warp |
|---|---|---|
| `step` | returns new `Data` | returns `None`, mutates in place |
| batching | a `vmap` axis over the pytree | `nworld` argument to `make_data` |
| state container | JAX pytree | Warp struct of device arrays |
| rendered output | not available | written into a `RenderContext` |
| host transfer | `Array.tolist()` | `array.numpy()` / `array.assign()` |

A Protocol loose enough to cover both would have declared a surface neither vendor has. So `mjx_warp_bindings.py` is a separate boundary with its own Protocols, and the two adapters share only what is genuinely identical.

## What they do share

Exactly one thing: compiling MJCF. `mujoco.MjModel.from_xml_string` is the same call with the same signature for both backends, so it lives once in `mujoco_bindings.py` and both binding modules import the compiled-model Protocol from there.[^1]

The claim is about what the adapters **import**, not about what `mujoco_bindings.py` contains, and the distinction became load-bearing on 2026-08-30: that module grew `MjDataProtocol`, `MjDataLoaderProtocol` and `ForwardProtocol` for the collision-pair work, so it now holds three members neither adapter touches. Both still import `MjModelProtocol` and nothing else, so "exactly one" is unchanged — but it is now a fact about two import lines rather than about the size of the shared module, and re-checking it means reading those lines rather than the module.[^4]

That extraction happened because the second adapter forced the question. Written independently, there would have been two declarations of `from_xml_string`, free to drift into disagreeing about a signature only one of them had checked — and the drift test for one would not have covered the other.

## Mutation makes the drift tests carry more weight

`step` and `render` both return `None` and communicate entirely by modifying their arguments. A call that silently did nothing is therefore indistinguishable from one that worked, unless the test asserts on state that changed.

So every Warp drift test asserts a difference rather than a return value: `step` is checked by `qpos` having moved, `render` by the depth buffer no longer being uniform, and `assign` by the values reading back.[^2] That last one matters most — an `assign` that silently failed would leave every world at the model's default, every world identical, and the batch carrying no information, **while still reporting perfect determinism**. It is the failure mode most likely to produce a confident wrong answer.

## The renderer inverts the usual data flow

`render(m, d, rc)` returns nothing and writes into the context, so the context is both an input and the place the observation is read from. That is why each simulator owns its own context rather than sharing one from the factory: two simulators sharing a context would overwrite each other's pixels between the render and the read.[^3]

[^1]: `src/navprobe/adapters/mujoco_bindings.py` — `MjModelProtocol`, `MjModelLoaderProtocol`, `MujocoModuleProtocol`, `load_mujoco`; imported by both `mjx_bindings.py` and `mjx_warp_bindings.py`.
[^2]: `tests/adapters/test_mjx_warp_bindings.py::TestDeclaredKeywordNames` — `test_step_mutates_the_state_in_place`, `test_render_writes_into_the_context`, `test_assign_writes_through_to_the_device`.
[^3]: `src/navprobe/adapters/mjx_warp_render.py` — `MjWarpRenderSimulator.__init__` creates the context per instance; the factory shares only the compiled and placed model.
[^4]: `src/navprobe/adapters/mjx_warp_bindings.py:25` and `src/navprobe/adapters/mjx_bindings.py:31` — both read `from navprobe.adapters.mujoco_bindings import MjModelProtocol`, one name each — `[observed]`. The three unimported members are at `src/navprobe/adapters/mujoco_bindings.py:34`, `:56` and `:71`; their only consumer is `tests/test_collision_pairs.py:45-46`.
