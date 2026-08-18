---
title: MuJoCo ships no type marker, so its surface is declared as Protocols
tags: [typing, mjx, adapters, drift]
related: ["[[vmap-requires-every-leaf-batched]]"]
source_paths:
  - "src/navprobe/adapters/mjx_bindings.py"
  - "src/navprobe/adapters/mujoco_bindings.py"
  - "src/navprobe/adapters/jax_bindings.py"
source_git_blobs:
  "src/navprobe/adapters/mjx_bindings.py": "7c30f87ec36dafc29bb627b33a91fe62578f853b"
  "src/navprobe/adapters/mujoco_bindings.py": "4c271831d631dd97a50a9a6129fd7d1abf367aa2"
  "src/navprobe/adapters/jax_bindings.py": "0e04487c1a15f649726a2320f5cc1d56c51e7985"
provenance:
  - "mujoco 3.11.0"
  - "jax 0.10.2"
fact_checked: 2026-08-13
confidence: high
hubs: [simulator-adapters]
---

# MuJoCo ships no type marker, so its surface is declared as Protocols

`mujoco` 3.11.0 has no `py.typed` marker; `jax` and `jaxlib` 0.10.2 do.[^1] Under this package's mypy settings — `strict` plus `disallow_any_unimported` and `disallow_any_expr` — importing `mujoco` directly pulls an untyped module into a package that forbids `Any` anywhere.

## The pattern

Declare the surface as Protocols and bind the module by **assigning it to the Protocol**. The annotation supplies the type; the import supplies the object.

```python
def load_mujoco() -> MujocoModuleProtocol:
    module: MujocoModuleProtocol = __import__(MUJOCO_MODULE, fromlist=["MjModel"])
    return module
```

This is the monorepo's established boundary pattern, not a local invention — TankpitBot binds Pillow the same way.[^2]

The Protocols declare only what the adapter calls: three MJX functions, two JAX transforms, one `jax.numpy` function, and the fields actually read.[^3] A Protocol that reproduced the rest of MJX would be a second, unverified copy of somebody else's API.

## `jax.vmap` needs two Protocols, not one

`jax.vmap` is genuinely polymorphic, and this package calls it two ways: over the step function with `in_axes`, and over state construction with no `in_axes`. A single Protocol loose enough to accept both would check neither, so the module is loaded behind two views, each declaring one use precisely.[^4]

## Keeping the declaration honest

A Protocol is a claim about someone else's code, and left unchecked it is a claim that *was* true. The failure mode is silent: the adapter keeps type-checking against a signature the vendor has changed, and the first symptom is a wrong measurement rather than a red build.

The drift test therefore **calls** every declared function by keyword using the Protocol's own parameter names, and drives each to a result only the real function could produce.[^5] All seven declared keyword names are accepted by the installed packages: `from_xml_string(xml=)`, `put_model(m=)`, `make_data(m=)`, `step(m=, d=)`, `asarray(a=)`, `vmap(in_axes=)`, `replace(qpos=)`.[^6]

Calling beats reading here. It checks presence, spelling, and behaviour in one assertion, and it is the only form available: the monorepo bans `inspect`, and reflecting over `__annotations__` would reintroduce `Any` at exactly the boundary these Protocols exist to keep typed.

One vendor claim is checked rather than assumed: that `jit` does not change the result. The adapter measures the compiled path, so if compiling altered the numbers, every verdict would describe a different computation from the one the caller believes was measured.[^7]

[^1]: `src/navprobe/adapters/mujoco_bindings.py:13` (the boundary rationale recording that mujoco ships no py.typed marker) — `[observed]` — `ls <venv>/Lib/site-packages/{mujoco,jax,jaxlib}/py.typed`: absent for `mujoco`, present for `jax` and `jaxlib`. Also absent at `mujoco/mjx/py.typed`.
[^2]: `clients/TankpitBot/src/tankpit_bot/_pillow.py` L77 — `image_module: PillowImageModuleProtocol = __import__("PIL.Image", fromlist=["open", "new"])`.
[^3]: `src/navprobe/adapters/mjx_bindings.py` L150-161, `MjxModuleProtocol` — declares exactly `put_model`, `make_data`, `step`.
[^4]: `src/navprobe/adapters/mjx_bindings.py` L194-208 `StepVmapProtocol` and L211-225 `StateVmapProtocol`; loaded at L289 `load_jax_step_transforms` and L299 `load_jax_state_transforms`.
[^5]: `tests/adapters/test_mjx_bindings.py::TestDeclaredKeywordNames` — one test per declared function, each a keyword call driven to an asserted result.
[^6]: The seven keyword names are declared across three binding modules, not one: `src/navprobe/adapters/mujoco_bindings.py:37` (`from_xml_string(xml=)`), `src/navprobe/adapters/mjx_bindings.py:150-161` (`MjxModuleProtocol`, whose `put_model`, `make_data` and `step` attributes carry the keyword surfaces), `src/navprobe/adapters/mjx_bindings.py:197` (`vmap(in_axes=)`), `src/navprobe/adapters/mjx_bindings.py:76` (`replace(qpos=)`) and `src/navprobe/adapters/jax_bindings.py:57` (`asarray(a=)`) — `[observed]` — each keyword form invoked against `mujoco` 3.11.0 / `mujoco-mjx` 3.11.0 / `jax` 0.10.2; all returned without `TypeError`.
[^7]: `tests/adapters/test_mjx_bindings.py::TestJitPreservesResults::test_compiled_and_eager_batched_steps_agree`.
