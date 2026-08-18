# NavProbe

**A reproducibility instrument for simulated navigation.** It answers one
question about a simulator before any policy result computed on it can be
trusted: given a fixed seed and a fixed action sequence, does the same rollout
produce the same bytes?

That question is open. GPUSimBench (IROS 2026) measured GPU-batched simulators
and found MJX and Genesis fully reproducible at `0.00 ± 0.00` on both
run-to-run and inter-environment variability — with **rendering disabled**, a
limitation the paper states and names perception and sensor rendering as
uncovered. MJX-Warp separately ships a hardware-accelerated batch renderer
producing RGB and depth across parallel environments, built on a per-step
bounding-volume-hierarchy update and a raycaster. That is a different numerical
path from the contact solver that was measured, and nothing published joins the
two.

## What it is

An instrument, not a policy and not a simulator. It drives someone else's
simulator, digests every step, and produces a verdict.

```
                        scenes ─────────────┐
                     (a scene is a value)   │
                                            v
canonical -> digest -> records -> wireformat -> rollout -> comparison -> experiment -> sweep
   bytes     stable     shapes      codecs       runs       verdicts       trials     families
                                       │                                      │
                                    storage                              dispersion
                                       │                              (how far apart,
                                  crossprocess                        not just whether)
                                       │
                                   adapters
                        (every vendor lives here, and only here)
```

* `canonical` turns floats and text into canonical little-endian bytes. NaN is
  rejected, because a value that compares unequal to itself cannot participate
  in an equality verdict. Every payload is length-prefixed, so no two different
  shapes flatten to the same bytes.
* `digest` folds canonical bytes into BLAKE2b digests with domain separation,
  so a step digest can never equal a run digest built from the same bytes.
* `records` declares every typed record — run spec, step, run, comparison, trial
  spec, trial, scene, dispersion, sweep entry — as `TypedDict`s whose collection
  fields are tuples.
* `wireformat` carries one `encode_*`/`decode_*` pair per record, with every
  decoded field passing a `require_*` check. Decoding never guesses: a record
  whose header and body disagree is refused, and each record type has its own
  banner so a comparison cannot be read as a run.
* `rollout` drives an injected `SimulatorProtocol` to a run record.
* `comparison` folds two run records into a verdict: whether they agree, and
  where they first stopped agreeing.
* `storage` moves run and trial records across a process boundary, which is what
  makes the fresh-process condition measurable at all.
* `experiment` composes the rest into a trial: `ProbeService` takes a simulator
  *factory*, builds a freshly constructed simulator per repetition, and reports
  whether every repetition matched the first. The factory is the injection point
  — a trial against MJX and a trial against an in-repo simulator differ only by
  what is passed in.
* `crossprocess` compares trials that did not share a process. One process
  records a trial to disk, another loads the recordings and reaches a verdict.
  This is the only comparison in the package whose two sides can come from
  different machines, backends, or library versions — and it is why `storage`
  exists.
* `scenes` turns a `SceneSpec` — body count, lattice width, spacing, radius,
  timestep — into MJCF. A scene is a *value*, so a published result cites five
  numbers rather than a string literal buried in whatever script produced it,
  and rebuilding the scene a finding was measured on is reading those numbers
  off a wiki page. Whether bodies touch each other is derived from the geometry,
  never stored, so it cannot disagree with the scene it describes.
* `dispersion` measures how far apart repeated rollouts *end up*, in the
  observation's own units. A digest comparison fails at the first differing bit,
  which is what makes it a good leading indicator and a poor answer to "would
  anyone notice". This is the other half.
* `sweep` runs one trial design across a family of scenes and reports where the
  verdict changes. The vendor arrives as a *builder* rather than a factory,
  because a sweep constructs a different factory per scene — that is the
  injection point which keeps the layer vendor-agnostic.
* `adapters` is the only layer that imports a simulator, and there are three.
  `adapters/mjx.py` drives MJX on JAX: MJCF compiled, model placed, one
  `jit`-compiled `vmap`-batched kernel shared across repetitions.
  `adapters/mjx_warp_state.py` and `adapters/mjx_warp_render.py` drive
  MuJoCo-Warp's solver and its batch renderer respectively — same vendor, same
  bindings, differing only in what counts as an observation. Keeping them apart
  is what lets a rendered rollout's failure be attributed to the raycaster or to
  the solver rather than to "the simulator".

  MJX and Warp are separate vendors, not variants: Warp's `step` mutates in
  place, `nworld` is an allocation parameter rather than a `vmap` axis, and
  there is no pytree. So each has its own bindings, with the genuinely shared
  parts factored out — compiling MJCF into `adapters/mujoco_bindings.py`, JAX's
  arrays and NumPy surface into `adapters/jax_bindings.py`. None of the vendors
  ships `py.typed`, so every symbol used is declared as a Protocol and each
  module is bound by assignment to it.

## What it has measured

> **Read this first.** Warp 1.16 ships an explicit determinism control —
> `warp.config.deterministic`, with `RUN_TO_RUN` and `GPU_TO_GPU` levels
> implemented by intercepting atomic calls at codegen. **MuJoCo-Warp 3.11.0
> cannot compile under either of them**: `_sensor_tactile` mixes `max` and `add`
> reductions on one array, and the failure is unconditional — a model with zero
> sensors still hits it, because Warp compiles the whole sensor module. So every
> result below characterises `NOT_GUARANTEED`, which is the default and, for
> MJWarp users today, the only mode available. That is arguably the most
> actionable thing here: a defect with a file, a line and a stated reason.


### The GPU is not reproducible once bodies touch each other

The MuJoCo-Warp documentation says GPU results "may differ between executions of
the same code" and advises the CPU device for deterministic results. It does not
say when. The answer is not "when there are many contacts":

| scene | contacts | GPU | CPU |
|---|---:|---|---|
| 32 spheres in a row, **not touching** (floor contacts only) | 64 | reproduces | reproduces |
| 6 spheres in a row, **mutually touching** | 14 | **12 distinct results from 12 runs** | reproduces |

Four times the contacts reproduce exactly; a handful of *coupled* ones do not. A
body resting on the floor constrains itself against the world, and thirty-two of
those share nothing; two bodies resting against each other write into both, and
once a chain of them shares degrees of freedom the accumulation order matters.

It is not last-bit trivia either. At 32 touching bodies the spread across
identical runs grows from 4.5 × 10⁻⁸ m to roughly the size of the container in
two simulated seconds, while the CPU stays at **exactly zero** — ordinary chaotic
amplification of a difference that a GPU reduction introduced. The final states
are physically sensible, not blown up: spheres at rest, inside the walls, all
finite.

Practically: flat-terrain locomotion will not see this; manipulation, clutter and
granular piles will. An earlier version of this finding blamed *stacking* and was
wrong — a flat single layer whose spheres touch laterally fails too, and that is
what separated "stacked" from "coupled".

### Across backends, on the physics

MJX 3.11.0, falling sphere on a plane, seed 7, 200 steps. CPU on Windows
(jax 0.10.2) and WSL2 Linux (jax 0.11.0); CUDA on an RTX 3090 Ti under WSL2.

**Within a backend, MJX is bit-reproducible.** Five repetitions agreed exactly at
every batch width — 1 to 64 on CPU, 1 to 4096 on CUDA — with a distinct
reference digest per width. Three independent GPU *processes* also agreed,
compared through recordings on disk. That much is the expected result: it
reproduces GPUSimBench's Type 2 finding on the path GPUSimBench measured, and an
instrument whose positive control failed would be measuring itself.

**Across backends, it is not.** Comparing recordings made in different
environments, at repetition zero:

| left | right | match | first divergence | what differs |
|---|---|:--|--:|---|
| Windows CPU, jax 0.10.2 | WSL CPU, jax 0.11.0 | **true** | — | OS and library version |
| WSL CUDA | WSL CUDA, fresh process | **true** | — | process only |
| WSL CPU | WSL CUDA | false | **57** | backend only |
| Windows CPU | WSL CUDA | false | **57** | everything |

CPU and CUDA emit **identical bytes for 57 steps**, then part. Not immediate
divergence — accumulate-then-diverge, which is the case a coarse check misses: a
50-step test passes, a 60-step test fails, nothing having changed.

**Step 57 is when the ball lands.** MuJoCo-Warp — a different compiler and a
different execution model — parts from its own CPU counterpart at the same step,
and the height trace shows the free-fall increment halving there as the contact
impulse first acts. Everything before it is a fixed-order integration that every
backend reproduces exactly; contact is an iterative solve that reduces across
constraints, and floating-point addition is not associative. So the divergence
point is a property of *the trajectory*, not of the simulator — and a
contact-free benchmark will certify a portability that evaporates on first touch.

### The rendered stream

MJX-Warp's batch renderer is a raycaster over a per-step bounding-volume
hierarchy — a different numerical path again. Pin the device and it is exactly
reproducible, at every batch width, in both channels. Change the device, from
**bit-identical inputs** (`qpos`, `geom_xpos`, `geom_xmat` all equal):

| quantity | cuda vs cpu |
|---|---|
| `rgb_data` | identical |
| `depth_data` | **2,794 of 8,192 pixels differ** (34.1%) |
| max absolute difference | 1.47 × 10⁻⁵ (123 float32 ULPs) |

Colour compares equal because it is quantised to 8 bits per channel and the
discrepancy is far below one quantisation step. So a team validating a rendered
pipeline by diffing RGB frames would conclude the renderer is device-portable.
It is not — and depth is the channel depth-based navigation policies consume.
This one is present from the very first frame, thirty-eight steps before the
solver's divergence, and does not accumulate.

So "deterministic" holds *within* a backend and is not portable *across* one, for
two independent reasons with two different onsets. A golden digest captured on
CPU cannot be compared against a GPU run.

Full conditions, method, and the limits of each claim are in `wiki/` — start at
`wiki/hubs/determinism-measurement.md`.

### Running it elsewhere

`jax-cuda12-plugin` publishes no Windows distribution, so the CUDA measurements
were taken under WSL2. The instrument needed no change: `PYTHONPATH` pointed at
the same source tree through `/mnt/c`, because the adapter is the only layer
that touches a vendor and nothing above it knows which backend produced the
numbers.

`SimulatorProtocol` is this package's own port. A driven adapter converts a
concrete simulator to it, which is where a vendor signature gets matched
exactly and where arrays are flattened to floats. Nothing above that adapter
imports a simulator, which is why the whole instrument is exercised without a
GPU.

## Testing discipline

The suite drives **real simulators, not mocks** — `tests/simulators.py` carries
a genuinely deterministic one, a genuinely divergent one, and the degenerate
cases. A determinism instrument validated against a mock would only establish
that the mock is deterministic.

The production storage hooks are exercised against a real filesystem in
addition to the in-memory fakes, so the implementations that ship are the
implementations that ran.

The MJX adapter is tested against **real MJX** — compiled, placed, traced,
batched. Its Protocols are drift-tested by *calling* every declared vendor
function with the Protocol's own keyword parameter names and driving it to a
result, so a renamed vendor parameter fails the suite rather than surfacing
later as a wrong measurement.

Scenes are compiled by the **real MuJoCo compiler** in their tests rather than
only string-matched: a scene that reads correctly and does not compile is still
not a scene. And the Warp state adapter is tested on the condition the finding
says is *safe* — a row whose bodies touch only the floor — because that one holds
on CPU and GPU alike and is therefore assertable unconditionally.

The fresh-process claim is tested with an **actual fresh process**: the suite
spawns a real child interpreter that records a trial, then compares against it
through the files alone. Every other test shares an interpreter with the code
under test, which is exactly the condition that layer exists to escape — so
asserting it in-process would have asserted nothing.

## Reproducing a measurement

Every result above was taken through the package API, not a script. A scene is a
value, so reproducing one is naming it:

```python
from navprobe.adapters.mjx_warp_state import MjWarpStateSimulatorFactory
from navprobe.records import TrialSpec
from navprobe.scenes import row_scene
from navprobe.sweep import first_irreproducible, run_scene_sweep

def build(model_xml: str, world_count: int):
    return MjWarpStateSimulatorFactory(
        model_xml=model_xml, world_count=world_count,
        perturbation=0.01, constraint_capacity=8192,
    )

touching = tuple(row_scene(n, 0.055, 0.03, 0.005) for n in (2, 4, 5, 6, 8, 16, 32))
entries = run_scene_sweep(build, touching, TrialSpec(seed=7, step_count=150, repetitions=12), 2)
print(first_irreproducible(entries)["scene"]["body_count"])
```

Swap `0.055` for `0.070` and the same sweep reproduces at every size — that one
character is the whole finding. `navprobe.dispersion.measure_dispersion` answers
the other half, how far apart the runs ended up, and
`navprobe.codecs.sweep.encode_sweep` writes the result to a diff-able file whose
floats are stored exactly.

A caveat worth stating: the boundary is **not** a universal constant. The same
scene family measured with identical worlds puts it at 6 bodies and with
per-world perturbation at 5. What survives both is the shape, not the number.

## Build

```bash
make check          # lint + test (the gate)
make lint           # guard + ruff + mypy over src, tests, scripts
make test           # pytest + coverage, 100% statements and branches
make install        # poetry install
```

Strictness matches the rest of the monorepo: mypy strict with
`disallow_any_expr`, no `Any`, no casts, no `type: ignore`, no `.pyi`, and the
shared `monorepo_guards` rule set over `src`, `tests`, and `scripts`.

`monorepo-guards` is a develop path dependency, so `scripts/guard.py` imports
it directly. The other clients reach it through `sys.path` insertion plus
`__import__`, which predates that dependency and is no longer load-bearing.
