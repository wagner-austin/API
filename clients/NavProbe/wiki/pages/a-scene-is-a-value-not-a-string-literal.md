---
title: A scene is a value, so a result can cite it instead of describing it
tags: [architecture, reproducibility, scenes]
related: ["[[warp-gpu-determinism-fails-on-coupled-bodies]]", "[[bit-equality-is-a-leading-indicator]]"]
source_paths:
  - "src/navprobe/scenes.py"
  - "src/navprobe/codecs/scene.py"
  - "src/navprobe/wireformat.py"
  - "tests/test_scenes.py"
  - "src/navprobe/sweep.py"
  - "tests/codecs/test_scene.py"
source_git_blobs:
  "src/navprobe/scenes.py": "4a05c692fbd2740bd717f015e7725fa8175fc207"
  "src/navprobe/codecs/scene.py": "f0ea50af3585bab267f52832eb73b75bb3f7514b"
  "src/navprobe/wireformat.py": "3937d4a49ee9e32db7e614afbb89b97d1861a7c8"
  "tests/test_scenes.py": "ebeb6878130073a665c2414a61c88d38bcce52eb"
  "src/navprobe/sweep.py": "3fbecbbf2caeec07827f618c307ded4699414dfd"
  "tests/codecs/test_scene.py": "cdbd6f14f617e5904d203f79b447a1b78b8828de"
fact_checked: 2026-08-14
confidence: high
hubs: [instrument-design]
---

# A scene is a value, so a result can cite it instead of describing it

Every determinism finding in this wiki is a claim about a scene. For a while every one of them was measured by a standalone script with the MJCF written inline, which meant the geometry that decided the result lived in a string literal in a temporary file.

That is not a reproducibility problem in the abstract. It caused a concrete one, described below.

## The shape of the fix

`SceneSpec` carries five numbers — body count, lattice width, spacing, radius, timestep — and :mod:`navprobe.scenes` turns one into MJCF.[^1] A result cites the value. Rebuilding the scene a finding was measured on is reading five numbers off a page rather than recovering a script.

Two properties are **derived, never stored**:

- `bodies_touch` compares spacing against one diameter. It is the variable the central finding turns on, and a stored flag could disagree with the geometry it claims to describe.
- `layer_count` divides body count by the lattice's grid. Whether a scene stacks follows from its shape rather than from an assertion about it.

Deriving both is what makes a scene self-describing: there is no combination of fields that can be internally inconsistent.

## What it caught

The scene family was re-measured through the package's own sweep after the fix, and the boundary moved by one body — 6 under the original script, 5 through the package.[^2]

The cause was mundane and invisible from the outside: the package's adapter perturbs each world's initial position from the seed, and the standalone script did not. Same scene family, same trial design, different initial conditions, different boundary.

Two things follow. First, the finding needed restating: the boundary is a property of a configuration, not of MuJoCo-Warp, and [[warp-gpu-determinism-fails-on-coupled-bodies]] now reports both harnesses and claims no universal number. Second, and more usefully — that discrepancy was only findable *because* both harnesses could be pointed at the same scene definition. While the scene lived in a string literal, the two measurements were not comparable and the difference would have gone unnoticed.

## Floats are stored exactly

The scene codec encodes floats through `float.hex` rather than `str`.[^3] The round trip through a decimal repr happens to be exact in CPython, but "exact because of a repr implementation" is a weaker guarantee than this package makes anywhere else, and the value decides which scene a published measurement refers to.

`0.055` is not exactly representable in binary. Any codec that goes through a rounded decimal string fails that round trip, and the test asserts it directly rather than trusting the format.

## Validation happens at build, not at construction

`require_scene` runs inside `build_scene` rather than at the point a spec is written.[^4] A spec arrives either hand-written or decoded from a file, and only one of those routes could be trusted; checking at the single point where a scene becomes MJCF covers both.

Without it, an invalid spec fails somewhere inside MuJoCo's compiler with a message about XML.

[^1]: `src/navprobe/scenes.py` — `SceneSpec` is declared in `navprobe.records`; `build_scene`, `bodies_touch`, `layer_count` and `row_scene` live here.
[^2]: src/navprobe/sweep.py:51 `run_scene_sweep` — `[observed]` — the same row family measured by a standalone script (identical worlds) reproduced at 5 bodies and failed at 6; measured by `navprobe.sweep.run_scene_sweep` through `navprobe.adapters.mjx_warp_state` with `perturbation=0.01`, it reproduced at 4 and failed from 5.
[^3]: `src/navprobe/wireformat.py` — `encode_float_field` / `require_positive_float_field`, and the shared `require_hexadecimal_float` beneath them; `tests/test_wireformat.py::TestEncodeFloatField::test_round_trips_a_value_with_no_exact_decimal_form` covers the encoding, and `tests/codecs/test_scene.py::TestSceneFloatFields::test_round_trips_a_spec_whose_floats_have_no_exact_decimal_form` covers a *scene* reaching it. Both helpers lived in `src/navprobe/codecs/scene.py` until 2026-08-19; they were lifted into `wireformat` when a third copy of the same hexadecimal check turned up in `codecs/observation.py` and a fourth in `codecs/dispersion.py`, all four sharing error code `NP-WIRE-014` for different checks.
[^4]: `src/navprobe/scenes.py` — `build_scene` calls `require_scene` before emitting anything; `tests/test_scenes.py::TestBuildScene::test_validates_before_building`.
