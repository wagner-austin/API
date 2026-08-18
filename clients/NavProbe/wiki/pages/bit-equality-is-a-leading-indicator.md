---
title: Bit-equality is a leading indicator, and needs a magnitude beside it
tags: [architecture, determinism, measurement-design]
related: ["[[gpu-nondeterminism-amplifies-to-macroscopic-scale]]", "[[a-scene-is-a-value-not-a-string-literal]]"]
source_paths:
  - "src/navprobe/dispersion.py"
  - "src/navprobe/canonical.py"
  - "tests/test_dispersion.py"
source_git_blobs:
  "src/navprobe/dispersion.py": "3c2f9b092175813dc16798b15be58968ac1a334f"
  "src/navprobe/canonical.py": "e766888d77755cf4e32080e97de2b16cb45184ba"
  "tests/test_dispersion.py": "257a715abf3d960df87827ad98c662cac62615c7"
fact_checked: 2026-08-14
confidence: high
hubs: [instrument-design]
---

# Bit-equality is a leading indicator, and needs a magnitude beside it

A digest comparison fails at the first differing bit. That is its strength — it catches a divergence long before anyone could see it — and its limit: "twelve runs produced twelve digests" says nothing about whether the twelve outcomes are a nanometre apart or half a metre.

Both questions matter and they have different answers at the same configuration. Six mutually-touching bodies already fail bit-equality while their positions stay within 10⁻⁸ m for the whole rollout; thirty-two reach the scale of the container ([[gpu-nondeterminism-amplifies-to-macroscopic-scale]]). A report of only the first sounds alarming and might not be; a report of only the second misses the failure entirely at the smaller size.

## The instrument does not name the units

`measure_dispersion` reports the element-wise spread of final observations in whatever units the observation is in.[^1] For a state observation those are metres; for a rendered one they are depth units or packed colour.

The layer has never known which, and that is deliberate: it drives a `SimulatorProtocol`, and the protocol's contract is "a sequence of floats in a stable order". Naming the units would mean the dispersion layer knowing which adapter produced its input, which is exactly the coupling the port exists to prevent. The caller knows what it asked for.

## Why NaN had to be rejected here specifically

The digest path rejects NaN because a NaN digest is meaningless. The numerical path needed the same rule for a sharper reason, and the first version of this module got it wrong.

NaN propagates silently through `max` and `min`, so an unguarded measurement returns a NaN spread — and a NaN spread compares false against *every* threshold. A caller asking "is the spread below tolerance?" is told **yes**.[^2] That is the worst available failure: not a wrong number, but a wrong number that reads as a pass.

The check is `require_encodable`, and it shares its implementation and its error code with the encoder's own NaN rule rather than restating it.[^3] One definition of what this package considers admissible, two callers.

## A test asserted the defect before it caught it

The first test written for this path asserted that a NaN observation *produces* a NaN spread — it checked `max_spread != max_spread` and passed. The docstring above it claimed the opposite: "NaN fails the measurement rather than dispersing to NaN."

The test and its own description disagreed, and the test won. It is the same failure mode as [[passing-test-can-miss-its-own-premise]]: a test that encodes the right intent in prose and pins the wrong behaviour in code. Worth noting that the prose was the correct half both times.

## What a fresh simulator per repetition is for

Each repetition is built from the factory rather than reusing one instance, matching the trial layer.[^4] Reusing an instance would measure whether `reset` restores state — a weaker question that produces a similar-looking number, and one nobody asked.

[^1]: `src/navprobe/dispersion.py` — `measure_dispersion` returns a `DispersionRecord` carrying `max_spread` and `mean_spread` over the observation's elements.
[^2]: `tests/test_dispersion.py::TestMeasureDispersion::test_rejects_an_observation_containing_nan` — a NaN observation now raises `NP-CANON-001` rather than returning a spread.
[^3]: `src/navprobe/canonical.py` — `require_encodable` and `encode_float` both delegate to `_require_not_nan`, so there is one NaN policy and one error code.
[^4]: `tests/test_dispersion.py::TestMeasureDispersion::test_builds_a_fresh_simulator_per_repetition` — asserts the factory was called once per repetition.
