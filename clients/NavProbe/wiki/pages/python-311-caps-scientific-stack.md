---
title: Python 3.11 caps jax, numpy and scipy below their latest releases
tags: [platform, dependencies, constraint]
related: ["[[jax-cuda-unavailable-on-windows]]"]
source_paths:
  - "pyproject.toml"
source_git_blobs:
  "pyproject.toml": "da2061d9199e5a8df0b5477e03c201cd88b96863"
provenance:
  - "PyPI"
fact_checked: 2026-09-03
confidence: high
hubs: [platform-constraints]
---

# Python 3.11 caps jax, numpy and scipy below their latest releases

`poetry show --outdated` reports four scientific dependencies behind their newest published versions. All four caps are the interpreter, not a stale lock: the newer releases require Python 3.12 and this package pins `python = "^3.11"`.[^1]

| package | installed | latest published | why not installed |
|---|---|---|---|
| jax | 0.10.2 | 0.11.0 | 0.11.0 requires Python >= 3.12 |
| jaxlib | 0.10.2 | 0.11.0 | no matching distribution for this platform |
| numpy | 2.4.6 | 2.5.2 | 2.5.x requires Python >= 3.12 |
| scipy | 1.17.1 | 1.18.0 | 1.18.0 requires Python >= 3.12 |

The installed versions are therefore the **latest reachable** on this interpreter, and `poetry update` correctly reports nothing to do.[^2] A reader running `poetry show --outdated` will see four rows, and they are the interpreter constraint at work, not neglect.

## The toolchain pins are a monorepo decision, not a package one

Bumping the linting and type-checking pins in this package alone would fork the toolchain across repos that share one guard rule set, which is the drift these pins exist to prevent. A mypy major-version bump in particular changes which programs type-check, so a single-package bump means one repo enforcing a different standard while claiming the same one. The upgrade is a monorepo-wide change or it is not made.

**And it was made, that way.** On 2026-09-03 commit 47d3c06f took one toolchain across all 42 packages: `ruff = "^0.16.6"` and `mypy = "^2.3.1"`, identical in this package, TankpitBot, RustedWarfareBot and `monorepo_guards`.[^3] The versions this page had named as available-but-deliberately-not-taken are, to within a patch release, exactly the ones adopted — so the section now records a prediction that held rather than a standing decision to hold back.

Until that commit the pins were **not** uniform, and this page said they were: mypy was `^1.19` here and in the two clients but `^1.13.0` in `monorepo_guards`, so "both pins match ... exactly" was true of ruff and false of mypy at the time of writing. The claim is true now for the first time.

## Consequence

Raising the floor to 3.12 would unlock the newer scientific stack, and it would have to be done across the workspace for the same reason as above. Nothing currently measured depends on it, and it would not lift the constraint in [[jax-cuda-unavailable-on-windows]] — that one is a platform gap, not a version gap.

[^1]: `[observed]` — `pip install --dry-run` for each version reported `Requires-Python >=3.12` for `jax==0.11.0`, `numpy==2.5.2` and `scipy==1.18.0`, and `No matching distribution found` for `jaxlib==0.11.0`. `pyproject.toml` `[tool.poetry.dependencies]` pins `python = "^3.11"`.
[^2]: `pyproject.toml:20` `python = "^3.11"` — `[observed]` — `poetry update jax jaxlib numpy scipy` reported `No dependencies to install or update`. Cited as line 21 until 2026-09-03; the pin is on line 20 in the blob this page was pinned to as well, so that anchor was never right.
[^3]: `pyproject.toml:30` `ruff = "^0.16.6"` and `:31` `mypy = "^2.3.1"`, against `clients/TankpitBot/pyproject.toml:80-81`, `clients/RustedWarfareBot/pyproject.toml:31-32` and `libs/monorepo_guards/pyproject.toml:23-24` (which lists mypy before ruff) — all four identical, set by commit 47d3c06f. The earlier version of this footnote recorded the pre-unification state: `ruff = "^0.14.4"` in all three and mypy `^1.19`, `^1.19`, `^1.13.0`, at line numbers that have since moved.
