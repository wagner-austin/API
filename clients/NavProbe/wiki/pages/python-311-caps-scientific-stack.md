---
title: Python 3.11 caps jax, numpy and scipy below their latest releases
tags: [platform, dependencies, constraint]
related: [[jax-cuda-unavailable-on-windows]]
sources: [PyPI, pyproject.toml]
fact_checked: 2026-08-13
confidence: high
---

# Python 3.11 caps jax, numpy and scipy below their latest releases

`poetry show --outdated` reports four scientific dependencies behind their newest published versions. All four caps are the interpreter, not a stale lock: the newer releases require Python 3.12 and this package pins `python = "^3.11"`.[^1]

| package | installed | latest published | why not installed |
|---|---|---|---|
| jax | 0.10.2 | 0.11.0 | 0.11.0 requires Python >= 3.12 |
| jaxlib | 0.10.2 | 0.11.0 | no matching distribution for this platform |
| numpy | 2.4.6 | 2.5.2 | 2.5.x requires Python >= 3.12 |
| scipy | 1.17.1 | 1.18.0 | 1.18.0 requires Python >= 3.12 |

The installed versions are therefore the **latest reachable** on this interpreter, and `poetry update` correctly reports nothing to do.[^2] A reader running `poetry show --outdated` will see four rows and should not read them as neglect.

## The toolchain pins are a monorepo decision, not a package one

`mypy` and `ruff` do have newer releases installable on 3.11 — mypy 2.3.0 against the pinned `^1.19`, ruff 0.16.3 against `^0.14.4`. Both pins match TankpitBot, RustedWarfareBot, and `monorepo_guards` exactly.[^3]

Bumping them in this package alone would fork the linting and type-checking toolchain across four repos that share one guard rule set, which is the drift these pins exist to prevent. A mypy major-version bump in particular changes which programs type-check, so a single-package bump means one repo enforcing a different standard while claiming the same one. The upgrade is a monorepo-wide change or it is not made.

## Consequence

Raising the floor to 3.12 would unlock the newer scientific stack, and it would have to be done across the workspace for the same reason as above. Nothing currently measured depends on it, and it would not lift the constraint in [[jax-cuda-unavailable-on-windows]] — that one is a platform gap, not a version gap.

[^1]: `[observed]` — `pip install --dry-run` for each version reported `Requires-Python >=3.12` for `jax==0.11.0`, `numpy==2.5.2` and `scipy==1.18.0`, and `No matching distribution found` for `jaxlib==0.11.0`. `pyproject.toml` `[tool.poetry.dependencies]` pins `python = "^3.11"`.
[^2]: `[observed]` — `poetry update jax jaxlib numpy scipy` reported `No dependencies to install or update`.
[^3]: `clients/TankpitBot/pyproject.toml` L71-72, `clients/RustedWarfareBot/pyproject.toml` L30-31, `libs/monorepo_guards/pyproject.toml` L24-25 — `ruff = "^0.14.4"` in all three; mypy `^1.19`, `^1.19`, `^1.13.0`.
