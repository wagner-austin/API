---
title: The sweep's noise floor is a range, so replicating harder reports a weaker result
tags: [ml, model-trainer, cartridges, measurement, statistics]
related:
  - "[[model-trainer-composition-ceiling]]"
  - "[[model-trainer-companioned-training-recipe]]"
  - "[[model-trainer-cartridge-question-set]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/core/contracts/replicated_measurement.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_plans.py
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/core/contracts/replicated_measurement.py": 2c6af8b1ad7bdae264d03ae689a8d76dcc89f0eb
  "services/Model-Trainer/src/model_trainer/cli/cartridge_benchmark.py": 18c3ae406e820dd322a48f6fb2174e63cfa5e2c2
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_plans.py": 14a8f4c663b59b0f266fcb22b58820dadfff82cb
provenance:
  - "measured 2026-09-06 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1, --controls none"
  - "plan gpt2-wiki-9seed, label gpt2-wiki-9seed-gpt2-w256-s4-e12-lr0.01-slots2.8.32.128.512-c128-seeds7.8.9.10.11.12.13.14.15-e2f23c635583"
  - "corpus e2f23c635583 = the 12 me-wiki pages carrying visibility: public; second corpus legal-wiki, both archived byte-for-byte"
  - "reproduction: the 7/8/9 subset of the nine-seed run matches the three-seed record to six decimals (slots-8 +0.822426 vs +0.822426228296; slots-32 +0.880853 vs +0.880852517628)"
  - "board task 1fc5afed-89a7-400e-b79e-378f322711c7 carries the full trail; per-seed observations added in commit 0633816e"
fact_checked: "2026-09-06"
confidence: high
hubs: [services]
---

# The sweep's noise floor is a range

`cartridge_benchmark` judges each 4x step of its slot sweep against a noise
floor, and on 2026-09-04 that floor said the top two steps did not separate.
The conclusion recorded at the time — that gpt2's gain is resolved only across
2..32 — was wrong. All four steps are real, and the floor is why they did not
look it.

## The floor is a range, and a range grows with the sample

```python
spread      = max(gains) - min(gains)        # per arm, across seeds
noise_floor = max(spread for arm in sweep)   # across arms
```

Both halves are literal: `spread` is set at
`replicated_measurement.py` L156, and `noise_floor` returns the maximum of
those spreads at L185.

The expected range of a normal sample scales with its size as `sigma*d2(n)`:
`d2(3) = 1.69` against `d2(9) = 2.97`. Adding seeds inflates the floor by
about 75% before any effect of the thing being measured, and taking a **max**
over five arms compounds it — the noisiest arm sets the bar for all of them.

Measured, same corpus and same code, only the seed count differing:

| seeds | `sweep_noise_floor` |
|---|---|
| 3 | 0.0202 |
| 9 | 0.0546 |

A 2.7x inflation. Against the nine-seed floor the run rejects **three of four**
steps, including `8 -> 32`, which the three-seed plan had *accepted* at +0.0584
against 0.0202. The identical measurement, replicated three times harder,
reports a weaker result. That is the estimator moving, not the phenomenon.

## It also discards the pairing, which is the evidence

Every arm trains under the same seeds, so seed 7's 32-slot gain and seed 7's
128-slot gain are two measurements of **one draw**. The steps are paired
differences. The floor compares two independent spreads instead, and a range
cannot be un-pooled afterwards — it carries no information about which
replicate produced which end of it.

Paired, nine seeds:

| step | mean diff | SEM | t | seeds positive |
|---|---|---|---|---|
| 2 -> 8 | +0.0921 | 0.0042 | 21.69 | 9/9 |
| 8 -> 32 | +0.0526 | 0.0041 | 12.73 | 9/9 |
| 32 -> 128 | +0.0276 | 0.0022 | 12.31 | 9/9 |
| 128 -> 512 | +0.0268 | 0.0055 | 4.83 | 8/9 |

Sign test as the distribution-free backup, since `t` at n=9 leans on
normality: 9/9 is p=0.002 one-sided, 8/9 is p=0.02. Both readings agree.

`32 -> 128` has the **smallest SEM in the table** at 0.0022 against a floor of
0.0546 — a factor of 25. The most precisely measured step of the four is the
one the floor called noise.

## What is not wrong with `noise_floor`

It is not being replaced, and it is not a defect. Within one plan every arm
shares `n`, so the maximum range is a fair estimate of how much one arm of that
kind wobbles, which is what its docstring claims and all it claims. It is also
correctly split into a sweep floor and a composition floor
(`cartridge_benchmark.py` L229-L230), because a composed arm trains two
cartridges and runs a doubled prefix and is noisier for reasons that say
nothing about the sweep.

Two reading errors, not implementation errors:

1. **It is not a significance test for a paired step.** It compares unpaired
   spreads; the step is a paired difference.
2. **Floors are not comparable across plans with different seed counts.** The
   number moves with `n` on its own.

Both were made here, and the second is the one that invites the mistake: "add
seeds and re-read the floor" makes a measurement *less* conclusive.

## The record could not have settled this, and now can

Until commit 0633816e the record carried only `{arm}_mean` and `{arm}_spread`.
The per-seed gains existed inside the process and were dropped on the way out,
so no later reader could compute a paired statistic from a stored run.

`per_seed_observations` (`replicated_measurement.py` L245) emits one named
scalar per (arm, seed) — `slots-32_seed7_gain` — which is what `RunRecord`
carries: a flat name-to-float mapping. The `ReplicatedGain` codec had encoded
exactly these fields all along; what was missing was never the serialisation,
only the call.

The test that earns it does not check that the names appear. Two arms with
gains `(0.10, 0.20, 0.30)` and `(0.30, 0.20, 0.10)` have identical means and
identical spreads — indistinguishable in the old record — while their paired
differences are `+0.2/0.0/-0.2` and `0.0/0.0/0.0`. One arm moved every draw and
the other did nothing.

## Consequences for the pages that used the floor

[[model-trainer-composition-ceiling]] and
[[model-trainer-companioned-training-recipe]] both judge against
`noise_floor`, and **their verdicts stand**. Every plan in that arc runs seeds
7/8/9, so no floor there is ever compared across seed counts, and the arc uses
the floor as a magnitude bound on a composed mean rather than as a
significance test on a paired step — its headline effects are far above floor
scale.

The direction of the error also matters: discarding the pairing **under**-powers.
An instrument that ignores pairing can miss a real effect; it cannot mint a
false one. So the open question those pages inherit is whether cells recorded
as noise-adjacent are in fact resolved — a strengthening risk, not a
retraction risk.

## What the capacity sweep actually shows

gpt2 is still gaining at 512 slots with no saturation point in range, and that
now holds across the whole sweep, 2..512, rather than the 2..32 the 2026-09-04
narrowing claimed. The gain per 4x step continues to fall — +0.092, +0.053,
+0.028, +0.027 — consistent with the logarithmic scaling the plan's slot counts
were chosen for, but every step is separated from zero.
