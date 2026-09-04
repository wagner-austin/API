---
title: Composition-aware training moves the compartment ceiling, with a dose curve and an overdose endpoint
tags: [ml, model-trainer, cartridges, composition, training-recipe]
related:
  - "[[model-trainer-composition-ceiling]]"
  - "[[model-trainer-cartridge-question-set]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/cli/cartridge_companion_sweep.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_companioned.py
  - services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/cartridge_model.py
  - docs/RESEARCH.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/cli/cartridge_companion_sweep.py": 832de4e79068336b1cb8e6491d8b0553029f528a
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_companioned.py": 9cb8dec410c4bb82a4a5dfddb693a46d18a02252
  "services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/cartridge_model.py": 4e4c110a6adcd5c917d68b45e1de9e4ed320de56
  "docs/RESEARCH.md": 18d598ede67c6522bf77fa343eeae214d9fcde58
provenance:
  - "measured 2026-09-04 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1"
  - "record bit-identical across two full-grid processes: sha256 9e87e81642a10db614159e0a8e3ef8ee (truncated), plan gpt2-companions, seeds 7/8/9"
  - "baseline being moved: the a67d6038 composition-sweep record (fixed-64: n2 62.8%, n4 -45.4%)"
  - "board task bc29dc3e-c32f-4e77-b2b8-e98c11564299 carries the full trail including three instrument-caught defects"
fact_checked: "2026-09-04"
confidence: high
hubs: [services]
---

# Composition-aware training moves the compartment ceiling

The [[model-trainer-composition-ceiling]] finding said two compartments was
the limit for naively trained cartridges. This page records the intervention
that moves it: train every cartridge with a frozen companion concatenated in
front of its slots at a per-step probability, so composition stops being an
untrained capability. With the best recipe, four-compartment retention goes
from **-45.4% to +44.6%** -- a +0.78 swing on the composed mean against a
0.049 noise floor, from the grid's tightest cell -- while two-compartment
retention rises from 62.8% to 78.3% and solo performance costs four
hundredths of gain. The record is bit-identical across two full-grid
processes.

## The grid and the dose curve

Two companion kinds crossed with presence probability and compartment count,
every cartridge in a cell trained companioned, measured by the same arms and
controls as [[model-trainer-composition-ceiling]] (untrained-composed
control, cross-gain relatedness arms, solo-cost axis mandatory), so every
row below subtracts cleanly against that page's baseline:

| recipe | alone (solo cost) | n2 retention | n4 retention |
|---|---|---|---|
| naive baseline | +0.8897 | 62.8% | **-45.4%** |
| noise p=0.25 | +0.8558 (-0.03) | 66.2% | -1.5% |
| noise p=0.5 | +0.8323 (-0.06) | 66.8% | +3.4% |
| trained p=0.25 | +0.8671 (-0.02) | 77.1% | +32.4% |
| **trained p=0.5** | **+0.8466 (-0.04)** | **78.3%** | **+44.6%** |
| p=1.0 (either kind) | solo destroyed | | |

Content-companionship beats noise-companionship on every axis: a real
stranger teaches attention competition better than static. The overdose
endpoint replicates in both kinds -- a cartridge trained under perpetual
company never learns to stand alone (noise -0.68; trained -0.32, and the
trained-p1.0 artifact's composed arms BEAT its alone arm, a cartridge
adapted to company). The companion corpus is held out from every
composition partner and the CLI refuses the overlap, so the robustness is
generalised, not partner memorisation.

## What the machinery guarantees

The companion is frozen by construction, not convention: its blocks are
detached at every forward, the optimizer sees only the trainee's slots, and
a test proves the companion byte-identical after a full training run. The
presence draw consumes the global generator at every probability including
1.0, so the p-sweep's arms share one RNG-consumption pattern and vary
exactly one thing. Training remains a pure function of its seed with the
companion machinery included, which is what the bit-identity certificate
rests on. A cell whose alone arm did not improve on the base carries its
raw arm means but no retention ratio, because a ratio against a non-gain
has no reading -- the p=1.0 collapse is a real cell every full grid hits,
and the first version of the CLI died on it.

## What this binds, and what is still open

For the compartmental serving design the recipe changes the operating
point: four simultaneously wired compartments are viable at trained-p0.5,
where naive training made them destructive. Still open, filed rather than
implied: whether the recipe holds at n8 and when deployment count exceeds
training exposure (varied-count companionship is the obvious v2), the
budget slot policy under companioned training, the V100 replication (the
newest cluster image predates the ratio-absence commit, the third live
catch for the image-freshness gap), and the scale rung -- whether any of
this survives a 7B base, which is cheap to ask because only slots carry
optimizer state. The RESEARCH.md entry under `mi` carries the run summary
and the extension list.
