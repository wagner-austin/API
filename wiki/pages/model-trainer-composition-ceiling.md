---
title: Cartridge composition has a ceiling of two compartments, measured with its own artifact detectors
tags: [ml, model-trainer, cartridges, composition, measurement]
related:
  - "[[model-trainer-companioned-training-recipe]]"
  - "[[monorepo-discipline]]"
  - "[[service-port-map]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/cli/cartridge_composition_sweep.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_measurement.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_plans.py
  - docs/RESEARCH.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/cli/cartridge_composition_sweep.py": 474530669efb45fff0d4725f11af1769068c7bf7
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_measurement.py": 81d64e75ebb103e728c46352bac905e25dc6b46e
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_plans.py": 49362ef25268637e431de591599fb509795f77b0
  "docs/RESEARCH.md": ae83aff837c0ec4f4b0639c1e5244867c05c59e8
provenance:
  - "measured 2026-09-04 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1"
  - "v2 record bit-identical across two processes: sha256 aa61330b9692f4c4bc13b410f4bf1874 (truncated), plan label gpt2-compartments-gpt2-w256-s4-e12-lr0.01-n2.4.8-f64-b512-seeds7.8.9"
  - "earlier pair agreed on 90 of 90 shared observations across a record-shape change"
  - "board task a67d6038-ef16-4bcf-acbc-77b47f7fd4ad carries the full trail including the two artifacts the run caught"
fact_checked: "2026-09-04"
confidence: high
hubs: [services]
---

# Cartridge composition has a ceiling of two compartments

The compartmental serving design assumed several wiki cartridges could be
wired into one request. Measured on a real pretrained gpt2 with
independently trained cartridges, the assumption fails past two: clean-roster
retention of the primary cartridge's held-out gain under the fixed-64-slot
policy is 62.8% at two compartments, negative (-45.4%) at four, and erased
(-7.0%) at eight; under a fixed 512-slot budget it is 44.3%, +14.4%, and
-7.0%. The collapse replicated across three roster rotations: n2 sits at
59-73% whichever corpus partners, n4 is negative in every fixed-policy
roster, and n8 erases the gain under both policies. The measurement is
`cartridge_composition_sweep` (source pinned above) and the numbers carry a
bit-identity certificate: two separate processes produced sha256-identical
records.

## The cost decomposes, and the attribution flips with scale

Every configuration runs an untrained-composed control: the trained primary
composed with freshly drawn noise cartridges of identical shape. At n2 the
cost is structural -- noise alone retains 41% and trained content adds about
twenty points back -- and by n4 content interference crosses over, with
trained strangers costing more than noise (-0.29 trained vs -0.12 untrained
at fixed-n4). Corpus identity modulates amplitude (n2 ranges 59-73% across
three different partner corpora) but never the shape, and a corpus's
hostility alone does not predict its composition damage: the most hostile
cross-gain corpus (-0.42) composed as benignly at n2 as the friendliest.

## The run caught two of its own artifacts

The first roster's n8 read +27% retention and was leakage: cartridges
trained on the tech and hpc3 wikis predict me-wiki text they never saw
(+0.18 and +0.41 cross-gain), because the operator's narrative wiki shares
their vocabulary. The cross-gain arm -- every foreign cartridge scored alone
on the primary held-out text -- exists in the record precisely to catch
this, and the clean-roster rerun put n8 at -7.0%. Relatedness between
compartments is measured, never assumed. The second artifact was
operational: MSYS path conversion rewrites only the first `/c/...` path in a
comma-joined argument, and the mangled remainder reached the corpus reader
as a directory that globs empty, which the reader refused loudly rather
than training on nothing.

## What this binds, and the way past it, which has since been measured

For the compartmental serving design the practical limit at this scale
UNDER NAIVE TRAINING is two simultaneously wired compartments, with
compartment selection mattering less than count. The escape route the
literature names -- composition-aware training, the ICAE multi-span
finding that concatenation works only when trained for -- was measured the
following day and MOVES this ceiling: see
[[model-trainer-companioned-training-recipe]], where the trained-p0.5
recipe puts n4 retention at +44.6% against this page's -45.4%. This page's
numbers stand as the naive baseline that recipe is judged against.
`docs/RESEARCH.md` carries both run summaries under the `mi` project.
