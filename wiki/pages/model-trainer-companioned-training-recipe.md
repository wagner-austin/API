---
title: Composition-aware training moves the compartment ceiling, with a dose curve and an overdose endpoint
tags: [ml, model-trainer, cartridges, composition, training-recipe]
related:
  - "[[model-trainer-composition-ceiling]]"
  - "[[model-trainer-cartridge-question-set]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/cli/cartridge_companion_sweep.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_companioned.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_varied_companion_sweep.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_diverse_companion_sweep.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_varied.py
  - services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/cartridge_model.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_base_lora.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_base_lora_sweep.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_content_lora.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_content_lora_sweep.py
  - docs/RESEARCH.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/cli/cartridge_companion_sweep.py": 9300942991b85dcc3bb354ecca84e4680ce9ee2d
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_companioned.py": 9cb8dec410c4bb82a4a5dfddb693a46d18a02252
  "services/Model-Trainer/src/model_trainer/cli/cartridge_varied_companion_sweep.py": 86c614151f8752fb3e16f78fca41f949448bdfab
  "services/Model-Trainer/src/model_trainer/cli/cartridge_diverse_companion_sweep.py": f41deeec935e92643b7263ab6e564cdd2e9347b8
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_varied.py": ceb89138c973e1f2d60bf1ddf8c5d04814903533
  "services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/cartridge_model.py": 75b3370cb8fd7ba5a7d5cac712e2a61c3abe6fdb
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_base_lora.py": 51621f94781dd5b27149fcc5931c4bb6e7209006
  "services/Model-Trainer/src/model_trainer/cli/cartridge_base_lora_sweep.py": 6e96538b0370caebbe7e459104283392b9a2941a
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_content_lora.py": 6950eadcbe579b6ee9b3cff54110b5c448baafd7
  "services/Model-Trainer/src/model_trainer/cli/cartridge_content_lora_sweep.py": dc758263db2d9ac7e08156d40fd79e11d728b12e
  "docs/RESEARCH.md": 09a58adce295fb8476f124041a0147b76659b155
provenance:
  - "measured 2026-09-04 on austinpc, RTX 3090 Ti, driver 591.86, HF_HUB_OFFLINE=1"
  - "record bit-identical across two full-grid processes: sha256 9e87e81642a10db614159e0a8e3ef8ee (truncated), plan gpt2-companions, seeds 7/8/9"
  - "baseline being moved: the a67d6038 composition-sweep record (fixed-64: n2 62.8%, n4 -45.4%)"
  - "board task bc29dc3e-c32f-4e77-b2b8-e98c11564299 carries the full trail including three instrument-caught defects"
  - "n8 cell measured 2026-09-04 on HPC3: job 55753007, Tesla V100-FHHL-16GB, driver 580.82.07, image v34 sha256 cdd1341b (truncated), plan gpt2-companions-n8, board task 684492dd"
  - "n8 record bit-identical across two DIFFERENT V100 nodes (jobs 55753007 on gpu-18-02 and 55753873 on gpu-16-04, both records sha256 6e63dad7 truncated) -- cross-node determinism"
  - "varied-count cells measured 2026-09-04 on HPC3: jobs 55759514/55761217, both on hpc3-gpu-16-02 (V100), image v35 sha256 4e02f3b0 (truncated), plan gpt2-companions-varied, records bit-identical sha256 1fd6bb9d (truncated), board task 7815a0fd"
  - "diverse-pool cells measured 2026-09-05 on HPC3: job 55772675 on hpc3-gpu-17-03 (V100) + twin 55773234, image v36 sha256 0401aa9b (truncated), plan gpt2-companions-diverse, board task d2c03dd4; companion-cross instrument in-record"
  - "n2/n4 grid REPLICATED on a V100 2026-09-05 (job 55773639, v36): every verdict survives the card -- trained-p0.5 78.0%/41.4% vs the 3090 Ti's 78.3%/44.6%, orderings identical, overdose replicated in both kinds"
  - "scale rung measured 2026-09-05 on HPC3: job 55776517 on hpc3-gpu-16-00 (V100) + twin 55786853 on hpc3-gpu-18-00, image v37 sha256 2adee62f (truncated), plan gpt2-medium-companions-diverse, plan commit e5476201, records CROSS-NODE BIT-IDENTICAL sha256 5179b893 (truncated)"
  - "base-LoRA cells measured 2026-09-05 on HPC3: gpt2 jobs 55787810 (gpu-16-05) + 55788364 (gpu-18-02) CROSS-NODE BIT-IDENTICAL sha256 efad0a93 (truncated), image v38 sha256 13bd47e9, plan commit e1bc2009; gpt2-medium job 55790169 (gpu-17-02), image v39 sha256 0bd7983b, plan commit f7f696a8, board task 6c752568"
  - "gpt2-medium base-LoRA record CROSS-NODE BIT-IDENTICAL 2026-09-06: job 55790169 (gpu-17-02, 78 min) + twin 55798416 (gpu-18-02, 2h13m -- same bytes, slower node), both records sha256 372cee59 (truncated)"
  - "content-lora (crowd-invariance) cells measured 2026-09-06 on HPC3: job 55801429 (V100, 103 min), image v40 sha256 798d234a (truncated) from commit 7288bc8f, plan gpt2-medium-content-lora, board task a85fbabe; twin 55801941 BYTE-IDENTICAL, both records sha256 9abd901a (truncated) -- SAME-NODE certificate (queue placed both on hpc3-gpu-16-01); the arc's four cross-node certificates establish the pipeline's cross-node determinism separately"
  - "the v40 build and the medium run were both caught by the hpc-wake bridge's tagged board announcements (task f6b04193) with no manual sacct polling -- the arm that retired hand-rolled waits"
  - "gpt2-small invariance anchor measured 2026-09-07 on HPC3: job 55806539 + twin 55806826 on hpc3-gpu-18-01 and hpc3-gpu-16-00 (~40 min each), unchanged v40 image, plan gpt2-content-lora, records CROSS-NODE BIT-IDENTICAL sha256 9abfbdd4 (truncated), board task 7642f7f9"
  - "1.5B rung (gpt2-xl) measured 2026-09-07 on HPC3, BOTH objectives on A30-24GB: jobs 55808450 (LM, 3h27m) + 55808466 (invariance, 3h28m), image v41 sha256 f38bc982 (truncated) from commit 40c55fa5; first content attempt 55807973 hit CUDA OOM on V100-16GB by 50MiB at the KL step (two 1.5B fp32 models), so the pair moved cards together, gpu_pinned_because declared in both run documents"
  - "1.5B twins 55809977/55809982 BYTE-IDENTICAL 2026-09-07: LM pair sha256 cff9f3ce (truncated), invariance pair sha256 6269120d (truncated) -- SAME-NODE certificates, the queue having placed all four runs on hpc3-gpu-l54-09; cross-node determinism established separately by the arc's five cross-node certificates"
  - "7B rung (Pythia-6.9B, GPT-NeoX, NF4 nf4+double-quant+bf16-compute) measured 2026-09-07 on A30, board task af35fc20: jobs 55810964 (LM, 3h04m) + 55810966 (invariance, 3h21m), image v42 sha256 270d8197 (truncated) from commit 09f5a4bb, 44/44 smokes incl. the dtype-boundary and architecture-policy self-asserts; model content-pinned (snapshot c0e3eee3 + shard sha256s) in the run documents"
  - "7B LM twin 55811523 BYTE-IDENTICAL (sha256 6ef4b9c9 truncated, same-node hpc3-gpu-l54-09); 7B invariance twin 55811542 CROSS-NODE (k54-01 vs l54-09) with ALL 198 observations bit-equal -- record shas 254d0126 vs 8c48e82a differ solely in fingerprint/host/logical_cores 32 vs 64, so NF4 training is certified deterministic cross-node in every measured quantity"
  - "headroom measurement 2026-09-07, board task afee6162: cartridge_headroom CLI (commit 77e12c3e), image v44 sha256 9e98d0a7 (truncated), jobs 55812858 + twin 55812861 (~4 min each on A30), records BYTE-IDENTICAL sha256 7668c51d (truncated), same-node hpc3-gpu-k54-01; plain-base held-out loss on the primary corpus 4.57 (gpt2) / 4.27 (medium) / 4.11 (xl) / 3.66 (pythia-6.9b NF4), pythia 0.34-0.90 nats below every GPT-2 base on all 11 corpora; per-corpus character and token rows carried for the tokenizer caveat (pythia ~7% fewer tokens on the same text)"
  - "nine-seed solo reliability 2026-09-07/08, board task b89cd348: cartridge_solo_seeds CLI (commit 4fc8dfbb), image v45 sha256 567cb42d (truncated), solo cartridges trained AND scored behind the PLAIN base at the recorded knobs; 7B pair 55813508 + twin 55813516 BYTE-IDENTICAL sha256 353ab575 (truncated, same-node k54-01); xl control pair 55818092 + twin 55813528 CROSS-NODE (l54-09 / l54-07) with all 14 observations bit-equal, shas differing solely in fingerprint/host/logical_cores 64 vs 32; xl mean +0.827 spread 0.079 over nine seeds, pythia-6.9b/NF4 mean +0.160 spread 0.320 with one negative draw"
  - "bf16 precision control 2026-09-09, board task c4b9a01b: StoredBf16Precision loader state (commit f4447989), image v46 sha256 bd6ca365 (truncated), jobs 55833896 + twin 55833931 (~9 min each, half NF4's wall clock), records BYTE-IDENTICAL sha256 445e345f (truncated) ACROSS nodes gpu-24-07 / gpu-l54-07 -- the arc's first full cross-node byte identity on a 7B record; bf16 mean +0.102 spread 0.248 two negatives; paired bf16-minus-NF4 -0.058 +/- 0.032 (t -1.80, MDE 0.075 nats): NF4 exonerated"
  - "MDE rows computed 2026-09-09 from stored per-seed rows (machine-wide MDE standard): 1.5B n8-equals-n4 holds (LM +0.0004 vs MDE 0.019; invariance +0.0001 vs 0.068); objectives' diverse-n4 tie at 1.5B holds (-0.008 vs MDE 0.047); diverse-n8 CORRECTED from tie to a resolved -0.0088 +/- 0.0014 LM advantage (t ~ -6.5), ~1% of the alone gain, no operating decision changes"
  - "hyperparameter grid 2026-09-09, board task 47d5f8c6 (operator-directed): cartridge_solo_grid CLI (commit 103bdaf7), image v47 sha256 290794b0 (truncated) from fcb39991, jobs 55841983 + twin 55841997 (1h08m each), records BYTE-IDENTICAL sha256 bc18d701 (truncated, same-node k54-05); the lr0.01xc64 anchor cell reproduces the certified 445e345f bf16 record BIT-FOR-BIT seed for seed; lr {0.001,0.003,0.01,0.03} x slots {64,256} at nine seeds: 0.001 all-negative, 0.03 divergent (means to -1.67), 0.003 indistinguishable-to-worse (t -1.04), 256 slots never better than 64 (at lr0.003 significantly worse, t -5.19) -- the recorded knobs sit at the grid's maximum"
fact_checked: "2026-09-07"
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

## Varied-count exposure, refuted at its target

The obvious v2 -- train beside a DRAWN number of companions (uniform 1..3
when present) so the recipe learns count-invariance -- was measured on the
cluster the same day (plan `gpt2-companions-varied`, jobs 55759514/55761217
bit-identical on a V100 under image v35) and it does NOT close the decay:
n8 retention reads +18.3% against the single-companion +26.5%, a composed
difference at ~1x the cell spreads, at a higher solo cost (-0.063). What
it does buy is n4: +51.0% against 44.6% with the composed spread
collapsing 0.049 to 0.010, the tightest composed cell in the program. The
mechanism reading is the finding: count-invariance WAS learned -- the
untrained-composed controls sit far above every earlier grid's
noise-composition arms -- yet real strangers still interfere, so the
count-decay is CONTENT interference, confirming from the opposite
direction that content-companionship is the load-bearing ingredient. A
pool of three same-corpus companions cannot teach content diversity; a
content-DIVERSE pool is the motivated follow-on, with this record as its
baseline.

## The recipe under seven strangers

The open question above the grid -- does single-companion training
survive when deployment count exceeds training exposure -- was measured
the same day on the cluster (plan `gpt2-companions-n8`, job 55753007 on
a V100 under image v34, whose own smoke asserts the plan's shape). It
does, degraded but decisive: trained-p0.5 puts eight-compartment
retention at **+26.5%** (composed +0.2243, spread 0.0562) where the
naive baseline was -7.0%, at the same four-hundredths solo cost, and
trained-p0.25 reads +22.2%. Noise companionship's n4 break-even
VANISHES at n8 (-4.4/-4.6%, indistinguishable from naive) -- content is
the load-bearing ingredient, and the n8 trained cells' composed arms sit
at their untrained-composed controls (+0.02): content interference
erased, where noise-trained cartridges lose -0.39/-0.46 to real
content. The seventh partner (plant-eco, chosen over the two corpora the
baseline caught leaking) measured +0.05 cross-gain against a 0.048
spread -- clean, verified in-run. Dose and kind orderings replicate from
the n2/n4 grid.

## The diverse pool: best recipe, and the decay's cause settled

The content-diverse pool -- three companions each trained on a DIFFERENT
held-out corpus (epi, the recorded companion by the shared seed formula,
plus metabolomics and atmospheric-chemistry) -- was measured the same
night (plan `gpt2-companions-diverse`, jobs 55772675/55773234 on a V100
under image v36) and it is the program's best recipe at both counts: n4
retention **+55.5%** (vs the single companion's 44.6%, ~1.4x floor) and
n8 **+28.0%** -- decisively above the same-content pool's 18.3% and a
within-floor tie with the single companion's 26.5%, stated as a tie. A
new companion-cross instrument scores every pool member alone on the
primary held-out, and all three read negative (-0.36/-0.11/-0.04): the
pool is measured clean, not assumed. The mechanism verdict settles the
arc: the n8 composed arm EQUALS its untrained-composed control (+0.2323
vs +0.2431), so content interference at eight compartments is fully
trained away and the residual count-decay is STRUCTURAL slot dilution --
seven 64-slot strangers are 448 foreign positions against 64 own, a cost
with no content component left for any companionship recipe to remove.

## The scale rung inverts the ceiling

The recipe on a base three times the size (plan
`gpt2-medium-companions-diverse`, identical in every field but the base,
a contract pinned by test and by the image's own smoke) answers the
scale question in both directions at once: **n4 transfers near-exactly
(+54.1% retention against gpt2's +55.5%) and n8 COLLAPSES (-86.6%
against +28.0%)**. The controls attribute the collapse: medium's n8
untrained-composed arm is itself negative (-0.29 where gpt2's read
+0.24) -- the 24-layer base's structural tolerance for a 512-slot
foreign prefix is far worse than the 12-layer base's before content
enters -- and the composed arm sits another 0.42 below that, so the
diverse recipe's content-erasure did not transfer either. The schedule
is not the confound: it learns the solo cartridge (+0.81) and composes
four compartments (+0.44) on medium, and only the crowded-prefix regime
fails. Depth compounds prefix interference: scale alone COSTS
many-compartment composition rather than buying it, and the larger-base
path runs through base-side adaptation. Four-compartment deployment,
by contrast, is scale-robust at ~55% on both bases.

## The base learns the crowd, and the two levers stack

The lever every cartridge-side measurement pointed at: a rank-8 LoRA on
the base's attention trains to do language modeling behind a DRAWN number
of frozen composed cartridges (uniform one to eight, from the three
held-out pool corpora), through the same ``train_on`` loop that trained
every recorded cartridge -- only which side learns switched. On gpt2 it
settles both questions at once. Base-side alone repairs the STRUCTURAL
catastrophe (plain cartridges at n4: -45.4% naive to -6.9% adapted, with
the noise-composition control leaping -0.12 to +0.28) but leaves content
interference untouched -- necessary, not sufficient. Stacked with the
diverse recipe it sets the program's records: **n4 +58.1%, n8 +33.3%**
(against diverse-alone's 55.5%/28.0%, the n8 gain at twice the tightest
floor in the program), at ~zero solo cost. On gpt2-medium the verdict
splits and completes the mechanism map: the structural repair TRANSFERS
to depth (the n8 noise control flips -0.29 to +0.42, exactly the
quantity the scale rung measured as the collapse's structural half) and
n4 sets a new medium best (+59.3%), but real-content composition at n8
still collapses (-79.4%, composed a full 1.04 below the repaired noise
control, cell floor 0.53 -- seed-chaotic). Depth amplifies CONTENT
interference in a way neither lever touches; structure is solved at both
scales.

## The objective is the measured quantity: crowd-invariance closes depth

The content lever the previous section's residual named, run 2026-09-06
with ONE change from the base-LoRA arm: the LoRA's training objective.
Language modeling behind a crowd rewards reading past the crowd's shape
and says nothing about ignoring its content, so
`train_composition_lora_invariant` (source pinned above) distils
crowd-invariance instead -- per step it draws a roster and a target
member, takes the window from the target's own corpus, and minimises the
KL from the plain base's predictions behind the target ALONE to the
adapted base's predictions behind the full roster. Every roster position
is drawable as target, so no positional shortcut exists and every
compartment stays live; counts draw one to eight, so the alone case is
distilled too. The plan rows are pinned equal to their base-LoRA twins
field for field, by test and by image smoke, so the two records isolate
exactly the objective.

At gpt2-medium the depth collapse is repaired and surpassed: diverse n8
retains +38.1% where the LM objective recorded -79.4% -- clearing its
family floor by 1.9x and beating gpt2-small's own best n8 -- and diverse
n4 sets the program record at +63.2%. Plain cartridges flip positive at
both counts, the composed-below-noise-control content gap shrinks from
1.04 to 0.30, and the alone arms RISE, so the objective is free at solo
-- the count-one anchor working as designed. Both interference
mechanisms the arc named now have a working lever, and both levers are
base-side: the cartridges themselves need nothing.

## The ladder verdict: the collapse is a mid-depth valley

The scale ladder run down and up from the medium finding (provenance
below) settles the depth question with measurements at 12, 24 and 48
layers for both objectives. At gpt2-small the invariance objective wins
where nothing was collapsing -- diverse n4 +63.3% (vs the LM
objective's +58.1%), n8 +49.6% (vs +33.3%), plain positive at both
counts -- and its n4 ceiling matches medium's +63.2% to a tenth of a
point, so the four-compartment ceiling is scale-invariant under the
objective. At gpt2-xl (1.5B, 48 layers) the count penalty VANISHES:
diverse n8 equals n4 to a tenth under BOTH objectives (LM
+54.8%/+54.8%, invariance +52.8%/+52.8%; each record's own separation
flag between the two cells reads 0.0), so the 24-layer n8 collapse is a
mid-depth valley, not a depth law -- the ladder reads 33.3 → −79.4 →
+54.8 for LM n8 and 49.6 → 38.1 → 52.8 for invariance n8. On diverse at 1.5B the
paired per-seed rows resolve what the range floors could not: n4 is a
genuine tie (−0.008 against an MDE of 0.047) while at n8 invariance
sits a RESOLVED −0.0088 ± 0.0014 below the LM objective (t ≈ −6.5) —
about 1% of the alone gain, changing no operating decision, corrected
from the earlier "tie" wording when the MDE rows were computed. The
invariance objective's remaining margin there is plain cartridges
(+22.5%/+5.8% against the LM objective's −11.5%/−27.9%) and the
ladder-wide fact that it collapses nowhere. The n8
composed-below-noise-control content gap reads ~0.21 under both
objectives at 48 layers, so depth's content amplification at 24 layers
(1.04) does not extrapolate either direction.

## What this binds, and what is still open

For the compartmental serving design the recipe changes the operating
point: four simultaneously wired compartments are viable at trained-p0.5
where naive training made them destructive, and eight retain over a
quarter of the solo gain where naive training erased it -- with the
diverse pool as the recipe of record. Companionship itself is EXHAUSTED
as an n8 lever, by three convergent measurements. The n2/n4 grid has since been
replicated on a V100 with every verdict surviving the card (provenance
below). The base-side LoRA has since been measured on both bases (section
below): the operating point of record is base-LoRA + diverse cartridges
at up to FOUR simultaneous compartments, scale-robust at ~58-59% on both
bases. The content lever has since been measured (section above) and
retires the deep-base caveat: with crowd-invariance distillation on the
base, eight compartments are deliverable at depth (+38.1%) and four at
+63.2% is the best cell on any base -- the scope-router constraint is no
longer forced by measurement. The scale ladder has since completed
(section above): four-compartment serving is scale-robust at +53-63% on
every measured base, and the n8 question is depth-shaped -- worst at 24
layers, gone at 48. The 7B architecture jump has since been measured
(Pythia-6.9B under NF4, both objectives, provenance below) and the
recipe does NOT survive it as-is: the failure is the measurement's
PRECONDITION, not composition -- the solo cartridge gain nearly
vanishes (~0.07-0.23 against ~0.81 on every GPT-2 rung, per-seed spans
the size of the means), so retention there is a division artifact and
no 7B composition claim in either direction is founded until the solo
gain exists. NF4 training itself is certified deterministic, cross-node
in every measured quantity. The headroom measurement (provenance below)
has since ATTRIBUTED the collapse: the 7B's plain base already predicts
every corpus at the level the smaller bases reach only with a cartridge
-- plain-base loss 4.57/4.27/4.11/3.66 down the ladder on the primary
corpus -- so most of the ~0.81-nat gain was never available at 7B, the
residual (~0.36 nats to the family's adapted level) matches the best
measured 7B per-seed gain (+0.40) within floors, and what headroom does
not explain is seed variance (one seed negative where two reach ~0.4).
The seed-variance question has
since been answered at nine seeds (provenance below): the family was
never lucky -- gpt2-xl's nine draws all land in +0.78..+0.86 -- while
7B/NF4 training behind the plain base delivers a mean of only +0.16
with a spread twice that and one draw negative; the sweeps' ~0.4
readings carried the adapted base's contribution, and
pythia-plus-cartridge (~3.50) does not reach xl-plus-cartridge
(~3.28), so the training-side deficit is real beside headroom. The bf16 precision control has
since run (provenance below) and EXONERATES NF4: quantization removed,
the same nine seeds land in the same broken regime (mean +0.102, two
negatives; paired difference −0.058 against an MDE of 0.075). The
hyperparameter grid has since run too (provenance below) and REFUTES
the tuning hypothesis in its measured ranges: the recorded lr 0.01 ×
64 slots sits at the grid's maximum — the learning-rate bracket is
worse in both directions, catastrophically at 0.03, and 256 slots
never beats 64 — with the anchor cell reproducing the certified bf16
record bit for bit. After three eliminations (headroom measured, NF4
exonerated, lr/slots refuted) the standing verdict is that ~0.10-0.16
nats IS the 7B solo regime for this recipe at 12 epochs. Still open,
filed rather than implied: the last unmeasured training axis (epochs /
schedule) — past which the conclusion becomes that KV-prefix capacity
itself does not transfer to this architecture at this scale, a finding
about the method rather than the tuning — any 7B composition rung, the
mechanism of the mid-depth valley, the remaining 0.30 content gap at
medium n8, and the budget slot policy. The RESEARCH.md entry under `mi` carries all the run summaries
and the extension list.
