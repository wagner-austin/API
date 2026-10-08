---
title: "Measuring a DISPOSITION needs a different dependent variable, and a second number beside it"
tags: [ml, model-trainer, cartridges, composition, traits, steering, measurement]
related:
  - "[[model-trainer-composition-ceiling]]"
  - "[[model-trainer-companioned-training-recipe]]"
  - "[[corpus-attachment-program]]"
  - "[[model-trainer-cartridge-question-set]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/cli/cartridge_trait_sweep.py
  - services/Model-Trainer/src/model_trainer/core/services/model/trait_arms.py
  - services/Model-Trainer/src/model_trainer/core/services/model/steering_vectors.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_scoring.py
  - services/Model-Trainer/src/model_trainer/core/contracts/trait_corpus.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_trait_plans.py
  - services/Model-Trainer/src/model_trainer/core/services/model/trait_roster.py
  - services/Model-Trainer/src/model_trainer/core/contracts/trait_plan.py
  - services/Model-Trainer/src/model_trainer/core/services/model/cartridge_measurement.py
  - services/Model-Trainer/src/model_trainer/cli/cartridge_trait_repair_sweep.py
  - docs/RESEARCH.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/cli/cartridge_trait_sweep.py": 09d07ae1dc34d45cf0beec0ebf48fdd262a46deb
  "services/Model-Trainer/src/model_trainer/core/services/model/trait_arms.py": bef8a485465c7d7ee19d7d83f74ba6d0d622fb4c
  "services/Model-Trainer/src/model_trainer/core/services/model/steering_vectors.py": e58bdfe51776fab7930905a05140d65c2e26e059
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_scoring.py": cc0a3b92e1157f2641a726ca8fb91fa14c3cf9c2
  "services/Model-Trainer/src/model_trainer/core/contracts/trait_corpus.py": dbf142aa49b044fd46ac095c6ecb0597ba586f4c
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_trait_plans.py": 890fab22058944f130aa642f59f8453fd8898fc1
  "services/Model-Trainer/src/model_trainer/core/services/model/trait_roster.py": c0d7aed65c0cfa0d6595a09cdaf7954d73096ce0
  "services/Model-Trainer/src/model_trainer/core/contracts/trait_plan.py": fb9d508936acc2ea84aff9a1f356a98a65f08376
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_measurement.py": 944ff9cfe755ed0d0968fed698d20d3852d50bc0
  "services/Model-Trainer/src/model_trainer/cli/cartridge_trait_repair_sweep.py": 4b9fe871de209d7c121c03714470270bce562785
  "docs/RESEARCH.md": 0a08b4f35dd2dc12852613006ee76f47a3e06ba0
provenance:
  - "landed 2026-09-11 as commit a8afd6c9 (repo api); make check green in services/Model-Trainer (3455 passed, 100.00% statements and branches, 14849 statements / 2468 branches, none missed) and in libs/platform_core (1447 passed, 100.00%)"
  - "re-read 2026-10-08 against API 38cbd5011 for MCPs board task 3d71a8e1, over commits d2ac35525 (core: style reading, tuned steering arm, derived SEI and seed gate, seed_stride), 6c5a2b293 (cli: seed blocks and shards, the solo precondition as a record row, cartridge_trait_repair_sweep) and 24b4ad65b (docs/RESEARCH.md section rewritten), all three for board task 83c25b86. The page said two readings, a refusal for the solo precondition, an untuned steering arm, a 0.375 smallest effect of interest and no repair families; each is corrected below, and trait_roster.py, trait_plan.py, cartridge_measurement.py and cartridge_trait_repair_sweep.py join source_paths because the corrected claims are theirs"
  - "Run on the cluster since 2026-09-11 as a three-seed variance pilot only, not as the full grid: per docs/RESEARCH.md, run documents tools/hpc3/runs/cartridge-traits-gpt2-pilot-cpu-v56.json and its -twin on image v56 (the v54 documents this page used to name could never have run, because v54 holds no trait code); the twin, job 57928006, completed. The corpus stage document is tools/hpc3/runs/trait-corpus-stage.json"
  - "board task 83c25b86-00e5-44bf-9b84-7f3d71f64de4 carries the spec, its two self-corrections, and the scope deviation stated below"
  - "the published account this arm is built to be compared against is on the personal wiki: subbiah-2026-limits-of-steering-vectors and han-2026-steer2adapt-composition, under the parametric-knowledge-and-model-editing hub"
fact_checked: "2026-10-08"
confidence: high
hubs: [services]
---

# Measuring a disposition needs a different dependent variable

Every compartment the cartridge arc had composed before this one was a
CORPUS. The dependent variable is held-out loss on the compartment's own
text, and what it measures is "the model knows the corpus" — which is the
right question for a wiki and the wrong one for a persona, because a persona
has no held-out corpus in that sense. So the composition ceiling measured in
[[model-trainer-composition-ceiling]] and the recipe that moved it in
[[model-trainer-companioned-training-recipe]] are both results about
knowledge, and whether the same curve holds for dispositions was not
askable with the existing instrument.

`cartridge_trait_sweep` asks it. Three properties of the instrument are worth
recording because each was forced by the measurement rather than chosen, and
each has a failure mode that produces a complete, plausible table.

## The score is a difference of differences

Per trait, a set of prompts each continued twice: once exhibiting the trait,
once not. Each arm is scored as a PREFERENCE — the loss gap between the two
continuations — and the comparison is between the plain base's preference and
the cartridge's.

A bare loss on trait-expressing text cannot answer the question, because a
base with any English priors already prefers some phrasings, and a prefix that
merely lowers loss everywhere would read as trait acquisition. Differencing
the two members cancels whatever the prefix does to both. Every reading maps
onto `PairedItemOutcome`, so the existing statistical layer applies unchanged
and each carries its own McNemar exact conditional test and its own outcomes
digest — and no judge model enters the measurement path, which is what keeps
two runs on one node bit-identical and is why the field-standard judge
approach was rejected for this arc.

## The cancellation that makes it work also makes a second number mandatory

The expression reading is blind BY CONSTRUCTION to a cartridge that expresses
a trait by becoming worse at everything: a prefix that damages both members
equally scores exactly zero. So the neutral member — the trait-free text
already in the measurement — carries a COHERENCE reading, taken on the same
forward passes.

They are different linear combinations of the same four losses.
`measure_trait_losses` runs the model once and pure reducers derive them,
and `trait_arm_observations` emits every reading from one function, which is
what makes it impossible for an arm to reach a record with an expression
number and no coherence number beside it. This is the difference between a
control that exists and a control nobody can forget to run.

Since 2026-10-07 there is a THIRD reading, and the three decompose exactly.
Style is the loss gain on the trait-expressing member alone, so expression =
style - coherence pair by pair (`cartridge_scoring.style_outcomes`). Style is
the corpus arc's own dependent variable applied to a style corpus, which is
why a `_style_retention` row now sits beside every `_expression_retention`:
where the two agree, the persona result is a style-corpus result and has to
be named as one. There is still no coherence retention, deliberately: a ratio
of two fluency costs is not a retained fraction of anything, so coherence is
read as a difference against the controls in the same record.

## The steering arm has no seeds, and the names say so

A contrastive activation vector is the mean difference over a fixed pair set.
Nothing is drawn, so running it three times produces one number three times.
Wrapping that in a `ReplicatedGain` would report a spread of exactly zero
beside cartridge arms whose spreads are seed noise, inviting a reader to
compare two estimates of different things. Its rows are named `_once` rather
than `_mean`, and in a flat mapping of names to floats the suffix is the only
place that distinction can survive.

The arm reads and perturbs the SAME tensor-valued module, which is why the
site is declared per plan rather than derived from depth — the layer at which
a contrastive direction is readable is a property of the model and this
programme has not measured it, so a derived layer would put an unchosen number
in every record.

**Its strength is tuned, since 2026-10-07, by the published rule.** The arm
used to run at strength 1.0 on a unit direction, and on gpt2 at that strength
it moved expression by +0.004 nats. `steering_vectors.tune_steering_strength`
now tries each strength in the plan's `steering_strengths` grid on the
TRAINING pairs and keeps the most expressive one whose coherence cost stays
within `steering_coherence_bar`, which the plans set to the trait cartridge's
own solo coherence cost in the pilot (1.8611 nats), so the two substrates are
compared at matched fluency. The bar is in this instrument's units, not the
paper's judge score, and no strength in the grid inside the bar is refused
with `TRAIT_STEERING_STRENGTH_UNREACHABLE`.

## Two gates run before the hours are spent, and the solo cell is a verdict

`trait_roster.gate_primary_trait` runs both gates before a model loads.

The first reads the REALISED held-out pair count through
`require_resolvable_pairs`. The committed corpus authors 32 pairs per trait
and the stride holds out half, so 16 are scored; 6 of 16 is the smallest net
rate any attainable outcome could report as significant at alpha 0.05 under
the exact test, and every plan declares exactly that 0.375. Until 2026-10-07
the plans called that number their smallest effect of interest, which is the
move [[model-trainer-cartridge-question-set]] had to learn to refuse after a
headline was published four times below what its instrument could resolve: a
threshold chosen so the corpus clears it is not a threshold. It is now
`pair_test_floor`, and all it claims is that the per-pair test is falsifiable
on this corpus.

The second, `require_resolvable_seeds`, holds the plans to an effect size that
is DERIVED rather than chosen. The smallest effect of interest is
`acted_on_retention` 0.0526 (the retention margin the corpus arc's operating
point was adopted on) times the pilot's solo gain `pilot_alone_gain` 2.6144
nats, 0.1375 nats, and the pilot's paired differences set the spread
`required_replicates` needs to resolve it: 138 seeds. Every plan now declares
seeds 7 to 144, and a plan with fewer is refused with
`CARTRIDGE_QA_UNDERPOWERED`.

The solo precondition is NOT a refusal any more. The 7B rung of the corpus
programme failed at exactly that step and not at composition — solo gain
+0.068 against a per-seed span of the same order — and every retention ratio
computed on those records became a division artefact, so the solo cell still
runs first and the composed cells still run only if the solo expression
cleared its own spread. What changed on 2026-10-07 is where a failure lands:
`trait_arms.solo_precondition_cleared` returns a verdict, the record carries
it as the `solo_precondition_cleared` row, and a failed run still writes its
solo and steering rows before it stops. A failed precondition is the first
result the task names, a null about the substrate, and a raise would have left
it in a job log. The untrained-prefix arm is still recorded beside the solo
arm: a solo gain that merely matches an untrained prefix is the other way this
fails.

## Two lifts rather than two forks

The composition geometry — seed offsets, fold order, the untrained-composed
draws — now has one owner that both the corpus grid and the trait grid drive.
A second copy would not have failed; it would have produced a complete table
whose arm names no longer line up with the recorded ladder, which is worse.

`composed_replicates` hands each replicate to a consumer rather than returning
a list, and that shape is load-bearing. Scoring puts the base in evaluation
mode, and the geometry probe inside `fresh_cartridge` consumes process-wide
randomness when the base is in TRAINING mode and none when it is in
evaluation mode — so a caller that collected every replicate before scoring
would hand training a different RNG state and get different numbers from the
same inputs. A list makes that impossible and a generator makes it merely
conventional; the callback makes it structural.

Since 2026-10-07 it also takes an explicit `seed_stride`, the PLAN's seed
count rather than the call's. At 138 seeds one job is a day of CPU, so the
sweep measures every cell in blocks of three seeds and can run them as shards
(`--shards DIR --shard-count N --shard K`) and adopt them in a merge
(`--shards DIR --shard-count N`). Passing the whole plan's count keeps a
block's partner draws exactly the ones a straight run would make, so no
block's partner lands on another block's primary seed; the corpus sweeps pass
`len(seeds)`, which is what the offsets always were.

## The repairs came after the baseline, and neither has a full-grid result

The diverse-companion, base-LoRA and crowd-invariance families were not built
when this page was written. Those three are interventions that REPAIR
composition, and the question they answer is only askable once a naive
interference number exists to repair — which is the order the corpus arc was
built in, each sweep importing the one before it. Building the repair before
the baseline would have been the same mistake the solo precondition exists to
prevent, one level up.

They exist now, as `cartridge_trait_repair_sweep` with one plan per lever:
`gpt2-traits-diverse`, `gpt2-traits-base-lora` and `gpt2-traits-content-lora`.
Each reads the naive plan's row by identity, so its cells subtract against
this sweep's, and builds its adapters through the lifted
`cartridge_crowd_adapters` module. The order still holds in the evidence:
only the naive grid has been measured, in a three-seed pilot, and neither the
138-seed naive grid nor any repair plan has a result yet.

The authored corpus was also invisible to git until the commit that landed
this. `services/Model-Trainer/corpus/` is ignored for downloaded bulk, and
re-including four hand-written files took three rules across two ignore files,
because git does not descend into an excluded DIRECTORY — a negation for a
file inside one is inert and reads as though it works. The suite that reads
those files to check the plans' declared effect would have passed here and
failed on any fresh checkout.
