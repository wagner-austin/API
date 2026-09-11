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
  - docs/RESEARCH.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/cli/cartridge_trait_sweep.py": 6f5ef7165d162b177581781e21ee2d66479ba652
  "services/Model-Trainer/src/model_trainer/core/services/model/trait_arms.py": 69f6fc8387dee01398d7947dfb870df10ae515ff
  "services/Model-Trainer/src/model_trainer/core/services/model/steering_vectors.py": 251c77c38dd8ae58962d39249f6e8253b66f69bc
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_scoring.py": 8f7b0f4b355602ef7e7766b21c10327b2929c9d1
  "services/Model-Trainer/src/model_trainer/core/contracts/trait_corpus.py": dbf142aa49b044fd46ac095c6ecb0597ba586f4c
  "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_trait_plans.py": 9cbab0c9869186e5446f2c525947452418490cbe
  "docs/RESEARCH.md": 575670696f235494c5f7c5a9d61f3dcedd208e77
provenance:
  - "landed 2026-09-11 as commit a8afd6c9 (repo api); make check green in services/Model-Trainer (3455 passed, 100.00% statements and branches, 14849 statements / 2468 branches, none missed) and in libs/platform_core (1447 passed, 100.00%)"
  - "NOT RUN ON THE CLUSTER. This page records an instrument and its refusals, not a finding. Run documents tools/hpc3/runs/cartridge-traits-gpt2-v54.json and its -twin are committed against image c66460f05275 (v54); the corpus stage document is tools/hpc3/runs/trait-corpus-stage.json"
  - "board task 83c25b86-00e5-44bf-9b84-7f3d71f64de4 carries the spec, its two self-corrections, and the scope deviation stated below"
  - "the published account this arm is built to be compared against is on the personal wiki: subbiah-2026-limits-of-steering-vectors and han-2026-steer2adapt-composition, under the parametric-knowledge-and-model-editing hub"
fact_checked: "2026-09-11"
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
the two members cancels whatever the prefix does to both. Both readings map
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

The two are different linear combinations of the same four losses.
`measure_trait_losses` runs the model once and two pure reducers derive them,
and `trait_arm_observations` emits both from one function, which is what makes
it impossible for an arm to reach a record with an expression number and no
coherence number beside it. This is the difference between a control that
exists and a control nobody can forget to run.

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

## Two refusals run before the hours are spent

`require_resolvable_pairs` reads the REALISED held-out pair count before a
model loads. The committed corpus authors 32 pairs per trait and the stride
holds out half, so 16 are scored; 6 of 16 is the smallest net rate any
attainable outcome could report as significant at alpha 0.05 under the exact
test, and every plan declares exactly that 0.375. Halving a plan's declared
effect is therefore a CORPUS decision, and the refusal names the count it
would take — the same discipline
[[model-trainer-cartridge-question-set]] had to learn after a headline was
published four times below what its instrument could resolve.

`require_solo_precondition` runs BETWEEN cells. The 7B rung of the corpus
programme failed at exactly that step and not at composition — solo gain
+0.068 against a per-seed span of the same order — and every retention ratio
computed on those records became a division artefact. A composed arm measured
against a solo arm indistinguishable from noise is a ratio with no reading,
and the composed cells are where the hours are, so the refusal is raised
before them and names the untrained-prefix control beside the arm: a solo gain
that merely matches an untrained prefix is the other way this fails.

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

## What is not here

The diverse-companion, base-LoRA and crowd-invariance families are not run.
Those three are interventions that REPAIR composition, and the question they
answer is only askable once a naive interference number exists to repair —
which is the order the corpus arc was built in, each sweep importing the one
before it. Building the repair before the baseline would be the same mistake
the solo precondition exists to prevent, one level up.

The authored corpus was also invisible to git until the commit that landed
this. `services/Model-Trainer/corpus/` is ignored for downloaded bulk, and
re-including four hand-written files took three rules across two ignore files,
because git does not descend into an excluded DIRECTORY — a negation for a
file inside one is inert and reads as though it works. The suite that reads
those files to check the plans' declared effect would have passed here and
failed on any fresh checkout.
