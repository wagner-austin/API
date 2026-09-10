---
title: The fine-tune learned the house style and forgot how to stop
tags: [ml, model-trainer, code-style, qlora, measurement, evaluation]
related:
  - "[[model-trainer-noise-floor-is-a-range]]"
  - "[[monorepo-discipline]]"
  - "[[code-style-guard-pass-instrument-limits]]"
source_paths:
  - tools/code-style-eval/runs/gen-v2/comparison.json
  - tools/code-style-eval/runs/gen-v2/base.outcomes.jsonl
  - tools/code-style-eval/runs/gen-v2/candidate.outcomes.jsonl
  - tools/code-style-eval/runs/gen-v2/base.generation.jsonl
  - tools/code-style-eval/runs/gen-v2/candidate.generation.jsonl
  - tools/code-style-eval/runs/gen-v1/base.outcomes.jsonl
  - tools/code-style-eval/runs/gen-v1/candidate.outcomes.jsonl
  - libs/platform_core/src/platform_core/continuation_task.py
  - services/Model-Trainer/src/model_trainer/cli/_test_hooks.py
  - tools/code-style-eval/README.md
  - tools/code-style-eval/pyproject.toml
  - tools/hpc3/runs/code-corpus-v2-digests.txt
  - tools/code-style-eval/src/code_style_eval/core/scoring.py
  - libs/platform_core/src/platform_core/power_distributions.py
  - tools/code-style-eval/tests/test_published_comparisons.py
  - services/Model-Trainer/src/model_trainer/core/services/model/continuations.py
  - tools/hpc3/runs/code-style-gen-v1-base.json
  - tools/hpc3/runs/code-style-gen-v1-candidate.json
  - tools/hpc3/runs/code-style-gen-v2-base.json
  - tools/hpc3/runs/code-style-gen-v2-candidate.json
  - tools/hpc3/runs/code-style-qlora-v1.json
  - tools/hpc3/runs/code-style-qlora-v2.json
  - libs/platform_core/src/platform_core/clustering.py
  - tools/code-style-eval/src/code_style_eval/core/clustering.py
  - tools/code-style-eval/src/code_style_eval/cli/clustering.py
  - tools/code-style-eval/src/code_style_eval/contracts/generation.py
source_git_blobs:
  "tools/code-style-eval/runs/gen-v2/comparison.json": b9daedc741c8c6506403d70cedc20021ccbd292e
  "tools/code-style-eval/runs/gen-v2/base.outcomes.jsonl": ba1a470e0abb607c5d1dd523b323638952ef64fd
  "tools/code-style-eval/runs/gen-v2/candidate.outcomes.jsonl": 17dbefe2994932bbc2e7cb93991ba6c9445cef89
  "tools/code-style-eval/runs/gen-v2/base.generation.jsonl": 28d3c2096b3ee0b42d3a5fc27b6369cc591372ba
  "tools/code-style-eval/runs/gen-v2/candidate.generation.jsonl": a474a6cd72b1b10778c8edc149fd92261706244b
  "tools/code-style-eval/runs/gen-v1/base.outcomes.jsonl": 06ec61b8eec320c038f58671d1bfc3849ccf606b
  "tools/code-style-eval/runs/gen-v1/candidate.outcomes.jsonl": cefb9901a37630e030f4155456c257b1a1b5ea9e
  "libs/platform_core/src/platform_core/continuation_task.py": eaab054bfd2b59a0227261e6455a2067630af31b
  "services/Model-Trainer/src/model_trainer/cli/_test_hooks.py": bb812bd0b9d9b2945d556dba0f81ebc82af5d3ad
  "tools/code-style-eval/README.md": 0d5b561e739e680393d9aa3906e7df0143c6a92e
  "tools/code-style-eval/pyproject.toml": 110037c5f0646c8fb6f44ae30ea2112767bffb30
  "tools/hpc3/runs/code-corpus-v2-digests.txt": 24a8666ada84178a782e6b6be3e00fd1227b1f73
  "tools/code-style-eval/src/code_style_eval/core/scoring.py": 9b1db6975f058fa38977baac3897cc0c14a49619
  "libs/platform_core/src/platform_core/power_distributions.py": 3ce9c397f4fba4f1b71fc4757a3ef4da077fae56
  "tools/code-style-eval/tests/test_published_comparisons.py": 9d8789a618d8e870136cb758d5059a70c4ebe678
  "services/Model-Trainer/src/model_trainer/core/services/model/continuations.py": 963041d82664195e4aa7fd6d69a54f64c1ebcb33
  "tools/hpc3/runs/code-style-gen-v1-base.json": c650a9dcb602022d5bc8ab9c35e9b586fa4ce133
  "tools/hpc3/runs/code-style-gen-v1-candidate.json": 0ba923bf44663318bbcb1ba24cd52321a22949be
  "tools/hpc3/runs/code-style-gen-v2-base.json": 28ba697b4326009567c8b5bbede772bbc627d83e
  "tools/hpc3/runs/code-style-gen-v2-candidate.json": 166db0baa7199f561a7a6a58035345c0ef89becb
  "tools/hpc3/runs/code-style-qlora-v1.json": d0b87665e7742bc7f563389707bd033db5a1bc61
  "tools/hpc3/runs/code-style-qlora-v2.json": 1f9a7f16b984d546eb21e17fc6754b1ab56d08e1
  "libs/platform_core/src/platform_core/clustering.py": 7611039dbbc21a817e4c10027e282a4e21e9fdbe
  "tools/code-style-eval/src/code_style_eval/core/clustering.py": 831f60a8e23af61ea78482287b4dadc6b6a972e3
  "tools/code-style-eval/src/code_style_eval/cli/clustering.py": 0e5c0f9a2ed85a28523bb6660743af36b0221e11
  "tools/code-style-eval/src/code_style_eval/contracts/generation.py": c1e757eba479597c84fb1e98e64fe2b0b7156e54
provenance:
  - "trained 2026-09-07, job 55806443, A30 on hpc3-gpu-l54-09, 3731s, image digest 5dfd78a7eb14"
  - "generated 2026-09-07, jobs 55809956 (base, A30 hpc3-gpu-k54-01) and 55809960 (candidate, A30 hpc3-gpu-l54-08)"
  - "adapter sha256 e675f88218fde25a0b410bd44da29c345a6701dac92e81dcf6bb9b624427b7fd"
  - "holdout sha256 5c1697e2fe06d8a5819fdb0348031f242b6609d029c9263405337fad27709e20, verified against the digests record on pull"
  - "scored 2026-09-08 on Windows with tools/code-style-eval, ruff + mypy strict + scripts.guard"
  - "THE RUNS THIS PAGE REPORTS, BY DIGEST rather than by directory: gen-v2 (the 875-item aggregate and both splits) payload_digest 0e3e9080fafbe2c46057c27a01298fd84002518e441bde50e076d26f1e9ddb9f; gen-v1 (the replication figures) payload_digest face4e797b8ac150fbaa56516232c92183807daa3d3cad3a5f4147f5b679b8ec. Name-paired sha256 over each run's two *.outcomes.jsonl, carried inside its comparison.json. A run name is not an identity -- gen-v1 and sweep-v1 both scored 226 items and agree on nothing else -- and quoting a figure by run name is how it gets matched back to the wrong artifact."
  - "cluster structure counted 2026-09-09 over the 875 shared item ids in runs/gen-v2/{base,candidate}.outcomes.jsonl: k = 14 / 64 / 336 by path segment, largest package 165, median 5"
  - "tech-wiki/sources/killip-2004-intracluster-correlation.txt sha256 2fae776213c2ab342d4380f8b5dabb92ed66e487d001e3e9b33bd9a18400877e -- outside this wiki's workspaceRoot; the design-effect formula and its equal-cluster-size limit, read from the archived text"
  - "tech-wiki/sources/lazic-2010-pseudoreplication-neuroscience.txt sha256 57643fecd2130189bffc9a8ea105de68dfc927a6ac24b357ba6d110a566bc0af -- outside this wiki's workspaceRoot; the IC=0.30 -> 37% figure, read from the archived text"
fact_checked: 2026-09-09
confidence: high
hubs: [services]
---

# The fine-tune learned the house style and forgot how to stop

A QLoRA adapter over Qwen2.5-Coder-1.5B, trained on 3,996 files of this
monorepo, was measured against its own base on 875 held-out files.[^8] Read in
aggregate it degraded everything. Split on whether the model finished the file,
it did the opposite of that on the half it finished.

Both arms decode the same items, same seed, same A30 card model, and differ in
exactly one thing: `candidate` reattaches the adapter, `base` loads the weights
the adapter was trained *from* and attaches nothing.[^1]

## Aggregate: everything is worse

Paired McNemar over all 875 items.[^2]

| checker | base | candidate | net | mid-p |
|---|---|---|---|---|
| ruff | 134 (15.3%) | 78 (8.9%) | −56 | 3.5e-07 |
| mypy | 54 (6.2%) | 50 (5.7%) | −4 † | 0.590 |
| guards | 431 (49.3%) | 359 (41.0%) | −72 | 9.1e-05 |
| all three | 29 (3.3%) | 24 (2.7%) | −5 ‡ | 0.362 |

**† A net this small could not have been significant under any arrangement,
and ‡ only under one.**[^14] A paired test's best case for a net of *k* is *k*
discordant pairs all falling one way, giving mid-p `0.5^k`. That is 0.0625 at
net 4 — above α — so the −4 and +4 rows above were never capable of a
significant result whatever the data did, and their p-values carry no
information about the adapter. At net 5 (‡) it is 0.03125, reachable only on a
perfect 5:0 split. This is a **different floor** from the discordant-count one
in "What each null could have detected": that one asks whether a given *d*
could ever reject, this one asks whether a given *net* could ever be
significant, and a row can pass the first and fail the second when *d* is
large and the split near even. The mypy row does exactly that — *d*=54, so its
own floor is effectively zero, and its net still cannot clear α.[^14]

The combined row is the least informative number here.[^2] It is the AND of three
checkers at a 3% floor, it is null, and reading only that would report "no
effect" on a run with two effects in it.

## Split on termination: the sign flips

The candidate finishes 360 of 875 completions on its own against the base's
564 — a 23.4-point collapse.[^3] A file cut off at the 1536-token cap fails
ruff on syntax alone, so the aggregate above cannot separate *style* from
*stopping*. Splitting it does.

**Both arms finished naturally, n=282:**[^9]

| checker | base | candidate | net | mid-p |
|---|---|---|---|---|
| guards | 203 (72.0%) | 221 (78.4%) | **+18** | **0.020** |
| ruff | 82 (29.1%) | 66 (23.4%) | −16 | 0.046 |
| mypy | 36 (12.8%) | 40 (14.2%) | +4 † | 0.473 |

**At least one arm truncated, n=593:**[^9]

| checker | base | candidate | net | mid-p |
|---|---|---|---|---|
| guards | 228 (38.5%) | 138 (23.3%) | −90 | 6.1e-08 |
| ruff | 52 (8.8%) | 12 (2.0%) | −40 | 9.6e-08 |

On files it finishes, the adapter passes **more** of this repo's guards than
the base model does.[^9] The collapse lives entirely in the truncated set.

## Truncation is the model's failure, not the harness's

`finishable` admits only items whose real continuation fits the token budget,
which is why 875 of 1674 prompts were scored.[^4] Every scored item had room to
finish. Its docstring refuses the obvious objection directly: excluding
over-budget items "is a stated limit on scope, not a way to flatter the result
... a model that rambles past the budget on an item that fits still fails that
item, which is a fact about the model and stays in."[^4]

So the 515 truncated candidate completions are files it had room to finish and
did not.[^3]

## What this does not establish

**"Both finished" conditions on a post-treatment variable.**[^3] Termination is
itself changed by the fine-tune, so those 282 items are not a random subset and
conditioning on them can manufacture an effect. The +18 on guards is strong
evidence, not a clean randomised contrast. What survives without that caveat is
narrower and still decisive: the degradation is *concentrated* in truncated
items rather than spread across all of them, so any reading of the aggregate as
"the fine-tune degrades code quality" is wrong on the evidence.

The ruff regression replicates across corpora — v1 net −15 (mid-p 0.009), v2
net −56 (mid-p 3.5e-07), −6.6pp against −6.4pp.[^5] The guards *gain* has no
second sample: v1 was +6 and null, and that null is uninformative rather than
contradictory — see below.

**The unit of *n* is the file, and files are not independent draws.** Every
count here is over held-out files from two repositories, and McNemar assumes
the pairs are independent of each other. These are not: files in one package
share an author, a layout and often a near-identical shape, and the emitter
deduplicates only byte-identical ones.[^8] Correlation between items inflates
significance, and **no p-value on this page is corrected for it** — the
correction is reported separately, below, rather than folded into the figures,
because it requires choosing a clustering unit and this corpus does not settle
which one. Termination at mid-p 2e-28 would survive a great deal of it; the
guards gain at mid-p 0.020 on 282 items is the result most exposed, and it is
already the one flagged above as sitting below its own MDE. Treat that number
as directional.

**How big the correction is — measured, not assumed.** An earlier version of
this section said the correction "still needs a rho nobody has" and printed a
table at ρ = 0.05 and ρ = 0.30 as hypotheticals. That was wrong about what was
knowable: ρ is estimable from the two outcomes files this page already pins,
and the estimate is now computed by a shipped instrument rather than
guessed[^17]. Over the 875 scored items[^12]:

| clustering unit | clusters *k* | *m₀* | largest | ρ of *d* | DE | effective *n* |
|---|---|---|---|---|---|---|
| top-level category | 14 | 51.32 | 265 | −0.0098 | 1.000 | 875.0 |
| package | 64 | 13.02 | 165 | +0.0619 | 1.744 | 501.7 |
| containing directory | 336 | 2.59 | 26 | +0.0585 | 1.093 | 800.3 |

And over the 282-item both-finished subset[^9], on the guards outcome — the
row this page flags as most exposed:

| clustering unit | clusters *k* | *m₀* | largest | ρ of *d* | DE | effective *n* |
|---|---|---|---|---|---|---|
| top-level category | 14 | 16.45 | 81 | +0.0442 | 1.683 | 167.6 |
| package | 53 | 5.14 | 45 | −0.0364 | 1.000 | 282.0 |
| containing directory | 186 | 1.51 | 7 | +0.0279 | 1.014 | 278.0 |

`DE = 1 + ρ(m₀−1)`, Killip's design effect[^12], with *m₀* the unequal-cluster
average rather than a plain mean — these clusters are severely unequal, and a
plain mean would have overstated every design effect here[^17]. **A DE of
exactly 1.000 beside a negative ρ is a floor, not a corpus that landed there**:
one-way ICC goes negative when within-cluster spread exceeds between-cluster
spread, and taken literally that yields an effective *n* larger than the
sample. Clustering cannot manufacture information, so the instrument floors DE
at 1 and publishes ρ unfloored beside it[^17].

**ρ is measured on the paired difference, and that is not a detail.** The
correlated quantity for a paired test is `d = candidate − baseline` per item,
in {−1, 0, +1} — not either arm's raw pass indicator. Measured both ways on the
same clusters, they disagree by up to five times and, at the aggregate, in
sign: guards at the directory unit is +0.028 on *d* against +0.141 on the raw
indicator, and the aggregate is +0.059 against −0.031[^17]. Either raw figure
correctly answers "do these files pass or fail together", which is not the
question McNemar asks. A correction built from the nearer-to-hand series would
have deflated one row five times too much and corrected another backwards.

**No row is still this page's answer**, and *that* part stands: *k* ranges from
14 to 336 purely on what one calls a cluster, and nobody has established which
level the correlation lives at. Lazic reports that an ICC of 0.30 turns a
nominal 5% false-positive rate into 37%, and ranks non-independence as more
serious than the normality and equal-variance assumptions that get checked
routinely[^12] — measured here, **no unit reaches ρ = 0.07**, so that figure
describes a hazard this corpus does not have. It is cited for why the question
was worth asking, not as a rate that transfers, and Lazic's is a two-group
comparison of continuous data rather than a McNemar table in any case.

**What survives.** Termination at mid-p 2e-28[^3] survives every row. The
guards gain at mid-p 0.020 on 282 items[^9] survives the package unit (ρ
negative, so no correction at all) and the containing-directory unit, where it
survives *robustly*: DE 1.014 deflates the table to (59.17, 17.75), and all six
parity-valid roundings of that reject, from 0.014 to 0.040[^17]. **It does not
survive the top-level-category unit**, and the
measurement changes only the *reason*, not the verdict: the earlier version of
this section reached the same conclusion from ρ = 0.30, which this corpus does
not have, and ρ = 0.044 gets there on its own because *m₀* at that unit is
16.45.

The way it fails is worth stating precisely, because a first pass at it came
out the other way. Deflating that stratum's discordant table (21, 39) by
DE = 1.683 gives *d* = 35.65 with a net of 10.70 — **not an integer table**,
and a McNemar table needs *d* and the net to share a parity. Of the eight
parity-valid roundings around it, only the two that round the net UP to 12,
past its own estimate of 10.70, reject[^17]:

| table (*d*, net) | minority | mid-p |
|---|---|---|
| (34, 12) | 11 | 0.0410 |
| (36, 12) | 12 | 0.0470 |
| (35, 11) | 12 | 0.0652 |
| (37, 11) | 13 | 0.0730 |
| (34, 10) | 12 | 0.0895 |
| (36, 10) | 13 | 0.0989 |
| (35, 9) | 13 | 0.1325 |
| (37, 9) | 14 | 0.1433 |

Every rounding that respects the net estimate lands between 0.0652 and 0.1433.[^17]
**A first draft of this paragraph reported 0.047 alone and called the result
survived** — it was corrected before this page was committed, but only because
the neighbourhood was enumerated rather than trusted. That draft had taken the
rounding a language default happened to produce, which is the same move as
picking the clustering unit that keeps a p-value under 0.05, one level further
down and much harder to see: choosing a unit at least looks like a choice.

Contrast the directory unit above, where the same enumeration is what licenses
calling the result survived: all six of its roundings reject. **The
enumeration is what distinguishes the two cases**, and neither conclusion
would be trustworthy without it.[^17]

That instability is itself the finding. When the answer depends on how a
non-integer table is rounded, the *effective-sample-size shortcut* — deflate
by DE, re-run the exact test — is being pushed past where it means anything.
It is an approximation: a properly clustered McNemar is a different statistic
rather than a rescaling of this one, which is why the shipped CLI reports the
design effect and deliberately does not rewrite the p-value beside it[^17].
The correct reading is not "0.047" and not "0.099" but that **at the coarsest
defensible clustering unit this instrument cannot resolve the guards gain**,
which is the same place the MDE analysis above already put it.

**The decode is deterministic, measured across three seeds and three nodes.**
An earlier version of this section said the opposite -- that every figure was
one realization and re-running at another seed would emit different
completions and a different 2x2 table. That was reasoning from the specs,
where `seed` is 0 in all six committed run documents.[^13] It is wrong, and
the seed axis run to test it is what showed so.[^15]

Three base runs at seeds 0, 1 and 2 -- same adapter, same holdout, same card
model, differing in `seed` and nothing else -- produced BYTE-IDENTICAL
finished-sets: payload digest `sha256:db1795e0...` and 564 of 875 completions
finished, in all three. They ran on three different nodes (k54-01, l54-07,
l54-08) on different days over a preemptible partition.[^15]

The mechanism is in the generator, not in the run documents. Continuation
decoding passes `do_sample=False` -- its own docstring says "Whether to
sample. Always False here" -- so there is no sampling for a seed to control.
`torch.manual_seed` IS called, once per batch, and the function says why: a
batch's result stays independent of how many batches preceded it, so a run
resumed after preemption reproduces what it redoes rather than replacing it
plausibly. The seed is a determinism GUARANTEE, not a variance axis.[^15]

So there is no decode variance to bound[^15], and the old caveat overstated
the uncertainty rather than understating it. **What this does not rescue is
the clustering limit above**, which is about correlation BETWEEN ITEMS and is
untouched by any amount of run-to-run determinism[^12]. The guards gain of +18
on 282 items therefore stands flagged for two reasons, not three: below its
own MDE[^14], and over non-independent units[^12].

And this instrument still has **no declared smallest effect of interest**, so
nothing here says whether an effect it can resolve is one anyone would act on.
Every power record this package emits is a falsifiability predicate --
`can_ever_reject`, `net_could_ever_be_significant` -- and neither instrument
takes a threshold, so neither can return a `PowerVerdict`.[^16]

**There is no SEI *yet*, and the rule that would create one is stateable.** A
noise floor was going to anchor the number without anyone choosing it; the
floor is exactly zero[^15], so `SEI >= 0` constrains nothing. Nor can one be
read off a decision this project took, the way a campaign that ADOPTS an arm
can cite the smallest gain it ever adopted on: code-style has promoted
nothing and put no adapter into use. What remains is legitimate and is not
derivation — a **pre-registered** rule, of the form "an adapter is worth
using if it improves guard-pass on the finishable stratum by at least X",
stated and dated BEFORE a result exists. That still chooses X. What it cannot
do is choose X to suit a number already seen, which is the failure mode this
page has refused twice. Until such a rule is written down, every verdict here
is falsifiability and none is actionability.

## What each null could have detected

A McNemar test conditions on the discordant pairs, so its power is a property
of the discordant count, not of n. Every null on this page therefore carries the
smallest effect an exact two-sided test at α=0.05 would have caught with 80%
power, given the discordant count actually observed.[^10]

Falsifiability comes first, because it outranks power: with *d* discordant
pairs the most extreme attainable outcome is *d*:0, so the test has a floor —
the smallest p any data could produce. If that floor exceeds α, no possible
outcome could have rejected and the null is unfalsifiable rather than merely
underpowered, which is strictly worse and invisible to an MDE.[^10]

Both columns below are computed against **mid-p**, because that is what this
package reports and it says why: the exact conditional test "did not perform
well for any of the considered scenarios" (Fagerland, Lydersen and Laake 2013)
and its conservativeness is largest exactly at small discordant counts.[^11]
An MDE computed against the exact test's rejection region would describe a
test nobody runs.

| null | observed | discordant | mid-p floor | MDE (80% power) |
|---|---|---|---|---|
| v2 combined | −0.57pp | 29 | 1.9e-09 | 1.59pp |
| v2 mypy | −0.46pp | 54 | 5.6e-17 | 2.36pp |
| v1 combined | −0.44pp | 7 | **0.0078** | 2.91pp |
| v1 mypy | +0.44pp | 15 | 3.1e-05 | 4.55pp |
| **v1 guards** | **+2.65pp** | **90** | 8.1e-28 | **11.79pp** |

All five are falsifiable at α=0.05. `v1 combined` is the marginal case: with 7
discordant pairs it rejects **only** on a perfect 7:0 split, one item away from
a test no outcome could have failed. The exact test's floor is twice the mid-p
floor, and that gap is not academic — at *d*=5 it is the difference between
"rejects only at 5:0" and "cannot reject at all", which is the condition the
sibling page [[code-style-guard-pass-instrument-limits]] runs into.[^10]

**v1's guards null could not have found the v2 effect.**[^10] Its MDE is 11.79pp
against a v2 effect of 6.38pp, so v1 was structurally incapable of detecting
what v2 measured. Reading it as a failure to replicate would be reading an
instrument's floor as a fact about the model.

The same arithmetic cuts the other way on the headline.[^10] The guards gain on the
both-finished subset is significant (mid-p 0.020) but its MDE is 7.49pp against
an observed 6.38pp — the test had **under** 80% power for the effect it found.
A significant result below its own MDE is the regime where point estimates are
inflated, so +6.4pp should be read as "positive, magnitude uncertain" rather
than as an estimate to plan against.

## The metric is this repo's own checkers

House style has no external benchmark, so `code-style-eval` uses ruff, mypy
strict and `scripts.guard` — "a completion passes if the tools the operator
already runs would accept it."[^6] The package ships an optional `corpus`
dependency group whose only job is letting mypy resolve what a generated file
imports, "instead of reporting a missing stub — a verdict about the scoring
sandbox rather than about the generated code."[^7] That group had drifted out
of the scoring venv before this run; scoring without restoring it would have
measured the sandbox.

[^1]: `services/Model-Trainer/src/model_trainer/cli/_test_hooks.py`
      `_default_load_continuation_arm` — "``candidate`` reattaches the adapter;
      ``base`` loads the weights it was trained against and attaches nothing."
      Both specs name the same `artifact_path`; the arm field is the only
      difference, which is what makes the pair a pair.
[^2]: `tools/code-style-eval/runs/gen-v2/comparison.json` [synthesis] — the
      combined row is the committed record verbatim (shared_items 875,
      net_improvement −5, mid_p 0.3616). Per-checker rows recompute from
      `base.outcomes.jsonl` and `candidate.outcomes.jsonl`, one JSON object per
      item carrying a boolean per checker.
[^3]: `tools/code-style-eval/runs/gen-v2/base.generation.jsonl` and
      `candidate.generation.jsonl` [synthesis] — one row per item,
      `{"item_id": ..., "finished": bool}`; 564 and 360 true respectively.
[^4]: `libs/platform_core/src/platform_core/continuation_task.py` `finishable`
      — "An item whose reference is longer than the budget CANNOT be completed
      ... a model that rambles past the budget on an item that fits still fails
      that item, which is a fact about the model and stays in."
[^5]: `tools/code-style-eval/runs/gen-v1/base.outcomes.jsonl` and
      `candidate.outcomes.jsonl` [synthesis] — recomputed the same way. v1 is
      NOT citable as a result: its corpus was emitted from two dirty
      repositories, recorded in `tools/hpc3/runs/code-corpus-v2-digests.txt`.
      It is used here only as a second sample of the same direction.
[^6]: `tools/code-style-eval/README.md` § What it measures — "a completion
      passes if the tools the operator already runs would accept it."
[^7]: `tools/code-style-eval/pyproject.toml` `[tool.poetry.group.corpus]` —
      "They exist so mypy can resolve what a generated file imports, instead of
      reporting a missing stub -- a verdict about the scoring sandbox rather
      than about the generated code."
[^8]: `tools/hpc3/runs/code-corpus-v2-digests.txt` § emitter — "kept 5,791 files
      -- 3,996 train, 1,795 holdout", both source repositories clean at
      emission (api 59beb485, mcp 782dd897). 875 of the holdout's 1,674 built
      prompts were in scope; see [^4] for why the rest were not.
[^9]: `tools/code-style-eval/runs/gen-v2/{base,candidate}.outcomes.jsonl` joined
      to `{base,candidate}.generation.jsonl` on `item_id` [synthesis] — the
      split is `finished == true` in BOTH manifests (n=282) against its
      complement (n=593); per-checker counts and the paired McNemar recompute
      from the outcome booleans over each subset. Both files are blob-pinned
      above, so the tables are re-derivable rather than assertions.
[^10]: `tools/code-style-eval/runs/gen-v2/base.outcomes.jsonl` and
       `candidate.outcomes.jsonl`, joined on `item_id`; v1 rows from
       `runs/gen-v1/base.outcomes.jsonl` and `candidate.outcomes.jsonl`
       [synthesis] — every discordant count in the table is the number of
       items whose `all_passed` (or the named checker's `passed`) differs
       between the two arms, read from the `checks` array of each row. The MDE
       is then arithmetic on that count alone: exact two-sided McNemar at
       α=0.05 conditions on the discordant pairs and tests them against
       Binomial(n, 0.5), so the rejection region is the binomial tail and the
       MDE is the smallest split reaching 80% power against it. No data beyond
       the pinned records is used.
[^16]: `tools/code-style-eval/runs/gen-v2/comparison.json` fields `power` and
       `net_power`, emitted by
       `platform_core.minimum_detectable_effect.mcnemar_power` and
       `net_difference_power`. Both return falsifiability predicates only;
       `PowerVerdict` (TESTED / NOT_TESTED) is set only by instruments that
       are HANDED THE EFFECT WORTH ACTING ON and can compare something to it,
       and this package calls none of them. **The criterion is the durable
       form and the list is not**, which this footnote has now demonstrated
       on itself: at 2026-09-09T23:55Z there were THREE setters
       (`paired_continuous_power`, `zero_failure_power`, `rate_floor_power`);
       at 2026-09-10T05:11Z there are **TWO**, because `d2bb1e4d` replaced
       `rate_floor_power`'s verdict with `rate_significantly_exceeds_floor`
       and `design_can_clear_floor` -- two booleans named after their own
       questions. The list decayed inside four hours, with nothing in this
       page able to notice; the criterion above did not move at all.
       Re-derive rather than trust either:
       `git grep -n "verdict = " -- libs/platform_core/src/platform_core/minimum_detectable_effect.py`.
       What does NOT move is the criterion, and it is the criterion this
       footnote rests on. So the
       library never offers a TESTED it has not earned, and the absence of one
       here is the instrument reporting its own limit rather than a gap in
       this page. An earlier version of this footnote listed
       `required_replicates` as a fourth setter. It sets none, and its record
       has no `verdict` field. The list is now counted by reading every
       assignment in the module at HEAD rather than by reading docstrings,
       which is how the wrong one got in: that function's docstring says it
       "answers the question a `NOT_TESTED` verdict raises", and mentioning a
       verdict is not setting one.
[^15]: SIX runs, both arms, seeds 0/1/2. New jobs 55877275
       (`gen-v2-base-s1`, 7998s, l54-07), 55877509 (`gen-v2-base-s2`, 7879s,
       l54-08), 55877370 (`gen-v2-candidate-s1`, 9227s) and 55877841
       (`gen-v2-candidate-s2`, 9555s), against the existing 55809956
       (`gen-v2-base`, 6297s, k54-01) and 55809960 (`gen-v2-candidate`,
       8191s, l54-08). Digests, truncated to 16:
       base `db1795e0bd114f9b` three times, 564 finished each;
       candidate `0fbcd792fd871772` three times, 360 finished each.
       IDENTICAL WITHIN EACH ARM AND DIFFERENT BETWEEN THEM -- the second
       half is the control, and without it identical digests would also be
       what a digest insensitive to everything looks like.
       Records at
       `/pub/wagnera3/code-style/results/gen-v2-{base,candidate}{,-s1,-s2}.json`; the
       `payload_digest` is a digest over which items finished, so it moves if
       the decode moves. Specs staged and certified via
       `tools/hpc3/runs/code-style-specs-v2-seeds-stage.json`, each differing
       from its own arm's seed-0 spec in exactly `seed` and `label`, verified
       before staging. Mechanism read from
       `services/Model-Trainer/src/model_trainer/core/services/model/continuations.py`
       at HEAD -- `do_sample=False` and the per-batch `torch.manual_seed`
       rationale in the same function's docstring. The acceptance for this
       reading was posted to board task bc307caa at 19:34Z, BEFORE the jobs
       finished: three distinct digests would have meant the seed varied the
       decode, any two identical meant stop and compute no floor.
[^14]: Discordant counts recomputed per checker over
       `runs/gen-v2/{base,candidate}.outcomes.jsonl` joined on `item_id`:
       aggregate mypy 54, all-three 29; both-finished mypy 30, all-three 24;
       truncated all-three 5. Best-case values from
       `platform_core.minimum_detectable_effect.net_difference_power`, with
       `smallest_resolvable_net_difference(0.05, MID_P)` returning 5
       — 0.25 at net 2, 0.125 at 3, 0.0625 at 4, 0.03125 at 5.
       THIS FOOTNOTE FIRST DERIVED THOSE AS `mcnemar_p(0, net, MID_P)`,
       which assumes the best case sits at *d* = net. That assumption is
       false — the margin needed to reject is not monotone in *d*, it
       oscillates with parity, and at net 0 the mid-p tie form dips to 0.75
       at *d*=2 before climbing back. The four values above are unaffected
       (the exception bites only at net 0), but two of this package's six
       committed comparisons ARE at net 0, where the hand derivation returns
       1.0 and the shipped function returns 0.75. Corrected to consume the
       function rather than restate its arithmetic. Four of this
       page's twelve per-checker rows fail this floor (aggregate mypy,
       both-finished mypy, both-finished all-three, truncated all-three); the
       eight that pass include every result this page argues from. The check is
       @opus-weight-injection-0902's, board 2026-09-09, who stated it at 6
       items for the EXACT test; under the mid-p this page reports the
       threshold is 5.
[^13]: `tools/hpc3/runs/code-style-gen-v1-base.json` key `seed`,
       and the same key in `code-style-gen-v1-candidate.json`,
       `code-style-gen-v2-base.json`, `code-style-gen-v2-candidate.json`,
       `code-style-qlora-v1.json` and `code-style-qlora-v2.json` — the value
       is `0` in all six. `tools/hpc3/runs/code-style-run-gen-v2-base.json`
       section `experiment` records the same value as `"seed": "0"` beside
       the card and decoding settings. No `seeds` key occurs in any
       `code-style-*.json` run document. Counted 2026-09-09 over the
       committed documents at commit `93d0bf6f`, by reading the documents
       rather than the pipeline that consumes them.
[^12]: Cluster counts computed over the 875 shared item ids in
       `tools/code-style-eval/runs/gen-v2/{base,candidate}.outcomes.jsonl`,
       grouping by the first path segment, the first two, and the containing
       directory: k = 14 / 64 / 336, largest package 165 files, median 5.
       The same counting over the 282-item both-finished subset of [^9], which
       has its own structure rather than the aggregate's: k = 14 / 53 / 186.
       Both tables' m0, rho and DE columns come from [^17], not from here;
       what this note carries is the corpus structure and the two papers.
       The design effect is Killip, Mahfoud & Pearce 2004 (Ann Fam Med, doi
       10.1370/afm.141), archived at `tech-wiki/sources/`
       `killip-2004-intracluster-correlation.txt` sha256
       2fae776213c2ab342d4380f8b5dabb92ed66e487d001e3e9b33bd9a18400877e,
       "EFFECTIVE SAMPLE SIZE AND THE DESIGN EFFECT" -- "DE = 1 + ρ (m-1),
       where m = number of subjects in a cluster, k = number of clusters, mk =
       total number of subjects in a clustered study, ESS = effective sample
       size". The equal-size limit is that section's own: it derives the
       formula for "the special case of clustered data with all groups having
       the same number of subjects". The 37% figure and the ranking against
       normality are Lazic 2010 (BMC Neuroscience, doi 10.1186/1471-2202-11-5),
       `lazic-2010-pseudoreplication-neuroscience.txt` sha256
       57643fecd2130189bffc9a8ea105de68dfc927a6ac24b357ba6d110a566bc0af --
       "a two independent group comparison with n = 10 in each group and with
       a modest within group correlation of IC = 0.30 would give an a
       probability of 0.37; in other words, 37% of the time (and not 5%) the
       null hypotheses would be (erroneously) rejected", and violating
       independence "can be more serious than violating the normality or equal
       variances assumption". Both read from the archived text, not from a
       summary. Neither paper is about this corpus and neither supplies a ρ
       for it — [^17] estimates one from the corpus itself.
[^17]: ρ, m₀, DE and the effective *n* in both tables are emitted by
       `code-style-eval-clustering`, run against the two blob-pinned outcomes
       files and, for the stratum table, the two generation manifests. Two
       commands, from `tools/code-style-eval`, each producing one table above
       transposed into markdown. THE AGGREGATE:
       `poetry run python -m code_style_eval.cli.clustering --baseline
       runs/gen-v2/base.outcomes.jsonl --candidate
       runs/gen-v2/candidate.outcomes.jsonl --checker all`. THE STRATUM: the
       same with `--checker guards --baseline-generation
       runs/gen-v2/base.generation.jsonl --candidate-generation
       runs/gen-v2/candidate.generation.jsonl`. Both manifests or
       neither: the CLI refuses one alone, because restricting by a single
       arm's truncations yields a stratum that is not the both-finished one
       and cannot be told apart from it afterwards.
       THE ARITHMETIC is
       `libs/platform_core/src/platform_core/clustering.py` —
       `intracluster_correlation` (one-way random-effects ICC),
       `average_cluster_size` (Killip's m₀ for unequal clusters, which is
       strictly below the arithmetic mean whenever sizes vary) and
       `clustered_paired_power`. Its tests check the ICC against the ANOVA
       definition rewritten independently in the test file rather than against
       the implementation's own output.
       THE FLOOR: `design_effect` is `max(1, 1 + ρ(m₀−1))` and
       `effective_sample_size` can never exceed `total_units`, while
       `intracluster_correlation` is published unfloored. The first draft of
       this arithmetic, run as a throwaway script, printed an effective *n* of
       1718.6 for 875 items; the module's docstring records that as the
       motivating defect and a test pins it.
       THE SERIES: `code_style_eval.core.clustering.paired_differences`
       returns `candidate − baseline` per item and is the only producer of a
       series here. The raw-indicator comparison quoted above was computed by
       substituting the candidate's own pass indicator for that series over
       the same clusters. The record carries NO `PowerVerdict`, joining
       `RequiredReplicates`, `McNemarPower` and `NetDifferencePower`, for the
       reason in [^16]. Named rather than counted, so the sentence stays true
       as that module gains and loses verdicts.
       THE EIGHT ROUNDINGS: each row is
       `mcnemar_p((d − net)/2, d, McNemarTest.MID_P)` over every (d, net) of
       matching parity with d in 34..37 and net in 9..12, the integer
       neighbourhood of the deflated (35.65, 10.70). Quoted to four decimals
       rather than three because (35, 9) lands on 0.1325, exactly a
       half-way case: a first version of the table hand-rounded it to 0.133,
       which is a rounding decision taken by the author inside a table whose
       entire subject is rounding decisions taken by the author. Every row is
       now the function's own output at a precision that hides none. The
       directory unit's
       claim of robust survival is the same enumeration over the
       neighbourhood of its own deflated table (59.17, 17.75) — d in 58..60,
       net in 16..19 — giving six valid tables at 0.0135, 0.0183, 0.0204,
       0.0273, 0.0363 and 0.0396, all below α. Computed at commit
       `7c05c2ca` against `platform_core.power_distributions`, the same
       function every other p on this page comes from. The CLI does not emit
       any of them: this is the effective-sample-size shortcut, stated as an
       approximation in the CLI's own module docstring, done deliberately here
       rather than published by the instrument. The (36, 12) row is the one a
       first draft of this section reported alone; Python's `round(12.5)`
       returns 12 rather than 13, and that is the whole reason it was the row
       that surfaced.
       THE STRATUM MANIFEST now has a decoder,
       `code_style_eval.contracts.generation`, which refuses a row with a
       missing `finished` field or an empty `item_id`. Until 2026-09-09 it was
       read with an inline `json.loads`, so a stratum four published tables
       rest on had no validation between the file and the figure: an absent
       `finished` would have read as falsy and moved the item out silently.
[^11]: The choice of variant and the arithmetic behind it now sit in two
       places, because the statistic moved out of this package on 2026-09-09
       (commit `63770146`) and citing the old home would be citing a symbol
       that no longer exists.
       WHY MID-P: `tools/code-style-eval/src/code_style_eval/core/scoring.py`
       module docstring — "Fagerland, Lydersen and Laake measured type I error
       and power over 9,595 scenarios and found the exact conditional test
       overly conservative in all of them, while the mid-p test never violated
       the nominal level and was almost as powerful as the asymptotic test."
       THE ARITHMETIC:
       `libs/platform_core/src/platform_core/power_distributions.py`
       `mid_p_mcnemar_p`, which owns both variants for the monorepo and
       documents the trap in its own words — "THE TIE NEEDS ITS OWN FORM: when
       the two discordant cells are equal the observation sits at the centre of
       the distribution, so doubling a tail double-counts it. A hand-rolled
       version that doubled anyway returned 1.0 at a 3:3 split instead of
       0.84375, agreed with this one on every unequal split, and was trusted
       for exactly that reason." That the move changed no number on this page
       is held by `tools/code-style-eval/tests/test_published_comparisons.py`,
       which rebuilds all six committed comparisons from their own outcome
       files rather than from a fixture.
