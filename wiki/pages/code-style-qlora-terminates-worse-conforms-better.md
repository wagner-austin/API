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
  - tools/hpc3/runs/code-style-gen-v1-base.json
  - tools/hpc3/runs/code-style-gen-v1-candidate.json
  - tools/hpc3/runs/code-style-gen-v2-base.json
  - tools/hpc3/runs/code-style-gen-v2-candidate.json
  - tools/hpc3/runs/code-style-qlora-v1.json
  - tools/hpc3/runs/code-style-qlora-v2.json
source_git_blobs:
  "tools/code-style-eval/runs/gen-v2/comparison.json": 59740d9ff18ef856ec3afee9b1e4cdf86995b3eb
  "tools/code-style-eval/runs/gen-v2/base.outcomes.jsonl": ba1a470e0abb607c5d1dd523b323638952ef64fd
  "tools/code-style-eval/runs/gen-v2/candidate.outcomes.jsonl": 17dbefe2994932bbc2e7cb93991ba6c9445cef89
  "tools/code-style-eval/runs/gen-v2/base.generation.jsonl": 28d3c2096b3ee0b42d3a5fc27b6369cc591372ba
  "tools/code-style-eval/runs/gen-v2/candidate.generation.jsonl": a474a6cd72b1b10778c8edc149fd92261706244b
  "tools/code-style-eval/runs/gen-v1/base.outcomes.jsonl": 06ec61b8eec320c038f58671d1bfc3849ccf606b
  "tools/code-style-eval/runs/gen-v1/candidate.outcomes.jsonl": cefb9901a37630e030f4155456c257b1a1b5ea9e
  "libs/platform_core/src/platform_core/continuation_task.py": eaab054bfd2b59a0227261e6455a2067630af31b
  "services/Model-Trainer/src/model_trainer/cli/_test_hooks.py": bb812bd0b9d9b2945d556dba0f81ebc82af5d3ad
  "tools/code-style-eval/README.md": 0d5b561e739e680393d9aa3906e7df0143c6a92e
  "tools/code-style-eval/pyproject.toml": 461d05a24ec38163c18471a6bee77c071960e823
  "tools/hpc3/runs/code-corpus-v2-digests.txt": 24a8666ada84178a782e6b6be3e00fd1227b1f73
  "tools/code-style-eval/src/code_style_eval/core/scoring.py": 9b1db6975f058fa38977baac3897cc0c14a49619
  "libs/platform_core/src/platform_core/power_distributions.py": a8a135d23069b89fac3a4c7a3ae4c0627130a3a3
  "tools/code-style-eval/tests/test_published_comparisons.py": 9d8789a618d8e870136cb758d5059a70c4ebe678
  "tools/hpc3/runs/code-style-gen-v1-base.json": c650a9dcb602022d5bc8ab9c35e9b586fa4ce133
  "tools/hpc3/runs/code-style-gen-v1-candidate.json": 0ba923bf44663318bbcb1ba24cd52321a22949be
  "tools/hpc3/runs/code-style-gen-v2-base.json": 28ba697b4326009567c8b5bbede772bbc627d83e
  "tools/hpc3/runs/code-style-gen-v2-candidate.json": 166db0baa7199f561a7a6a58035345c0ef89becb
  "tools/hpc3/runs/code-style-qlora-v1.json": d0b87665e7742bc7f563389707bd033db5a1bc61
  "tools/hpc3/runs/code-style-qlora-v2.json": 1f9a7f16b984d546eb21e17fc6754b1ab56d08e1
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
| mypy | 54 (6.2%) | 50 (5.7%) | −4 | 0.590 |
| guards | 431 (49.3%) | 359 (41.0%) | −72 | 9.1e-05 |
| all three | 29 (3.3%) | 24 (2.7%) | −5 | 0.362 |

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
| mypy | 36 (12.8%) | 40 (14.2%) | +4 | 0.473 |

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
significance, and nothing here corrects for it. Termination at mid-p 2e-28
would survive a great deal of it; the guards gain at mid-p 0.020 on 282 items
is the result most exposed, and it is already the one flagged above as sitting
below its own MDE. Treat that number as directional.

**How big the unknown is, measured.** The correction still needs a rho nobody
has, but the *cluster structure* is a fact about the corpus and can be
counted. Over the 875 scored items[^12]:

| clustering unit | clusters *k* | mean per cluster *m* | effective *n* at ρ=0.05 | at ρ=0.30 |
|---|---|---|---|---|
| top-level category | 14 | 62.5 | 215 | 45 |
| package | 64 | 13.7 | 536 | 182 |
| containing directory | 336 | 2.6 | 810 | 591 |

Effective *n* is `mk / DE` with `DE = 1 + ρ(m−1)`, Killip's design effect[^12].
**No row is this page's answer**, and that is the point: *k* ranges from 14 to
336 purely on what one chooses to call a cluster, and nobody has chosen. Lazic
reports that an ICC of 0.30 turns a nominal 5% false-positive rate into 37%,
and ranks non-independence as more serious than the normality and
equal-variance assumptions that get checked routinely[^12].

Two limits on that table, both in the direction of it being *too kind*. Killip
states `DE = 1 + ρ(m−1)` for the special case of equal cluster sizes, and these
are severely unequal — the largest package holds 165 of the 875 files against a
median of 5 — so a mean *m* understates the design effect and every effective
*n* above is an upper bound[^12]. And Lazic's 37% is measured on a two-group
comparison of continuous data, not on McNemar; it is cited for the magnitude of
the hazard, not as a transferable rate.

So this is now quantified as a *range* rather than dismissed or corrected. The
honest reading is that the termination result at mid-p 2e-28[^3] survives every
row of that table, and the guards gain at mid-p 0.020 on 282 items[^9] does not
survive the coarser ones. That subset carries its own cluster structure rather
than the aggregate's — *k* = 14 / 53 / 186 over the same three units, effective
*n* falling to 42 at the coarsest, again before the unequal-cluster
correction[^12]. Picking the row that keeps a p-value under
0.05 is exactly the move this table exists to make visible, and it is not made
here.

**Every figure here is one realization of the decode, and there is no second
one.** `seed` is 0 in all six of this project's committed run documents, v1
and v2, both arms; no document carries a seeds list and the axis has never
been varied.[^13] The two arms *sharing* a seed is deliberate and correct —
it is what makes the decode paired — but never moving it across runs means
this project has no noise floor at all. McNemar conditions on the discordant
pairs, which handles item-level pairing; it says nothing about
decode-to-decode variance, and re-running at seed 1 would have both arms emit
different completions and therefore a different 2x2 table.

Unlike the clustering limit above, this one **cannot be bounded here**. A
range needs at least two seeds and this corpus has one, so the size of the
effect is unknown rather than estimated — stating an interval would be
inventing it. What can be said is which results are exposed: termination at
−204 items of 875 and mid-p 2e-28 would survive essentially any plausible
decode variance, while the guards gain of +18 on 282 items is now the same
result flagged for a third independent reason — below its own MDE, over
non-independent units, and from a single decode. Three reasons to read it as
directional, and the case for not calling it a result until a seed axis
exists.

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
[^13]: `tools/hpc3/runs/code-style-gen-v{1,2}-{base,candidate}.json`,
       `code-style-qlora-v{1,2}.json` — `"seed": 0` in all six, and no `seeds`
       key in any code-style run document. Counted 2026-09-09 by grepping the
       committed documents rather than by reading the pipeline.
[^12]: Cluster counts computed over the 875 shared item ids in
       `tools/code-style-eval/runs/gen-v2/{base,candidate}.outcomes.jsonl`,
       grouping by the first path segment, the first two, and the containing
       directory: k = 14 / 64 / 336, largest package 165 files, median 5.
       The same counting over the 282-item both-finished subset of [^9], which
       has its own structure rather than the aggregate's: k = 14 / 53 / 186,
       m = 20.1 / 5.3 / 1.5, effective n 42 to 275 at rho=0.30.
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
       for it; nothing here estimates one.
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
