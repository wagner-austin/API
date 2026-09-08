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
source_git_blobs:
  "tools/code-style-eval/runs/gen-v2/comparison.json": adcdf3a42c78828456790b7b9f6f5a8e17f354ba
  "tools/code-style-eval/runs/gen-v2/base.outcomes.jsonl": ba1a470e0abb607c5d1dd523b323638952ef64fd
  "tools/code-style-eval/runs/gen-v2/candidate.outcomes.jsonl": 17dbefe2994932bbc2e7cb93991ba6c9445cef89
  "tools/code-style-eval/runs/gen-v2/base.generation.jsonl": 28d3c2096b3ee0b42d3a5fc27b6369cc591372ba
  "tools/code-style-eval/runs/gen-v2/candidate.generation.jsonl": a474a6cd72b1b10778c8edc149fd92261706244b
  "tools/code-style-eval/runs/gen-v1/base.outcomes.jsonl": 06ec61b8eec320c038f58671d1bfc3849ccf606b
  "tools/code-style-eval/runs/gen-v1/candidate.outcomes.jsonl": cefb9901a37630e030f4155456c257b1a1b5ea9e
  "libs/platform_core/src/platform_core/continuation_task.py": eaab054bfd2b59a0227261e6455a2067630af31b
  "services/Model-Trainer/src/model_trainer/cli/_test_hooks.py": 3abe6387a146450ad24bd415d589ed3ed09bb744
  "tools/code-style-eval/README.md": 0d5b561e739e680393d9aa3906e7df0143c6a92e
  "tools/code-style-eval/pyproject.toml": 461d05a24ec38163c18471a6bee77c071960e823
  "tools/hpc3/runs/code-corpus-v2-digests.txt": 24a8666ada84178a782e6b6be3e00fd1227b1f73
  "tools/code-style-eval/src/code_style_eval/core/scoring.py": be9442c3047ea93eb16c955704a8e51a86c19907
provenance:
  - "trained 2026-09-07, job 55806443, A30 on hpc3-gpu-l54-09, 3731s, image digest 5dfd78a7eb14"
  - "generated 2026-09-07, jobs 55809956 (base, A30 hpc3-gpu-k54-01) and 55809960 (candidate, A30 hpc3-gpu-l54-08)"
  - "adapter sha256 e675f88218fde25a0b410bd44da29c345a6701dac92e81dcf6bb9b624427b7fd"
  - "holdout sha256 5c1697e2fe06d8a5819fdb0348031f242b6609d029c9263405337fad27709e20, verified against the digests record on pull"
  - "scored 2026-09-08 on Windows with tools/code-style-eval, ruff + mypy strict + scripts.guard"
fact_checked: 2026-09-08
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
sibling page [[code-style-guard-pass-instrument-limits]] runs into.

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
[^11]: `tools/code-style-eval/src/code_style_eval/core/scoring.py`
       `mid_p_mcnemar_p` — "THE EXACT CONDITIONAL TEST IS THE WRONG DEFAULT
       HERE ... guaranteeing the nominal level makes it overly conservative,
       so it fails to detect real differences." The tie form is separate and
       easy to get wrong: when the two discordant cells are equal the observed
       outcome sits at the centre of the distribution, so doubling a tail
       double-counts it and mid-p is `1 - 0.5*point` rather than a doubled
       tail. A hand-rolled helper that misses that case disagrees with the
       shipped CLI on exactly the ties.
