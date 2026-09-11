---
title: Guard-pass as a fine-tune metric — what it can and cannot measure
tags: [ml, evaluation, measurement, code-style-eval, provenance, statistics]
related:
  - "[[monorepo-discipline]]"
  - "[[determinism-env-read-once-at-library-load]]"
  - "[[code-style-qlora-terminates-worse-conforms-better]]"
source_paths:
  - tools/code-style-eval/src/code_style_eval/core/checks.py
  - tools/code-style-eval/src/code_style_eval/core/provenance.py
  - tools/code-style-eval/src/code_style_eval/cli/evaluate.py
  - tools/code-style-eval/pyproject.toml
  - tools/code-style-eval/src/code_style_eval/core/scoring.py
  - libs/platform_core/src/platform_core/power_distributions.py
  - libs/platform_core/src/platform_core/minimum_detectable_effect.py
  - libs/platform_core/src/platform_core/mcnemar_detectability.py
  - libs/platform_core/src/platform_core/mcnemar_design.py
  - libs/platform_core/src/platform_core/clustering.py
source_git_blobs:
  "tools/code-style-eval/src/code_style_eval/core/checks.py": 425fe00e5793f7ded70c3e89eb7325b1b983a6cb
  "tools/code-style-eval/src/code_style_eval/core/provenance.py": 2ae8f760d7e83ea5e8a37c8d81939219dbe6e202
  "tools/code-style-eval/src/code_style_eval/cli/evaluate.py": 7ee61da22b34a03c91377041ec92772af0679237
  "tools/code-style-eval/pyproject.toml": 110037c5f0646c8fb6f44ae30ea2112767bffb30
  "tools/code-style-eval/src/code_style_eval/core/scoring.py": 9b1db6975f058fa38977baac3897cc0c14a49619
  "libs/platform_core/src/platform_core/power_distributions.py": 34420ac8a768198e03a96de0f9d175eac9b15f41
  "libs/platform_core/src/platform_core/minimum_detectable_effect.py": b3039c16d0b41636c975a78a8363a56644143d22
  "libs/platform_core/src/platform_core/mcnemar_detectability.py": 25cf0e8370dda398055ea06870f63acb87612a81
  "libs/platform_core/src/platform_core/mcnemar_design.py": b2e29f46a21a50f451be02fd109681e800428119
  "libs/platform_core/src/platform_core/clustering.py": 7611039dbbc21a817e4c10027e282a4e21e9fdbe
provenance:
  - "EVERY RUN BELOW IS NAMED BY ITS payload_digest, not only by its directory. Two runs of this package can agree on every published figure -- sweep-v1 and gen-v1 both scored 226 items -- so a run name is not an identity and a figure quoted off this page was, until 2026-09-09, traceable only by resemblance. It was traced to the wrong run once. The digest is name-paired sha256 over the two *.outcomes.jsonl the comparison was computed from, carried inside comparison.json since 63770146."
  - "runs/sweep-v1/comparison.json + .runrecord.json (label sweep-v4-cap1536-reppen1.1-corpusdeps, 19 distributions recorded) -- payload_digest 1fd266b0f0dc050c54eaeef33523a59b45a7337fb2f586ff28fc3767853fb5e6"
  - "runs/sweep-v1-cap384 -- payload_digest 75efd9fd3192447173d245615830425e1b3a3631961b70d45953a9044bba1876"
  - "runs/sweep-v3-nodeps/ — the same generations scored before the corpus group existed (3 distributions recorded) -- payload_digest d00c4b5057f2ba0bb073d771df9986a85886a30e08a950ba6dfd5c3002a129e5"
  - "runs/sweep-v1/{base,candidate}.outcomes.jsonl and .generation.jsonl — per-item verdicts and termination flags"
  - "runs/sweep-v1-cap384/perplexity.json — teacher-forced NLL per item, both arms"
fact_checked: "2026-09-10"
confidence: high
hubs: [infrastructure]
---

# Guard-pass as a fine-tune metric - what it can and cannot measure

`code-style-eval` scores a generated Python file with the monorepo's own three checkers -- `ruff`, strict `mypy`, and `monorepo_guards` -- and reports the paired McNemar comparison between two arms[^1]. It was built to answer whether a QLoRA fine-tune on this codebase makes a model write code that passes the repo's own standards.

**It cannot answer that at this corpus size, and the reason is arithmetic rather than tuning**[^2]. This page records what the instrument measures, the two effects that dominate its output, and the sample size the question actually needs.

## The measurement

Sweep `sweep-v3-cap1536-reppen1.1`: Qwen2.5-Coder-1.5B base against the QLoRA adapter, 1536-token budget, `repetition_penalty` 1.1, 392-item held-out corpus, 226 items scored in both arms[^2]. Every stratum below was computed by the shipped comparison CLI rather than by a hand statistic[^3]:

| stratum | n | base | candidate | discordant | mid-p | mid-p floor | exact floor |
|---|---|---|---|---|---|---|---|
| all scored | 226 | 2.2% | 2.2% | 3 v 3 | 0.844 | 0.0156 | 0.0312 |
| finished in both arms | 90 | 5.6% | 4.4% | 3 v 2 | 0.688 | 0.0312 | **0.0625** |
| free of unresolvable imports too | 49 | 10.2% | 8.2% | 3 v 2 | 0.688 | 0.0312 | **0.0625** |

**These nulls are very nearly unfalsifiable, and two of them are unfalsifiable
outright under the exact test.** The last two columns are the *floor*: the
smallest p the stratum could have produced under any outcome, given its
discordant count. With *d* discordant pairs the most extreme attainable split
is *d*:0, so the floor is fixed before the data arrive[^16].

At *d*=5 the exact floor is 0.0625 — above α=0.05, so **no possible outcome
rejects**[^16]. Under mid-p, which this page reports, *d*=5 rejects only on a
perfect 5:0 split. Either way, "no difference detected" in those two strata is
a statement about the sample size, not about the adapter.

An earlier version of this page said "The null holds in every stratum" as
though it were a finding. It was not one available to be made: two of the three
strata could not have contradicted it. The MDEs make the same point in the
other direction — 2.46pp, 5.07pp and 9.31pp at 80% power[^16] — each at or above
its own stratum's base rate, so the smallest detectable effect was "the adapter
flips essentially every item that flips at all, in one direction".

Those three figures are mid-p. **Under the exact test the last two strata have
no minimum detectable effect at all**, and the instrument refuses to report one
rather than returning a maximal value: at *d*=5 no split rejects, so no true
effect reaches 80% power however large it is[^16]. That refusal is the same
unfalsifiability the paragraph above states in prose, arriving as an error code
instead of a number a reader could quote.

**The third figure read 9.32pp on this page until 2026-09-10 and is corrected
to 9.31pp here.** It was not a typo. Both *d*=5 strata share an identical net
of 4.5635 items and differ only in denominator, and the published pair was
produced by a search that had not converged — an early-stopped bisection
approaches the answer from above, and at 10 or 11 halvings it reproduces 2.46
and 5.07 exactly while giving 9.32 for the third. Only the *n*=49 stratum is
sensitive enough at that resolution to show the error, because its small
denominator magnifies a residual of under a thousandth of an item[^16].

**90% attrition between the corpus and the set on which the metric is actually measuring code style**[^2].

## Truncation is a hard gate, and it is signal

**Zero truncated items passed, in either arm.** Syntax errors are almost entirely truncation: of base's 69 `[syntax]` failures every one is a truncated item; of candidate's 92, 88 are[^4].

This is not unfair attrition, and the distinction matters. `finishable()` admits an item only when its **reference** fits the token budget, so a model that runs past the budget on such an item has genuinely failed to produce the file[^5]. A truncated completion is a real failure, correctly scored.

The consequence is that guard-pass is gated by termination before it measures anything about style: an arm that rambles more is punished before a single checker runs[^4].

## The import boundary

Scoring places each generated file alone in its own throwaway tree -- which is what `monorepo_guards` requires, since guards are scoped to a tree -- and that is exactly what starves `mypy`. `package_source_roots()` hands `mypy` every `<repo>/<category>/<package>/src` in the monorepo to restore what the file would have seen in place[^6].

Two classes remain unreachable by that mechanism, measured over the 287 parseable generated files[^7]:

| unresolvable, and why | import sites |
|---|---|
| `scripts`, `tests` -- the sandbox puts each file alone, so sibling-module imports resolve against nothing | 56 |
| `outlook_mcp`, `mcp_shared_py`, `voice_call`, `doc_extract_api`, `credential_client`, `corvis_db` -- these live in `~/PROJECTS/MCPs`, outside this monorepo | 39 |

A third class *was* reachable and was closed on 2026-09-03: the corpus's third-party imports (`numpy`, `fastapi`, `discord`, `torch`, `pillow`, and the rest), absent from the scoring environment and therefore reported as missing stubs -- a verdict about the sandbox, not about the generated code. They are now declared as the optional `[tool.poetry.group.corpus]` group and installed for scoring[^8].

**Installing them changed the instrument and changed no conclusion**, which is the part worth remembering. Re-scoring the identical generated files with the group present moved unresolved-import failures from 88 to 66 in the base arm and from 79 to 61 in the candidate, and moved `mypy` passes by one item each, 12 to 13 and 13 to 14. The headline was byte-identical: 5 passes per arm, 2/3/3/218, mid-p 0.844[^9].

The reason is that `mypy` reports the **first** error in a file. The unresolved import was the first error, not the only one; removing it exposed the `Any`-typing error underneath. An import failure that looks like the binding constraint usually is not one, and the only way to know is to remove it and re-measure. The honest gain is narrower and real: the stratum on which the instrument is measuring code rather than its own sandbox grew from 39 items to 49[^9].

Because installing them changes the instrument, `CORPUS_DISTRIBUTIONS` records every member in the run record's packages axis[^10]. A generated file importing `numpy` is a typing failure when `numpy` is absent and a passing file when it is present, with no change to the model's output; two runs scored against different resolvable sets must not be able to claim the same fingerprint. A record naming 3 distributions was scored without the group; one naming 19 was scored with it[^10].

`tensorflow` is deliberately excluded -- one import site in the whole corpus against a roughly 600 MB dependency with no clean Windows/py3.11 wheel. It stays an unresolved import and is reported as one[^8].

## The power calculation

Exact McNemar, alpha 0.05, computed rather than asserted, against the discordant rates the sweep actually produced[^2]:

| comparison | discordant rate | n=226 | n=800 | n=1600 |
|---|---|---|---|---|
| guard-pass, to detect a 70/30 split | 5.6% (observed) | 0.21 | 0.73 | 0.96 |
| guard-pass, to detect a 70/30 split | 11% (clean stratum) | 0.44 | 0.96 | 1.00 |
| termination, to detect a 60/40 split | 33% (observed) | 0.37 | 0.89 | -- |

The corpus holds 392 items. **No amount of retuning the run reaches significance; only more items do.** A guard-pass sweep reported on this corpus without its discordant count beside it is not evidence, and the count is the number to read first[^3].

**And the number of items is now computed rather than left as "more".** At the
observed 5.6% discordant rate, to catch a 70:30 split at 80% power under the
exact test, the run needs **924 scored pairs** — about 52 of which would be
expected to disagree. The corpus holds 392, so the question this instrument
was built to answer needs roughly two and a half times the corpus that
exists[^17].

**That 924 is a floor, because it assumes the pairs are independent draws and
they are not.** At the design effects measured on a comparable monorepo-drawn
corpus — up to 1.744, under which 875 files carried the information of 502 —
the requirement rises toward **~1,600 pairs**, or about four times this
corpus[^15]. The correction is not folded into the instrument, and
deliberately: that 1.744 belongs to gen-v2's 875 items, not to this sweep, and
**this sweep's own design effect cannot usefully be estimated — it would have
to be computed from six discordant pairs.** The instrument that would correct
the sample size needs a larger sample than the one being corrected. So the
honest statement is the range and its reason, not a single adjusted number:
somewhere between two and a half and four times the corpus that exists, and
nothing in between is measurable here.

That table above was itself hand-rolled until 2026-09-10, and it is worth
saying which convention it used, because the page's MDEs used the other one.
These powers average over a RANDOM discordant count — *d* ~ Binomial(*n*,
rate) — while the MDEs condition on the *d* a run actually produced. Both are
defensible and they are not the same number: at *n*=226 the averaged form
gives 0.2061 and the conditional form 0.2026, and it is the averaged one that
rounds to the published 0.21. The shipped instrument reproduces all eight
entries of the table to two decimals[^17].

**And those columns assume the items are independent draws, which held-out
files from a shared monorepo are not.** Every *n* in that table is a count of
files, not a count of independent units, so the powers are upper bounds. The
gap has since been measured rather than left as a caveat: on the 875-item
gen-v2 corpus the intracluster correlation of the paired difference runs
−0.010 to +0.062 depending on whether a cluster is a top-level category, a
package or a directory, for design effects up to 1.744 — under which those 875
files carry the information of 502. The measured table, the instrument
that produces it, and why the correlation must be measured on the paired
difference rather than on either arm's pass rate are all on
[[code-style-qlora-terminates-worse-conforms-better]]; it is not restated
here. The practical consequence for this page's recommendation is narrow and
worth stating: **more items help, and more items drawn from packages already
in the corpus help less than their count suggests.**[^15]

## What the sweep did establish

**Perplexity, and only perplexity.** Teacher-forced on the held-out reference: 2.833 to 1.963, **392/392 items improved, 0 worsened**[^11]. Leakage was checked by path *and* by content -- the latter matters because `scripts/guard.py` is byte-identical across 41 packages.

The one comparison with real power does not favour the adapter. Termination: base 134/226 (59.3%) against candidate 121/226 (53.5%), 44 base-only against 31 candidate-only, net -13, exact p 0.165 and mid-p 0.135[^4]. **Not significant, but the point estimate is a regression** -- the fine-tune appears to make the model ramble more, which is the opposite of the intended effect and is the result most worth re-testing at a larger n.

Read together: the adapter clearly learned this codebase's next-token distribution, and produced no detectable change in whole-file guard-pass[^11][^2].

## An operational hazard, and how it is contained

`make check` runs `poetry sync --with dev`, and `sync` **removes** an optional group. Every lint or test run therefore leaves the package unable to score at full fidelity[^12].

This is worse than it first reads. `code-style-eval` (scoring) builds no fingerprint -- only `code-style-eval-compare` does. So scoring with the group missing did not fail: it quietly produced outcome files with sandbox-artifact verdicts, and the refusal only arrived at compare time, after the expensive part[^10].

`verify_scoring_environment()` now runs before any checker, so the mismatch costs a tenth of a second and writes no outcome file, instead of costing seven minutes and writing a plausible wrong one[^13]. Two tests hold that line: one that the refusal happens, and one that **no checker ran before it** -- a refusal after the work is a refusal that saved nothing[^14].

## What this costs a reader who forgets it

Three passes out of three and thirty out of thirty are both a rate of 1.0, and only one is evidence. Every stratum above is reported with its counts for that reason. The `comparison.json` sidecar records `both_passed`, `baseline_only`, `candidate_only` and `neither` alongside the rates, so a later reader cannot take a pass rate without the denominator that makes it meaningful[^3].

[^1]: `tools/code-style-eval/src/code_style_eval/core/checks.py` section `checker_command` and `score_item` -- the three checkers and how each is invoked.
[^2]: `runs/sweep-v1/base.outcomes.jsonl` and `runs/sweep-v1/candidate.outcomes.jsonl` (226 rows each) with `runs/sweep-v1/comparison.json`; strata recomputed over the same rows.
[^3]: `tools/code-style-eval/src/code_style_eval/core/provenance.py` section `comparison_observations` -- counts are recorded beside rates deliberately, and the docstring states why.
[^4]: `runs/sweep-v1/base.generation.jsonl` and `runs/sweep-v1/candidate.generation.jsonl` (the `finished` flag per item) joined to the outcome rows by `item_id`.
[^5]: `runs/sweep-v1/base.generation.jsonl` -- 226 rows, one per prompt admitted out of the 392-item corpus. The driver that applied the budget filter is a scratchpad script rather than a repo path, so the selection is citable only through the artifact it produced, not through source.
[^6]: `tools/code-style-eval/src/code_style_eval/core/checks.py` section `package_source_roots` and `checker_environment` -- MYPYPATH construction, and the docstring's stated limit on third-party imports.
[^7]: AST import scan over `runs/sweep-v1/base/**/*.py` and `runs/sweep-v1/candidate/**/*.py` -- 287 of 452 files parsed, the rest being truncated and unparseable; top-level names filtered against `sys.stdlib_module_names` and the 54 package roots matching `<repo>/*/*/src/*`.
[^8]: `tools/code-style-eval/pyproject.toml` section `[tool.poetry.group.corpus]` -- the declared set, and the comment recording why tensorflow is absent.
[^9]: `runs/sweep-v3-nodeps/comparison.json.runrecord.json` (label `sweep-v3-cap1536-reppen1.1`, 3 distributions recorded) against `runs/sweep-v1/comparison.json.runrecord.json` (label `sweep-v4-cap1536-reppen1.1-corpusdeps`, 19). The same generated files, scored before and after the group; mypy failure buckets counted per arm from the two `*.outcomes.jsonl` pairs.
[^10]: `tools/code-style-eval/src/code_style_eval/core/provenance.py` section `CORPUS_DISTRIBUTIONS`, `FINGERPRINT_DISTRIBUTIONS` and `scoring_fingerprint`.
[^11]: `runs/sweep-v1-cap384/perplexity.json` -- `items_improved` 392, `items_worsened` 0, per-item NLL for both arms.
[^15]: `libs/platform_core/src/platform_core/clustering.py` --
       `intracluster_correlation`, `average_cluster_size` (Killip's m0 for
       unequal clusters) and `clustered_paired_power`, whose module docstring
       carries the measured difference-versus-raw-indicator table and the
       reason the design effect is floored at 1. The figures quoted here are
       that instrument's output over
       `tools/code-style-eval/runs/gen-v2/{base,candidate}.outcomes.jsonl`,
       via `code-style-eval-clustering`; the full tables and their reading are
       on [[code-style-qlora-terminates-worse-conforms-better]] and are not
       duplicated here. NOTE THE CORPUS IS NOT THIS PAGE'S: gen-v2 scored 875
       items and the sweep this page reports scored 226, so the design effects
       transfer as an order of magnitude for a monorepo-drawn corpus rather
       than as this sweep's own correction, which nobody has computed.
[^17]: `libs/platform_core/src/platform_core/mcnemar_design.py`
       `unconditional_power` and `mcnemar_design_size` (commit `b408a45f`) --
       the UNCONDITIONAL sibling of `mcnemar_detectability`, averaging the
       rejection chance over every discordant count the design might produce
       rather than conditioning on one. It reproduces this page's whole power
       table to two decimals: 0.21 / 0.73 / 0.96 at *n* = 226 / 800 / 1600 on
       the 5.6% stratum, 0.44 / 0.96 / 1.00 on the 11% one, and 0.37 / 0.89 on
       the termination row. The 924-pair figure is
       `mcnemar_design_size(0.056, 0.70, 0.05, 0.80, EXACT, 2048)`, whose
       `expected_discordant_pairs` is 51.744.
       POWER IS NOT MONOTONE IN THE SAMPLE SIZE, which is why that function
       reports two counts and not one. Measured 2026-09-10 across 216
       configurations over *n* = 1..600: 14,214 sample sizes at which adding
       one more pair LOWERS the averaged power, the largest drop 7.8
       percentage points. The cause is discreteness in the rejection boundary,
       and it is worst at high discordant rates and extreme splits — this
       page's own configuration is clean, with `sawtooth_gap_pairs` zero at
       924, which is exactly why a narrow check of it would have missed the
       effect entirely. Where the gap is non-zero the first sample size to
       reach a target is one that running a few more items takes back.
       THE SAME COMMIT MADE `mcnemar_power` BISECT for its rejection boundary
       instead of scanning, which changed no value and is pinned to the
       exhaustive scan over 720 cases by test. It is recorded here because it
       is what made the figures above computable at all: building a boundary
       for every discordant count up to 1,600 took 58 s by scan and 0.97 s by
       bisection, and at 3,000 the scan had not finished after eight minutes.
[^12]: `tools/code-style-eval/Makefile:9` and `tools/code-style-eval/Makefile:30` -- the `lint` and `test` targets, each running `poetry sync --with dev`.
[^13]: `tools/code-style-eval/src/code_style_eval/core/provenance.py` section `verify_scoring_environment`, called from `cli/evaluate.py` section `main` before any work.
[^14]: `tools/code-style-eval/tests/test_evaluate_cli.py` section `TestRefusingAWrongInstrument`.

[^16]: `libs/platform_core/src/platform_core/power_distributions.py`
      `mid_p_mcnemar_p` and `exact_mcnemar_p` [synthesis] — the floor is the
      p-value each function returns at the most extreme attainable split for
      the stratum's discordant count (*d*:0), and the MDE is the smallest true
      split reaching 80% power against that function's own rejection region.
      Both are arithmetic on *d* alone, which is why they are fixed before any
      data is collected. The discordant counts are the table's own, from
      `runs/sweep-v1/comparison.json`.
      These two functions lived in
      `tools/code-style-eval/src/code_style_eval/core/scoring.py` when this
      table was computed and moved to `platform_core` on 2026-09-09 (commit
      `63770146`), which changed no value here: the two implementations were
      compared over every split of every *d* up to 60, both arm orientations,
      and agreed at exact float equality on both tests.
      `libs/platform_core/src/platform_core/minimum_detectable_effect.py`
      `mcnemar_power` now returns each floor as `smallest_attainable_p`
      beside a `can_ever_reject` boolean, and reproduces this table's four
      published floors exactly — 0.015625 and 0.03125 at *d*=6, 0.03125 and
      0.0625 at *d*=5, the last of them `can_ever_reject` **false**, which is
      the unfalsifiability this page reports in prose.
      THE MDEs WERE HAND-ROLLED UNTIL 2026-09-10 AND ARE NOW COMPUTED.
      `libs/platform_core/src/platform_core/mcnemar_detectability.py`
      `mcnemar_detectable_effect` (commit `db7baf3e`) is the magnitude sibling
      of `mcnemar_power`: that one asks whether ANY split rejects, this one
      asks how lopsided the split had to be to reach a target power. Run at
      alpha 0.05, target power 0.80, mid-p, over this page's own discordant
      counts it returns 2.4610pp at *d*=6/*n*=226, 5.0706pp at *d*=5/*n*=90
      and 9.3133pp at *d*=5/*n*=49, each with `achieved_power` 0.800000. Under
      the exact test the *d*=6 stratum is unchanged and both *d*=5 strata
      raise `POWER_TARGET_UNREACHABLE`.
      THE 9.32 -> 9.31 CORRECTION, AND WHY IT IS NOT A TYPO. Two competing
      explanations were tested against the published triple before either was
      written down. A coarse grid over the split is falsified: no step size
      reproduces 9.32 together with the other two (1e-2 gives 2.50/5.11/9.39,
      1e-3 gives 2.46/5.08/9.33, and 1e-4 and finer give 2.46/5.07/9.31). An
      under-converged bisection is not: because bisection retains the
      satisfying bound it converges from ABOVE, and at 10 and 11 halvings it
      yields 2.4630/5.0727/9.3172 and 2.4617/5.0727/9.3172, which round to the
      published 2.46/5.07/9.32. The residual is 0.00083 items of net — the
      published figure needed a net of at least 4.56435 against a true
      4.56352 — which is invisible at *n*=226 and *n*=90 and moves the second
      decimal at *n*=49. The shipped module fixes its halvings at 60 rather
      than stopping on a tolerance, and its constant's comment gives the
      reason; this page's own 9.32 is the measured instance behind it.
      A FOOTNOTE-LABEL COLLISION WAS FIXED IN THE SAME EDIT: this note was
      labelled `[^8]`, which `[tool.poetry.group.corpus]` also claimed, so two
      unrelated citations resolved to one definition. The power citations are
      `[^16]` from 2026-09-10; `[^8]` now means the pyproject group alone.
