---
title: CRSP version-3 refused-path source diagnosis
tags: [wrds, crsp, missingness, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]", "[[wrds-baseline-v3-freeze]]", "[[crsp-monthly-holdings-validation]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json"
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/native-evidence.json"
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/completion-preparation-report.json"
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/audit-execution.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json": "5fca13ce388bd45adc74f8e52efa5d7d42af7775"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/native-evidence.json": "ef35b0e783656e3573b4aa35f741d66a778b30bd"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/completion-preparation-report.json": "032a26faf8678ed0dd56d2d70e7895c036d62210"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/audit-execution.py": "c9c37944ab5ca06f18953e9f9e3d4d21cb0e8438"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py": "41c02cc391a40f28bd3dbd300e3608579d5db586"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py": "13da81586926004b2f0ea07534af2dd295c5a1ea"
provenance:
  - "Vendor definitions use the existing tech-wiki capture crsp-ciz-guide-20260731.txt, SHA-256 bb4792deb927b972a8bef1a656ba1fc5442b80783e10eecb3e633d4243334b2e, corpus document 0264d5cc-4dec-4715-b17b-45f3c8e5fbec."
fact_checked: 2026-11-02
confidence: medium
---

# CRSP version-3 refused-path source diagnosis

Native job `58766611` completed `0:0` in 2,450 seconds. The existing inspector
verified seventeen inputs, result SHA-256
`3dd11ddad1ea74724a44f7bcc1b82b4f581ea51462647fbd7592664abf75ec65`
and the shared RunRecord. The unchanged frozen evaluator reproduced all
262,000 paths: 261,973 accepted and the same 27 first refusals. This is source
diagnosis, not model fitting or a successful baseline.[^native][^cases]

## Every original refusal

The following interpretation groups the native `refused_cases` by their source
fields. Ordinals identify paths in the frozen scan, not securities. Counts are
mutually exclusive first-refusal counts; additional problems can coexist.[^cases]

| Source reason | Count | Path ordinals and observed boundary |
|---|---:|---|
| Positive `BA` quote on a delisting boundary | 6 | Exits 9165, 105499, 135749; next-window entries 10141, 106500, 136773. Each exit/entry pair shares the same row and terminal-record digests. A quote is not a permitted `TR` fill. |
| Not tracked (`NT`) | 6 | Entry 84356 has a null price. Held paths 54734, 62905, 83373, 84496, 92184 have `NT/NT/MR` price/return/duration flags and missing returns through exit. |
| Missing corporate-action value (`MV`) | 6 | Held paths 10402, 59635; terminal paths 134760, 173979, 179591, 191467. All six contain finite numeric returns, but the vendor flags them `MV`, not available `NA`. |
| Terminal valuation after scoring exit | 5 | 18863, 34781, 44820, 187445 have amount dates one calendar day after exit; 27156 has amount/next-price dates two days after exit. Terminal returns are available, but the frozen timing rule refuses them. |
| Suspended (`SU`) price gap | 2 | 12343 has nine `SU/MP/MR` sessions followed by an available `P9` return linked exactly to entry. The frozen gap permission requires price flag `MP`. Path 23610 remains suspended through exit with no closing span. |
| Late delisting return (`DG`) | 1 | 133712 has null `DelRet`, flag `DG`, and amount/next-price dates 69 calendar days after the event, beyond exit. The frozen numeric read raises a type error before its flag refusal. |
| Unclosed missing-price tail | 1 | 169697 ends with `MP/MP/MR`, null return and no closing span by exit; `CRSP-011` is explicit. |

The earlier stack-derived `held_path` category included the five late-valuation
cases. It must not be interpreted as fifteen missing ordinary-return paths.
The source fields distinguish ten held-return refusals from five valuation-date
refusals. Twenty-five total refusals are training paths; validation contributes
187445 and 191467.[^cases]

## Data limitations versus implementation

The CIZ guide distinguishes a quote from a closing trade, suspended/not-tracked
prices, missing corporate-action value and late delisting returns. A populated
number alone is not proof of an available complete return. DelDlyDt is a
conventional storage date, and DelAmtDt dates the amount used in calculation;
neither establishes actual cash receipt.[^vendor]

No correctable extraction defect is established by this evidence. The frozen
files pass their checksums; extraction projects the price/return fields directly,
and decoding preserves blank nullable values. This proves retained-byte
consistency, not that the vendor's economic observations are complete or that a
fresh vendor vintage would be identical. No fresh vendor correction or external
replacement observation was measured.[^native][^extract]

There is a narrow reporting defect: the frozen execution validator reads a null
price before issuing its documented `CRSP-004` refusal. The current-source
correction reuses the existing numeric validator and still refuses the price;
[[crsp-monthly-holdings-validation]] records the change and regression cases.
The shared current terminal resolver already checks the missingness flag before
reading a null return. Neither correction supplies missing values or changes
the frozen image.[^cases][^implementation]

## Next action

**Keep version 3 stopped.** Do not retry its fourteen unsubmitted requests,
turn quotes into trades, accept `MV` merely because a number exists, or remove
securities using future availability. The original journal remains unchanged:
one failed, fourteen admitted, zero selections and zero final bindings.[^journal]

Before any new freeze, independently review explicit execution and incomplete-
outcome rules: the three rebalance delistings, the exactly linked suspended span,
and valuations after the current label boundary are protocol questions. The
unresolved `NT`, `MV`, `DG` and open-gap cases additionally need evidenced source
corrections or an explicitly justified incomplete-outcome design. Expanding the
gap predicate alone cannot qualify this dataset. Candidate fitting and bounded
agent experiments remain downstream of that qualification.[^cases][^implementation]

The program asserts normal-exit private-directory removal. That observation is
not a new forced-termination isolation test. Public evidence contains flags,
digests and relative day offsets, not security identifiers, calendar dates,
prices or return values.[^program]

[^native]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/native-evidence.json`, `accounting`, `measurement.verified_files`, `measurement.result_sha256`, `measurement.record_sha256` and `record`; live service post 94d8e8a5-54c1-4b42-a708-75148ea4ed09.
[^cases]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json`, `results[0].refused_cases`, each `path_ordinal`, `source_window.daily_rows`, `source_window.terminal_rows`, boundary digests, counts and `prior_counts_reproduced`; source `087f67b4c`, unchanged image `369ae415a20a1ad8a7d672d839e12c7e62ee29d1432667d95f479d333c578fde`.
[^vendor]: [CRSP CIZ guide](https://indexes.morningstar.com/api/docs/6a71c517e35d02a71403136d), pp. 11–12 (delisting fields), 17–18 (daily fields), 88 (PC), 90–91 (RD), 91–92 (RM); archived capture identified in frontmatter. Existing tech-wiki pages `crsp-ciz-daily-return-contract` and `crsp-ciz-daily-return-quality` retain the primary-source analysis.
[^extract]: `libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py`, `crsp_relation_sql` returns/delists projections; `libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py`, `_cell` and `decode_crsp_rows`; no missing numeric value is replaced with zero.
[^implementation]: `wiki/pages/crsp-monthly-holdings-validation.md:67`, current shared resolver; `wiki/pages/crsp-monthly-holdings-validation.md:86`, execution-price refusal and boundary tests; [[wrds-baseline-v3-freeze]], frozen design and execution boundary. Proposed permission changes here are not installed rules.
[^journal]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/completion-preparation-report.json`, `observed_at`, `journal_sha256` and all fifteen `attempts`; live service post c13f1621-06fc-4af0-adb0-d9037f7f48ab after native completion.
[^program]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/audit-execution.py`, `source_window_context`, train-validation key acquisition, prior-result comparisons and private-directory removal assertion.
