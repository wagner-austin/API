---
title: Evaluation instruments have explicit numerical contracts and bounded claims
tags: [evaluation, research, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]", "[[model-trainer-run-record-provenance]]"]
source_paths:
  - "libs/covenant_ml/docs/EVALUATION_CONTRACTS.md"
  - "libs/covenant_ml/src/covenant_ml/metrics.py"
  - "libs/covenant_ml/src/covenant_ml/ranking.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/auc.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/intervals.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/learning_curve.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/temporal.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/paired.py"
  - "libs/covenant_ml/tests/evaluation/test_numerical_contracts.py"
  - "libs/covenant_ml/tests/evaluation/test_ablation.py"
  - "libs/covenant_ml/tests/evaluation/test_learning_curve.py"
  - "libs/covenant_ml/tests/evaluation/test_moving_block.py"
  - "libs/covenant_ml/tests/evaluation/test_wrds.py"
  - "libs/covenant_ml/tests/markets/test_research_trial.py"
  - "libs/covenant_ml/tests/markets/test_research_execution.py"
  - "libs/platform_core/src/platform_core/run_record.py"
  - "libs/platform_core/tests/test_run_record.py"
source_git_blobs:
  "libs/covenant_ml/docs/EVALUATION_CONTRACTS.md": "6b242c7bc3244b1078a0162f87ec6d90c548d21d"
  "libs/covenant_ml/src/covenant_ml/metrics.py": "5652fb6f2a94f6fd489c3e66f6fe8741ed9d3b65"
  "libs/covenant_ml/src/covenant_ml/ranking.py": "158cb9e9b19de478088f7b341b305f3c4adb5d13"
  "libs/covenant_ml/src/covenant_ml/evaluation/auc.py": "91166dbd6b12d79928b05401f5550e092e5337f1"
  "libs/covenant_ml/src/covenant_ml/evaluation/intervals.py": "75135545a8723b607494992b04ddfc54ad40d420"
  "libs/covenant_ml/src/covenant_ml/evaluation/learning_curve.py": "cfe54855e0f95144defe6b8a60c8d7bf02195232"
  "libs/covenant_ml/src/covenant_ml/evaluation/temporal.py": "f037a07c988d1bb53d7e5a228ea346f328dd4c7e"
  "libs/covenant_ml/src/covenant_ml/evaluation/paired.py": "930ff0876202a68c51454352dc63c0b7d80160c8"
  "libs/covenant_ml/tests/evaluation/test_numerical_contracts.py": "19b1eb136500fef16ecdd2bc94b8c833ea4eb551"
  "libs/covenant_ml/tests/evaluation/test_ablation.py": "0c3d647524af79572f425686748de3ea19b64920"
  "libs/covenant_ml/tests/evaluation/test_learning_curve.py": "7b555bc91238c0b3e897695cbe6af66db1df549e"
  "libs/covenant_ml/tests/evaluation/test_moving_block.py": "3b797d26f5e65ddee1f4ad7472fd05ccf0549c28"
  "libs/covenant_ml/tests/evaluation/test_wrds.py": "638d804f3b8bdca4657cc6ac6832d5b51a757299"
  "libs/covenant_ml/tests/markets/test_research_trial.py": "1637ec192a1cf1e0f7b6a3a4eefda34a7a33ca3e"
  "libs/covenant_ml/tests/markets/test_research_execution.py": "6c83658eb2e0447a2a06fd12d23e6c3b138c1d30"
  "libs/platform_core/src/platform_core/run_record.py": "7d3ca55d81674311342b179cbf476a3d61b14dd6"
  "libs/platform_core/tests/test_run_record.py": "5bd7f783973789c95e490238b34eec53d90e5237"
provenance:
  - "fleet job 43e15f3c-c588-4bb8-a203-f4035975776a: make check for libs/covenant_ml at API 43c23e111a4888d387d1da9647be328bbb23ddf6 on lavender, exit 0, 4699 passed, 0 failed, coverage total 100.00%, ended 2026-10-30T00:48:12Z"
  - "fleet job 172cc4e1-b31f-4dc9-b8db-ac2cdc60ee3e: make check for libs/platform_core at API 43c23e111a4888d387d1da9647be328bbb23ddf6 on sedona, exit 0, 1694 passed, 0 failed, coverage total 100.00%, ended 2026-10-30T00:44:55Z"
  - "hand re-derivation 2026-10-30 for board task 82a072db: unequal-cluster variance 3/2 x 38/36 = 19/12; t(1) radius 0.5 tan(0.45 pi) and p = 1 - 2 atan(4)/pi; moving-block means 0.5, 2.5 and 6.5 have variance 56/9, so two draws give 28/9, and 1/9 tail masses put the 2.5 and 97.5 percentiles at 0.5 and 6.5; label purge keeps train times 0 and 1 and validation time 3 when availability must precede the next fold's first time"
fact_checked: 2026-10-30
confidence: medium
---

# Evaluation instruments have bounded claims

The existing evaluation package supplies paired comparisons, intervals,
feature ablation, sample-size curves and chronological partitions. These are
instruments for a study; they do not establish an investment result. The
[contract reference](../../libs/covenant_ml/docs/EVALUATION_CONTRACTS.md)
names their tests and interpretation limits.[^contracts]

| Capability | What the implementation verifies | Limit |
|---|---|---|
| Paired comparisons | Shared observations; unequal clusters retain the row-weighted estimand. | Inference depends on the chosen dependence model. |
| Confidence intervals | Analytic one-degree-of-freedom Cauchy oracle; finite estimates/errors and valid sample sizes. | Correct arithmetic does not validate assumptions for a new dataset. |
| Ranking scores | Headline and paired AUC share tie-aware ranks; average precision groups distinct score thresholds. Undefined scores fail. | Earlier reports with ties require a versioned rerun. |
| Ablation | Real LightGBM fits remove the named columns, retain labels and compare paired seeds. | Conditional feature contribution is not causal importance. |
| Learning curves | Nested random training prefixes with unchanged stopping/scoring rows. | These are sample-size curves, not expanding historical windows. |
| Chronology | Actual label availability must precede the next stopping/scoring decision. | A split function does not restrict file or tool access. |
| Moving blocks | Identical consecutive month identities, shared overlapping noncircular blocks and exact sample length. | Edge observations appear in fewer blocks; short histories limit inference. |
| Provenance | Changed extraction/staging identities are refused; shared records retain exact observations and configuration/runtime identity. | Owner-writable evidence is not adversarial attestation. |

The numerical tests use independent computations: a Cauchy quantile,
hand-calculated unequal clusters, pairwise AUC enumeration and an enumerated
small-sample moving-block distribution. Ablation and learning-curve cases fit
the actual learner. These checks establish specific contracts, rather than
proving every dataset or model statistically valid.[^numerics][^fits]

The across-seed correction and error floor are declared heuristics. Seeds
are repeated fits, not extra independent market histories. The percentile
record's `p_value` is an uncentered sign-tail diagnostic, not a calibrated
null-bootstrap test. The primary monthly comparison must use the aligned
month series and its stated block rule.[^intervals]

The WRDS adapter hashes the query and CSV it reads. Staging hashes the exact
bytes written, and the worker rechecks protocol, manifest, code and image
identities. RunRecord itself does not reread every source payload or prove
historical publication availability. Restricted holdout execution remains a
separate integration requirement in the linked WRDS protocol.[^provenance]

At API `43c23e111` the covenant_ml check passed 4699 tests and the
platform_core check passed 1694, each at 100% statement and branch coverage.
The cluster, Cauchy, moving-block and label-purge expectations were
re-derived by hand and agree with the tests. These results verify the
instruments at that commit, not the monthly holdings study built on
them.[^checks]

[^contracts]: `libs/covenant_ml/docs/EVALUATION_CONTRACTS.md`, instrument table and Interpretation limits; board task 82a072db-6973-4b24-a76b-ed06166d1a02.
[^numerics]: `tests/evaluation/test_numerical_contracts.py`, Cauchy and unequal-cluster cases; `test_moving_block.py`, enumerated distribution and paired cancellation; source `metrics.py`, `ranking.py`, `evaluation/auc.py`, `evaluation/temporal.py` and `evaluation/paired.py` under `libs/covenant_ml`.
[^fits]: `libs/covenant_ml/tests/evaluation/test_ablation.py`, `test_removing_the_signal_costs_auc_and_removing_noise_does_not`; `test_learning_curve.py`, nested-prefix and real-fit cases; `evaluation/learning_curve.py`, `learning_curve_seed`.
[^intervals]: `libs/covenant_ml/src/covenant_ml/evaluation/intervals.py`, `pool_across_seeds` and `percentile_interval`; `evaluation/paired.py`, `compare_moving_block`.
[^provenance]: `libs/covenant_ml/tests/evaluation/test_wrds.py`, modified evidence and symlink cases; `tests/markets/test_research_trial.py`, frozen identity refusals; `test_research_execution.py`, exact-byte bundle test; `libs/platform_core/src/platform_core/run_record.py` and `tests/test_run_record.py`, observation round trips and incompatible comparison cases.
[^checks]: fleet jobs `43e15f3c-c588-4bb8-a203-f4035975776a` (libs/covenant_ml) and `172cc4e1-b31f-4dc9-b8db-ac2cdc60ee3e` (libs/platform_core) at API `43c23e111`, with the hand re-derivation, all recorded under `provenance:`; board task 82a072db-6973-4b24-a76b-ed06166d1a02.
