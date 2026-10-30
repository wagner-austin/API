---
title: WRDS evaluation reuses the shared numerical contracts
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]", "[[crsp-monthly-study-design]]", "[[wrds-proposer-isolation]]"]
source_paths:
  - "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py"
  - "libs/covenant_ml/src/covenant_ml/markets/equity_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/equity_outcomes.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/resampling.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/paired.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/splits.py"
  - "libs/platform_core/src/platform_core/run_record.py"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/covenant_ml/src/covenant_ml/markets/research_execution.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial.py"
  - "tools/fleet/src/fleet/core/research_control.py"
  - "tools/fleet/src/fleet/core/research_hpc.py"
  - "tools/fleet/src/fleet/core/research_wrds.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py"
  - "libs/covenant_ml/tests/markets/test_crsp_holdings.py"
  - "libs/covenant_ml/tests/markets/test_crsp_study.py"
  - "libs/covenant_ml/tests/markets/test_research_trial.py"
  - "tools/fleet/src/fleet/core/research_proposal.py"
  - "tools/fleet/src/fleet/cli/research_propose.py"
  - "tools/fleet/tests/test_research_proposal.py"
  - "tools/fleet/tests/test_research_proposal_host.py"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py": "bb7b6f6db20f501916ba08821c0e4efd2166bbe7"
  "libs/covenant_ml/src/covenant_ml/markets/equity_study.py": "e4e44e46f0031df3686b09d5067a1cea49033eb8"
  "libs/covenant_ml/src/covenant_ml/markets/equity_outcomes.py": "2ff9370dd458fcc4cf3e5a7d21a490583f88ae05"
  "libs/covenant_ml/src/covenant_ml/evaluation/resampling.py": "6d95079f002b370fa9a8920c3d361a4f803e6136"
  "libs/covenant_ml/src/covenant_ml/evaluation/paired.py": "930ff0876202a68c51454352dc63c0b7d80160c8"
  "libs/covenant_ml/src/covenant_ml/evaluation/splits.py": "1a5b1a1a94e4666d039ea21557480b66eb95bbac"
  "libs/platform_core/src/platform_core/run_record.py": "7d3ca55d81674311342b179cbf476a3d61b14dd6"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "e30492e1ef4ae30e3c3391560a066f51805d6250"
  "libs/covenant_ml/src/covenant_ml/markets/research_execution.py": "01bd2d4443987b275bc58d71b6e127e2382c9f4a"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "09aad4f0cf5563d59f0b431ffed6e3c5911bd42a"
  "tools/fleet/src/fleet/core/research_control.py": "ba591e0282354ffd6b6293717384ada763077f05"
  "tools/fleet/src/fleet/core/research_hpc.py": "ac4a658ffe4ca6beae15ce56df08f712e7bddca2"
  "tools/fleet/src/fleet/core/research_wrds.py": "2130dac9b040de079efcdb5b5b238c6cb6022927"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "c63fd95cfe672e9e198cd1b5cab895e97466dc77"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "2288c853df8b360d8d8654b9eb2578af2fb3982e"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py": "47eaddf94090471d7dc84e188829de4d8cfaf1dd"
  "libs/covenant_ml/tests/markets/test_crsp_holdings.py": "b5081a606ba9f8d21673ef75c9a7da5a197c6dde"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "b797984c6422a9c99cfd4db6c09ef92a40c755f9"
  "libs/covenant_ml/tests/markets/test_research_trial.py": "2b0cd1e2362f58475205774e7dc320b2ae8991d2"
  "tools/fleet/src/fleet/core/research_proposal.py": "dd0d9f77373e490d69aef66d305768a959b6f525"
  "tools/fleet/src/fleet/cli/research_propose.py": "1c5e8bf6bab8bd79564235a292cc875a54a71fe2"
  "tools/fleet/tests/test_research_proposal.py": "e3032d998c3d7e4b0951b2db7d1be643978ac2c5"
  "tools/fleet/tests/test_research_proposal_host.py": "177c817e1992334468ed07142a01df4f71908aa6"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
fact_checked: 2026-10-30
confidence: medium
---

# WRDS evaluation reuses the shared numerical contracts

- Research entry point: [[wrds-equity-research-protocol]].

The existing WRDS adapter verifies the exact query and CSV, builds chronological
splits with an embargo, and predicts next-observation return sign from one prior
return. That is the engineering pilot, not the monthly investment study in [[crsp-monthly-study-design]];
its acquisition timestamp does not prove historical vendor availability.[^adapter]
Existing equity-study code already binds a snapshot, calendar, cohort, weights
and costs. Existing market research verifies a frozen protocol before acquisition.
Reuse those invariants and shared validation/provenance primitives; do not force
CRSP return/delisting semantics into a provider-adjusted-price record.[^equity][^market]
Keep the shared RunRecord and its comparison functions as the numerical evidence
contract; a completed coordination task alone does not establish reproducibility
or out-of-sample performance.[^records]

The existing implementations provide useful parts of the study, but their
contracts differ from it in specific ways. The right column records how the
monthly study now uses or replaces each seam:[^equity][^window][^sampling][^splits][^holdings]

| Existing seam | Verified behavior | Monthly study |
|---|---|---|
| `equity_study.load_equity_study` | Binds a verified snapshot, cohort, selected weights and independently supplied session calendar. | Not reused: `crsp_panel` binds effective-dated CRSP membership and decision/entry/exit dates from the frozen protocol. |
| `equity_outcomes.evaluate_equity_window` | Requires the entire frozen cohort and every declared session; scores adjusted-price endpoints and subtracts one fixed round-trip cost. | Not reused: `crsp_holdings` carries drifted holdings between rebalances, charges absolute traded notional against drifted pre-trade positions and values each terminal event once on its own date. |
| `purge_label_overlap` (also inside `ChronologicalGroupedSplits`) | Requires actual label availability strictly before the stopping and scoring boundaries. | `crsp_study` passes each window's exit close as availability, so each fold drops exactly the decision immediately before its stopping and scoring boundaries. |
| `moving_block_weights` | Samples overlapping, noncircular blocks and truncates the last block to preserve the exact observation count. | Used through `compare_moving_block` with the frozen block length, draw count and seed. |
| `compare_moving_block` | Requires identical consecutive month identities and records a distinct moving-block percentile method. | Compares each candidate and the equal-weight universe with momentum at the primary cost. Its uncentered sign-tail probability is a diagnostic, not a calibrated null-hypothesis test. |

The shared admission journal, queue, native scheduler adapters and RunRecord
publisher compose with the monthly worker without another coordinator or
attempt ledger. The worker never constructs a final-period decision, but its
manifest's snapshot resides under the evaluator's account, so the date bound
alone is not an access boundary. That boundary is the proposal process
described in [[wrds-proposer-isolation]], which holds no path, credential or network route to those
files.[^execution][^controllers][^splits][^holdings][^installed]

[^installed]: `tools/fleet/src/fleet/core/research_proposal.py`, `propose`; `tools/fleet/src/fleet/cli/research_propose.py`, `main`; `tools/fleet/src/fleet/core/research_control.py`, `read_pinned` and `invoke`; `tools/fleet/tests/test_research_proposal.py`, admitted and refused recording, no evaluator path in any daemon argument, drifted-input and isolation refusals; `tools/fleet/tests/test_research_proposal_host.py`, `test_installed_proposer_cannot_reach_evaluator_assets`. Host lane job fc8bc3c5-ad1b-479e-b44c-ddfcd40152ad passed tools/fleet-execution-linux at API 06636dbbebd56771e170b45a7f190309f396cde1 on Sedona at 2026-10-30T05:46:34Z: 79 cases ran, 0 skipped, 0 failed. Full check b01963a6-2105-49a3-a81a-8551982ca23d passed tools/fleet at the same commit on Sedona at 2026-10-30T05:43:08Z: 2,479 tests, 100.00% coverage. Commit 06636dbbe was staged on a pushed-tests ref and rebased unchanged onto main as part of 85e408c1c08cd3ff5faaf88416f4d62965ed64a3; there, host lane job 6833f095-6fe8-48c5-a464-548cf228597b passed on Colossus at 2026-10-30T05:58:34Z (79 cases, 0 failed, 0 skipped) and full check f7581b88 passed tools/fleet on Loki at 2026-10-30T05:55:25Z (2,495 tests, 100.00% coverage). The admission controller in the host case is a recording stand-in for `research_proposals propose`; the real controller's recording of proposals and refusals is the covenant_ml suite's. Board task 364a4ba8-0ff0-461c-ba7f-dfe7a4b039ba.

[^window]: `libs/covenant_ml/src/covenant_ml/markets/equity_outcomes.py:67` `evaluate_equity_window`, complete-calendar check and `net_selected = gross_selected - cost` arithmetic.

[^sampling]: `libs/covenant_ml/src/covenant_ml/evaluation/resampling.py`, `build_resampling`, `bootstrap_weights` and `moving_block_weights`; `libs/covenant_ml/src/covenant_ml/evaluation/paired.py`, `paired_mean_difference` and `compare_moving_block`. The monthly study's use of them is cited in the holdings footnote.

[^splits]: `libs/covenant_ml/src/covenant_ml/evaluation/splits.py` class `ChronologicalGroupedSplits`, methods `__call__` and `validation_split`.

[^controllers]: `tools/fleet/src/fleet/core/research_control.py` functions `read_specification` and `verified_input`; `tools/fleet/src/fleet/core/research_hpc.py` and `tools/fleet/src/fleet/core/research_wrds.py`, function `poll` in each.

[^adapter]: `libs/covenant_ml/src/covenant_ml/evaluation/wrds.py:129` `load_wrds_evaluation`.

[^equity]: `libs/covenant_ml/src/covenant_ml/markets/equity_study.py:55` `decode_equity_study` and `:113` `load_equity_study`.

[^market]: `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Evidence and arithmetic" and preceding command/plan contract.

[^records]: `libs/platform_core/src/platform_core/run_record.py:309` `agree_across_runs`.

[^execution]: `libs/covenant_ml/src/covenant_ml/markets/research_execution.py` submission and reconciliation; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py` `execute_trial`, the monthly validation worker since board task 8f7247a1; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission", execution settings and limitations. Board task 6d2d086c-5242-482b-ac81-8a61dd8e354d.

[^holdings]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `evaluate_crsp_holdings` and `crsp_window_path`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation` and `crsp_study_rows`; `libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py`, `decode_research_evaluation`; `libs/covenant_ml/tests/markets/test_crsp_holdings.py`, `test_crsp_study.py` and `test_research_trial.py`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Monthly holdings and validation study". Initial integration evidence: fleet job 48a2d9c8-b9f6-4c6b-8a65-917bd3828f37 passed make check for libs/covenant_ml at API fd3f7723bd6eb7f5fbc5caa3157efed306bf3949: 4743 tests, 100% coverage. Board task 8f7247a1-aa1a-41d7-be0e-f2b8e97c1d2d. The substantive monthly CRSP study has not run.
