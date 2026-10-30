---
title: Evaluator-owned WRDS final scoring
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/covenant_ml/src/covenant_ml/markets/research_journal.py"
  - "libs/covenant_ml/tests/markets/test_research_selection.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_final.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_final.py"
  - "libs/covenant_ml/scripts/research_proposals.py"
  - "libs/covenant_ml/scripts/research_trial.py"
  - "libs/covenant_ml/tests/markets/test_research_final.py"
  - "libs/covenant_ml/tests/markets/test_crsp_final.py"
source_git_blobs:
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "e30492e1ef4ae30e3c3391560a066f51805d6250"
  "libs/covenant_ml/src/covenant_ml/markets/research_journal.py": "a36df5c2e20b454eabd26f3acfc75f07f1c3485e"
  "libs/covenant_ml/tests/markets/test_research_selection.py": "b2939285c746d390261861ed80edae3455000a84"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "09aad4f0cf5563d59f0b431ffed6e3c5911bd42a"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "2288c853df8b360d8d8654b9eb2578af2fb3982e"
  "libs/covenant_ml/src/covenant_ml/markets/research_final.py": "854dd095f76364bc1a00ca0e9e35ca2e06ce88f7"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_final.py": "fa810eb315b13925c5664c35b883b80be0afee2d"
  "libs/covenant_ml/scripts/research_proposals.py": "f62f4f85739a00b2fbdc41f1e8724dd59188632f"
  "libs/covenant_ml/scripts/research_trial.py": "3e8ead7063b2e1835bb59131d73b9df0095b25d5"
  "libs/covenant_ml/tests/markets/test_research_final.py": "81e4a8f1f5220b502de59369c9a8ac1868f8ab77"
  "libs/covenant_ml/tests/markets/test_crsp_final.py": "67af9e640314ef33871e9a182d6511fb75dc0abc"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
fact_checked: 2026-10-30
confidence: medium
---

# Evaluator-owned WRDS final scoring

Research entry point: [[wrds-equity-research-protocol]].

The existing `research_journal` accepts evaluator-owned `select` and `release_final`
transitions. Selection requires settled validation work, binds a completed
candidate and its validation payload to an exact final manifest, and seals
further admissions. Release is consumed once against those identities while
preserving the validation record and its `complete` execution status for queue
reconciliation; `selection` records the separate final-release state.
These are state invariants, not caller authentication: the journal must remain
evaluator-owned.[^selection]

`research_final` issues them. `research_proposals select` binds the named
completed attempt's journal candidate and payload to the final manifest's
SHA-256. `research_trial --ledger --final-manifest` requires that selection and
those bytes, rechecks policy, protocol, code and image lineage, and commits
`release_final` before reading any final-period relation. A later failure,
even a missing data file, leaves the release consumed, and a second run is
refused. The candidate, seed and features come from the journal's own admitted
request; a proposal naming another partition is refused at admission and can
never be selected, so no request field authorizes final scoring.
`crsp_final.run_crsp_final` fits once on every decision whose exit precedes
`holdout_start` and scores every decision from `holdout_start` through
`holdout_end` with the validation study's shared fold, arm and record code. Changing every outcome
after the last final decision changes no fit or selection in the tests.
Choosing which attempt to select remains an evaluator decision.[^final]

[^selection]: `libs/covenant_ml/src/covenant_ml/markets/research_journal.py`, `_transition` and `_final_identity`; `libs/covenant_ml/tests/markets/test_research_selection.py`, real SQLite selection, release and atomic refusal cases; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission".

[^final]: `libs/covenant_ml/src/covenant_ml/markets/research_final.py`, `select_candidate` and `execute_final`; `libs/covenant_ml/src/covenant_ml/markets/crsp_final.py`, `run_crsp_final`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `crsp_fold`, `crsp_arms` and `crsp_known_panel`; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py`, `require_trial_lineage` and `publish_monthly_record`; `libs/covenant_ml/scripts/research_proposals.py` and `libs/covenant_ml/scripts/research_trial.py`; `libs/covenant_ml/tests/markets/test_research_final.py` and `test_crsp_final.py`, real journals, extracts and trainers; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission". Fleet job 379d582f-130b-47c4-8579-56ec1be8d586 passed make check for libs/covenant_ml at API 06636dbbebd56771e170b45a7f190309f396cde1 on Colossus at 2026-10-30T05:45:28Z: 4,786 tests, 100.00% coverage. The earlier job 2e9095f9 at bddfcab33 ran every test green but failed the test-quality guard on one substring assertion, repaired in 06636dbbe. On main at 85e408c1c08cd3ff5faaf88416f4d62965ed64a3, job 8054c71b passed libs/covenant_ml on Colossus at 2026-10-30T05:57:25Z: 4,786 tests, 100.00% coverage. Board task 364a4ba8-0ff0-461c-ba7f-dfe7a4b039ba.
