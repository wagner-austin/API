---
title: WRDS equity research extends the existing experiment and coordination systems
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]", "[[crsp-monthly-study-design]]", "[[crsp-monthly-data-contract]]", "[[crsp-monthly-holdings-validation]]", "[[wrds-evaluation-reuse]]", "[[wrds-research-queue-admission]]", "[[wrds-integration-ownership-review]]", "[[wrds-proposer-isolation]]", "[[wrds-final-scoring]]", "[[evaluation-numerical-contracts]]", "[[model-trainer-run-record-provenance]]"]
source_paths:
  - libs/covenant_ml/docs/MARKET_RESEARCH.md
source_git_blobs:
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "e30492e1ef4ae30e3c3391560a066f51805d6250"
provenance:
  - "Decomposed on 2026-10-30 from API 14b60b8610c98f17357869422cf2239ef4fddbf8 under board task daca401b-a9cc-4cc8-8626-8ee760bedad5. Detailed claims, source pins and dated measurements moved together."
fact_checked: 2026-10-30
confidence: medium
---

# WRDS equity research extends the existing systems

The research libraries own admission, evaluation and provenance; the fleet queue
and maintained cluster controllers own execution. The monthly worker, isolated
proposal admission and evaluator-owned final release extend those existing
systems. Implementation is distinct from deployment and a measured study:
start with [[wrds-baseline-readiness]] for that boundary.[^contract]

## Read by question

| Question | Authoritative page |
|---|---|
| What has run, and what remains before the baseline? | [[wrds-baseline-readiness]] |
| What are the frozen target, universe, dates and costs? | [[crsp-monthly-study-design]] |
| Which CRSP relations, flags and date rules construct the panel? | [[crsp-monthly-data-contract]] |
| How do holdings, turnover, label purging and uncertainty work? | [[crsp-monthly-holdings-validation]] |
| Which existing evaluation components are reused? | [[wrds-evaluation-reuse]] |
| How do proposals enter admission and the shared queue? | [[wrds-research-queue-admission]] |
| Which integration ownership and recovery paths were reviewed? | [[wrds-integration-ownership-review]] |
| What can a proposal program access, and what was tested on a real host? | [[wrds-proposer-isolation]] |
| Who selects a candidate and consumes the single final release? | [[wrds-final-scoring]] |

## Related systems

- [[evaluation-numerical-contracts]] explains the shared statistical instruments.
- [[model-trainer-run-record-provenance]] explains shared result identity.
- [HPC3 operations](../../tools/hpc3/wiki/hubs/operations.md) holds cluster
  run and installed-controller evidence.
- [Research index](../../docs/RESEARCH.md) lists registered projects.

[^contract]: `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission" and § "Monthly holdings and validation study"; numerical implementation and runtime evidence are separated in the linked pages.
