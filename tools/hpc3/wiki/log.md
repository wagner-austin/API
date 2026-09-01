# Wiki Operation Log

Append-only. Log structural operations (new hubs, decomposition, audits,
cleanups). Routine page edits don't need a log entry — git history covers
those.

## [2026-09-01] init | wiki scaffolded from the README split
Hubs created: submission, images-and-staging, cluster-facts, operations
Pages written: run-documents, sweeps-and-artifacts, preemption-and-campaigns, submission-rules, chains, unsupported-shapes, image-build-flow, image-ledger-lessons, environment-pins, staging-identity, determinism-posture, known-answers, partitions-and-billing, facts-are-code, job-identity-on-cluster, triage-conditions, ledger-closures, budget-model
Notes: operator-directed split of the 1,045-line README (pre-split state: `README.md@4dc63f17`). The incident narrative moved VERBATIM into atomic pages — the measurements, dates, job ids and command output travelled with their claims; the split did not independently re-verify them, which is why every page's `sources:` names where the claim is checkable (a module, a test, a dated measurement) alongside the pre-split README commit. The README shrank to the command reference and points here. Announced on the agent board before the move (the README was co-edited); one commit carries both sides so the text never left HEAD.

## [2026-09-01] correction | the split cut tested reference, and its commit landed on a red gate
Pages updated: unsupported-shapes (its table now lives only in the README, where the test holds it)
Notes: the first split commit (`ef1a857d`) went in while `make check` showed six failures — `test_examples.py` holds the README to being a LIVE example (eight decodable JSON blocks, three project entries, the cannot-submit table), and the trim had cut all of it as if it were narrative. Both mistakes are corrected in `5eb68519`: the tested examples are restored to the README under "Documents by example" (they are reference, and the test is their drift-guard), the wiki page keeps the reasoning without duplicating the tested table, and the gate is green (1262 passed). The process lesson stands on its own: the gate ran, reported red, and the commit happened anyway in the same compound command — gate and commit belong in separate steps so a red result can stop the hand.
