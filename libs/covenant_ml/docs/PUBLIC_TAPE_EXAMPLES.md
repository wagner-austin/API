# Public tape examples and calendar expansion

Board task 6196ba83-eab1-4690-aa72-1040970a3c78.

`scripts.public_tape_examples` reads completed public tape reports, verifies each
RunRecord and every referenced source, independently recounts native rows, and
writes examples with a new provenance record. It refuses overlapping input
windows rather than adding their executions twice. Contract and event counts
remain per window: summing them would not establish unique identities across dates.

```powershell
python -m scripts.public_tape_examples --reports day1/results.json day2/results.json --out examples.json
```

Every ticker namespace receives its own execution count, contract count, and
the native execution with the lexicographically smallest trade ID. That rule
does not examine outcomes or returns. The full metadata record accompanies each
example and is explicitly labeled as observed at capture time. A namespace is
an exchange identifier, not a verified category. Declared combination-leg event
counts expose dependence without claiming to measure independent sample size.

The native recount rejects duplicate trade or metadata identities, missing or
extra contract metadata, wrong reported counts and the excluded window end.
Example selection is invariant to input row order. Real filesystem and child
process tests cover the CLI, source corruption, output refusal and overlap.

An independent review of the preceding collector found that its tier-routing
test reused one trade ID on both tiers: omitting the live tier could pass. The
revised real HTTP test supplies a historical-only trade, a live-only trade and
one overlapping trade, then asserts exact retained rows, quantities and both
request URLs. No capture result is inferred from a test's passing status.

## Fixed expansion selection

The expansion plan was saved at 2026-10-15T12:08:56.6596760Z, before capture:
October 7 through 13, 2026, every day from 20:00:00 inclusive to 20:05:00 UTC
exclusive. It retains all public executions exposed in each window, without
price, outcome, category or block exclusions. Each window has its own plan,
archive, result, offline replay and RunRecords. The original October 14 hour
remains a separate, nonoverlapping eighth date.

This is exploratory coverage, not a forecasting holdout. Seven short windows
do not represent all hours, and calendar dates alone do not establish independent
outcomes. The execution-selected sample excludes untraded contracts. Neither
public trade prices nor subsequent metadata demonstrate accessible quote size,
queue position, actual fees paid or achievable fills. This procedure reports no
return and selects no strategy.

Artifacts: `files/market-edge-20261012/public-tape-week-20261015/`.
The frozen `manifest.json` binds every per-day plan hash. Replay uses the existing
`scripts.public_tape --mode replay` command documented in `PUBLIC_TAPE_COLLECTION.md`.
The report records the implementation digest actually used by each invocation;
changes to the separate example reporter do not alter captured native responses.
