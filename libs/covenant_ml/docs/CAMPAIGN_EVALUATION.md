# Frozen campaign evaluation

`scripts.paper_campaign_report` evaluates the saved prospective study without
acquiring market data, submitting orders, refitting a model or altering the
collector. Run it from the current `libs/covenant_ml` package, using the original
manifest digest and a new output path:

```powershell
poetry run python -m scripts.paper_campaign_report --root C:/Users/Test/PROJECTS/files/market-edge-20261012/campaign-20261014-20261212 --digest 4f350cf553c68f6f3f467b59b6f92a8a8c24cf07ddada0c0527224ace17cc077 --out C:/Users/Test/PROJECTS/files/market-edge-20261012/campaign-evaluation-new.json
```

The original isolated collector continues executing the frozen study. The
evaluator verifies its manifest, protocol, fitted model files and exact plan
bytes, but has its own implementation identity. Completed acquisition reports
must identify the original collector. The output and RunRecord bind all read
evidence and the evaluator's implementation separately.

Each plan retains both acquisition dispositions. `future` means its window has
not opened; `due` means an open window has no result. `unrecorded_overdue` means
the window ended without a terminal result. `started_unfinished` retains a start
marker without diagnosing a still-running collector as interrupted. Recorded
`missed`, `failed`, `interrupted` and `dependency_missing` results remain explicit.
The reader does not repair any of these records or retry an acquisition.

Every completed capture must match its frozen plan and original prediction seal.
Every completed settlement must bind that capture's exact bytes. The evaluator
replays native archived settlement responses through the existing outcome codec
and proper-score implementation, and requires the full saved settlement result
to agree. Missing evidence, changed hashes, wrong identities and receipts later
than the analysis timestamp raise errors; no partial report is substituted.

The primary comparison is 12 UTC market-plus-forecast minus market-calibrated
Brier score. Eligible contracts are averaged within an event date across cities,
then event dates receive equal weight. The 00 UTC comparison remains secondary;
both cycles retain all four arms and per-date values. Negative paired differences
favor forecast augmentation. No confidence interval, significance test, cycle
selection or profitable-trading conclusion is added to the frozen protocol.

The existing settlement rule remains intact: a snapshot containing any pending
contract contributes no aggregate score. Scalar payouts and changed rules are
retained and excluded from binary scoring. Captures' original exclusions remain
visible. Scores are descriptive available-case results, so omitted observations
can bias them; the output is not evidence that missingness is harmless. It is a
read of saved evidence, not an atomic snapshot of an actively running collector.
If evidence changes during a read, verification can fail and a later independent
report can be requested with a new output path.

## Actual campaign observation

On **2026-10-13 at 18:00:34 UTC**, the actual evaluator produced
`campaign-evaluation-20261013-initial.json` and its RunRecord in the machine-local
`market-edge-20261012` evidence directory. It verified all frozen inputs and
reported **720 future acquisitions, zero eligible dates and zero eligible
contracts** in both cycles. Both mean Brier scores and paired differences were
null. No future outcomes or demonstrated edge were claimed. The collector and
demo settlement observer both had latest scheduled exit codes of zero at this
check. This observation precedes the first study capture at 20:00 UTC.

The probability study is separate from the filled demo order described in
[KALSHI_DEMO_ORDER.md](KALSHI_DEMO_ORDER.md). A working order workflow verifies
execution mechanics; it does not validate this study's forecasting hypothesis.

## Verification

The completed evaluator passes the existing package `make check`: 3,361 tests,
strict mypy across 557 source/test/script files, all repository guards, and 100%
coverage of 15,916 statements and 3,044 branches. The full check took 151 seconds.
The integration tests execute actual HTTP capture, archived native settlement
replay, CLI subprocesses, filesystem failures and report provenance checks.
Unequal contract counts across dates verify that the study mean gives dates
equal weight. Mutation tests cover altered plans, collector identity, report
digests, prediction bindings, future receipts and disagreement with replay.
