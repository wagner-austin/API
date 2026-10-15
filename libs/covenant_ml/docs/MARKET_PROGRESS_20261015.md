# Market evidence observation, October 15 UTC

This observation refreshes the running studies without changing any frozen
decision rule or sending an order. Inspected code: published commit
e03bbd2eba3255c1df569b209e5a58986205a22e. Board task:
ffbf2686-3427-49d6-8989-b743123066d5.

## Frozen weather campaign

The real offline evaluator ran at 2026-10-15T02:48:46.077351+00:00 and wrote
`files/market-edge-20261012/campaign-evaluation-20261015-0248.json` plus its
RunRecord under `C:/Users/Test/PROJECTS/`. All 520 listed source-file hashes
were independently recomputed and matched. It reports 12 complete acquisitions
and 708 future acquisitions, zero eligible settled dates/contracts in either
cycle, and null Brier scores. No missed or failed disposition was reported.

The scheduled collector last ran successfully at 2026-10-15T02:48:01Z.
The frozen `acquisition_window` function gives the first settlement collection
window as October 16, 15:00-16:00 UTC (08:00-09:00 Pacific). This is the planned
collection time, not a promise that every exchange contract will have settled.
The full original study still runs through December 12 with later settlements.

Reproduce the observation using `scripts.paper_campaign_report`, the original
campaign root and manifest digest from `CAMPAIGN_EVALUATION.md`, and a new
output path. A later observation may differ because the collector continues.
Captured report SHA256:
85539228f6615c3354e1421c2247c040138134de627b4992936d4fe944a27a98.

## Completed demo lifecycle

The real `scripts.kalshi_demo_trial ... status` command replayed the existing
execution journal. The saved status is
`files/market-edge-20261012/demo-status-20261015-0252.json`, SHA256
88cf87ac8ab35190688ea4ee91cb1f5bb348e590c8c5572404de4c9c784ea3f7.
Its last validated observation is 2026-10-15T02:50:40.915Z. It reports one
confirmed order, one confirmed settlement, no issues and zero exposure.

The one YES contract cost $0.30, fees were $0.0147, and credited revenue was
$1.00. Realized DEMO profit is $0.6853. Native reconciliation also records
$50.6853 mock cash from an initial $50 and no unsettled position. The exchange
settlement timestamp is 2026-10-14T02:51:32.85762Z. This is mock money from one
execution test, not a statistically supported strategy or real-money profit.
`live_ready` and `can_reserve` remain false; this single-order trial is complete.

## Next execution requirement

`DemoTrialService.submit` checks active market identity, executable book depth,
price limit and freshness, reserves risk, submits once, and saves the native
acknowledgement. Its current implementation does not persist the pre-send book
or separate submission-start and acknowledgement-receipt timestamps.
Reconciliation persists orders, fills, positions, balance and settlement.

The next focused implementation is durable pre-send evidence for future demo
trials: exact quote/size and contract rules, local receipt and decision times,
submission invocation and acknowledgement receipt times, intent identity and
eventual fills/fees. Local invocation time must not be described as an exchange
matching-engine timestamp. Failure or ambiguity must retain the reservation;
there must be no inferred fill or automatic repeat order. The completed order's
missing pre-send observations cannot be reconstructed from later books.

This recorder would test execution assumptions, not establish forecast skill.
Any later profitability experiment still needs a separately frozen decision and
fee policy, comparable baselines, independent event observations and full
reporting of abstentions, rejections and unfilled orders. The weather campaign
is a probability-scoring pilot and is not retroactively converted into that
economic experiment.
