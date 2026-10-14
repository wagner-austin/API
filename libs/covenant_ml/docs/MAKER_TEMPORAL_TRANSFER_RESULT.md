# Philadelphia 2025 transfer: the declared group does not advance

The [protocol](MAKER_TEMPORAL_TRANSFER_PROTOCOL.md) was committed as `f5052401b`
before price/trade acquisition. All 2,190 contracts in the declared 2025
Philadelphia inventory were processed. The high-variation/falling group chosen
from the earlier 2026 analysis lost a hypothetical **$5.61**, or **$7.00** after
an extra cent per fill. No orders were submitted.

This is a retrospective temporal transfer test, not a pristine forward holdout.
It is negative evidence for the specific proposed filter, not a finding that
trading or other forecasting models cannot work. The comparison also changes
from a three-city 2026 panel to one city in 2025, so it cannot isolate a causal
year effect. We do not switch to a different winning group after seeing results.

## Primary two-cent-through execution

| Group | Contracts examined | Fill proxies | Filled event dates | Hypothetical P&L | Extra-cent P&L |
|---|---:|---:|---:|---:|---:|
| Low variation, falling | 246 | 85 | 78 | -$1.50 | -$2.35 |
| Low variation, flat | 155 | 36 | 33 | $1.08 | $0.72 |
| Low variation, rising | 222 | 52 | 46 | -$2.11 | -$2.63 |
| **High variation, falling: declared primary** | **509** | **139** | **121** | **-$5.61** | **-$7.00** |
| High variation, flat | 45 | 11 | 10 | -$0.08 | -$0.19 |
| High variation, rising | 550 | 72 | 70 | $2.86 | $2.14 |
| Unclassified | 463 | 100 | 67 | -$1.40 | -$2.40 |
| Total | 2,190 | 495 | 294 unique dates | -$6.76 | -$11.71 |

Dates overlap across groups. The declared group's mean per fill was -$0.04036,
with a date-clustered 95% interval of [-$0.09496, $0.01424]. Its calendar-day
mean was -$0.01537, with a seven-lag Newey-West 95% interval of
[-$0.03307, $0.00233] over the 365-day calendar. Neither lower bound is positive,
and extra-friction profit is negative: all three declared checks fail.

Every group's primary-execution interval includes zero under both estimators.
The positive rising-group total is descriptive and does not authorize selecting
the opposite rule. There is no multiple-comparison correction across groups.

## Coverage and execution limits

1,727 contracts have all six required two-sided hourly closes. The remaining
463 stay explicitly unclassified. This preserves 100 fill proxies that would
otherwise disappear from group accounting.

The complete execution accounting is 495 filled proxies, 401 orders without
sufficient matching volume, 1,095 contracts outside the ask band, 106 stale
quotes, 81 proposed orders crossing the bid, and 12 without quotes. These sum
to 2,190. Every original exclusion and order is retained in the report.

The rule assumes zero maker fees and uses an additional one-cent-per-fill
stress. It does not verify historical fees, order-book depth, queue priority,
partial fills, cancellations, intraday equity or available collateral.
The capture is retrospective and does not establish historical receipt times.

## Reproduction

Evidence root: `C:/Users/Test/PROJECTS/files/maker-transfer-20261014`.
`first.json` was produced by actual public acquisition and `replay.json` by a
separate offline CLI process. Both report files and both RunRecord sidecars are
byte-identical. Each report independently verifies 3,097 evidence paths.

Report SHA-256:
`5c9e4bbe911741b24f236ce6c7877804278ce15d189d2ad67a74e7eb104d6ee9`.

Implementation SHA-256:
`61732c1206c2c0c3cb9f3cd2edc4c0388c48f3feed2c01aeee4ad9cfc7a50e02`.

Protocol SHA-256:
`763fdf28497d32a4917838eb89b3dfec462dd3c4a396629a4114d2ee31f1247e`.

Plan SHA-256:
`e627f4d77921d9a4f8d1a34c7c9d2512afd4d41266992181c514a57b182710c6`.

Use the existing `scripts.market_research` CLI, `plan.json` in the evidence
root, the linked exact protocol bytes, `raw` as the archive, `--mode replay`
and `--workers 3`, with a new output path. The report retains the predeclared
one-cent-through and touch sensitivities; neither replaces this primary test.

The [validity audit](MARKET_VALIDITY_AUDIT.md) documents the two reproduced
shared-library defects, their repairs, unchanged original 2026 market results,
and the passing 3,847-test full check. No model or trading threshold was altered
in response to the 2025 results.
