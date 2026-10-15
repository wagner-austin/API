# One hundred controlled execution lifecycles

Board task: 2ebf8a55-8d84-4383-b3a5-1d5ea9f2f503.
Measured October 15, 2026 UTC.

The new execution matrix passed all 100 distinct scenarios against a real
loopback HTTP server, using the production signer, transport, decoders,
submission recorder, SQLite journal and reconciliation service. These are
controlled local tests, not 100 orders on Kalshi's demo exchange. No additional
exchange order or real-money order was submitted by this test run.

## What varied

The Cartesian product contains five purchase prices ($0.01, $0.10, $0.30,
$0.50 and $0.98), four filled quantities (0.25, 0.50, 0.75 and 1.00 contracts),
and five settlement values ($0, $0.20, $0.40, $0.60 and $1 per contract).
Every submitted order requests one contract. Partial fills produce terminal
cancellations for the unfilled remainder; full fills produce executed orders.
The native fill evidence spans two pages and has a total fee of $0.0147.
That fee is an accounting input, not a claimed exchange fee schedule at each
test price. Scenario outcomes and liquidity are controlled test inputs.

Every scenario checks exact acquired exposure, native filled quantity,
pagination, recorded submission evidence, settlement proceeds, realized
profit and gross loss. Expectations use independent integer arithmetic.
It then reconstructs the service, replays persisted state and submission
evidence, repeats reconciliation, and proves that another submission fails
without a second HTTP POST. Failed and successful reconciliations are checked
against the actual journal, not against a mocked repository.

## Deliberate failures

Before accepting settlement, each scenario supplies an independent fill fee
one microdollar above the native order total. Reconciliation must raise
DEMO-TRIAL-003 and preserve the exact prior journal state. It separately
supplies an incorrect account cash balance; that must raise DEMO-TRIAL-007
and retain the filled exposure. Only a later complete, consistent observation
can record the credit. All 200 injected accounting disagreements were rejected.

The existing suite separately exercises zero-fill cancellation, actual socket
disconnects, delayed order visibility, stale quotes, credential mismatch and
evidence-file corruption. The new matrix does not replace those cases.

## Reproduction

From `libs/covenant_ml`:

```text
.venv/Scripts/python.exe -m pytest tests/markets/test_demo_execution_matrix.py -n 0 -q --tb=short
```

The targeted run passed 100 tests in 50.17 seconds on the local machine.
Full `make check` passed 4,192 tests, strict typing across 679 files and all
guards, with 100% coverage of 18,874 statements and 3,924 branches in 106 seconds.
The package-level check log is
`C:/Users/Test/PROJECTS/files/demo-execution-matrix-rebased-check-20261015.log`.

## What a larger exchange demonstration still requires

The current production demo executor intentionally admits one order only and
requires an empty account order history before preparation. The account already
contains the earlier completed demo order. Simply looping the existing command
cannot execute twenty more orders; deleting its journal would lose evidence
and would not satisfy account admission.

A bounded exchange batch needs persistent aggregate account reconciliation,
known historical orders, distinct client IDs, an attempt cap, cumulative cash
and exposure limits, and a halt on unexplained orders, fees or balances.
Those capabilities must be implemented and tested before the exchange batch.
Twenty observed submissions would be an initial integration sample, with fills,
cancellations and rejected attempts reported separately. A larger count would
not establish completeness or reliability for unobserved failure modes.

Controlled tests validate specified code behavior. Demo exchange observations
can additionally reveal actual endpoint semantics, visibility delays and demo
fill behavior. Neither alone establishes production liquidity or a predictive
edge, and repeating the same successful path does not replace failure tests.
