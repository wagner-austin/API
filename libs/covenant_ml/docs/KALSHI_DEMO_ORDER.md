# Durable single-order demo trial

## Actual demo execution, October 13, 2026

The credentialed trial completed against Kalshi's demo exchange. Starting mock
cash was $50. One YES contract in `KXNHLSPREAD-26OCT13STLOTT-STL2` filled at $0.30,
with $0.0147 in fees, leaving $49.6853 cash and one contract held. Native order
`01a13626-9e58-743f-80e0-2c25e6edf733` and client UUID
`ee2a8948-0b17-42e8-a8db-4c84ed580e19` matched the journal and independent fills.

The first order-list read briefly returned no order after the fill acknowledgement.
Submission therefore exited with an unresolved reservation. An explicit
`reconcile` command subsequently verified the order, fill, position and cash;
it did not resend. A restarted `submit` command was then exercised and refused
with `DEMO-TRIAL-005` before network I/O. Another authenticated reconciliation
still found exactly the same one order and one fill.

Evidence is under
`C:/Users/Test/PROJECTS/files/market-edge-20261012/kalshi-demo-trial-20261013/`:
`trial.json`, `acknowledgement.json`, `execution.sqlite`,
`reconciliation-*.json`, `verified-status.json` and
`duplicate-submit-refused.log`. No private key is stored there. At this October 13
observation, the $0.3147 cost remained exposure awaiting the market outcome.
This is a completed demo order/fill/account lifecycle, not a production pilot.

Later observation: the [October 15 evidence refresh](MARKET_PROGRESS_20261015.md)
confirms the same order settled, paid $1 and realized $0.6853 in mock profit
after its $0.0147 fee. Exposure is zero. The pending-settlement observations
below describe the earlier October 13 checks, not the latest account state.

The settlement-aware command was subsequently run on that same trial at
17:12 UTC. Authenticated settlement pagination returned no settled positions;
the market remained active. The command reported `confirmed_settlements=0`,
retained $0.3147 exposure and did not submit another order. Its native capture is
`reconciliation-37fcfef7-f3fd-4a5a-8a20-3fc272011937.json` in the directory above.
At that October 13 check, an actual exchange payout had not yet occurred; local
final-outcome tests did not represent a completed remote settlement.

## Automatic demo observation

Windows task `API-KalshiDemo-Observe-20261013` runs the existing `reconcile`
command every 15 minutes and at boot, through October 20, 2026 at 10:20:38
Pacific. Its action contains no `submit` command. It runs under the operator's
account using S4U, normal priority, IgnoreNew concurrency and a two-minute
execution limit. Each invocation either records validated native evidence or
exits with the original failure; the scheduler exposes that exit code.

The task uses its own Poetry environment and 653 hash-verified source files under
`C:/Users/Test/PROJECTS/files/market-edge-20261012/demo-observer-runtime-20261013/`.
The loaded project modules were verified to resolve inside that snapshot. It
does not use the working tree or alter the frozen research campaign runtime.
`source-receipts.json` records the deployment file hashes.

An actual Task Scheduler launch completed at 17:19 UTC with exit code zero,
one order, one fill, zero settlements and $49.6853 cash. The trial directory
contains `observer-scheduler.xml`, `observer-verification.json` and the native
capture `reconciliation-4e6232f7-175a-4b09-8b15-6915139d200f.json`. This verifies
execution under the scheduled principal, not just registration. Observation
continues to validate credited history after settlement within its time window;
it does not select or submit another trade. Any nonzero scheduler result needs
inspection, and the local `status` timestamp identifies the last validated pass.

## Execution contract

Future submissions also retain [validated quote and timing evidence](DEMO_SUBMISSION_EVIDENCE.md).
The `evidence` action verifies those files locally. The completed October 13
trial predates this recorder; its missing quote cannot be backfilled.

`scripts.kalshi_demo_trial` connects the existing $1 order reservation policy to
Kalshi's demo Create Order V2 endpoint. A trial submits at most one one-contract
YES immediate-or-cancel order. It never selects the production origin and never
reports live readiness. This tests execution mechanics, not a profitable strategy.

The [credential setup](KALSHI_DEMO.md) and [risk journal](EXECUTION_REHEARSAL.md)
remain shared; signing, HTTP transport, pagination and accounting are not forked.

## Commands

Run from `libs/covenant_ml`. The root's parent must exist and the trial directory
must be new. Prices are integers in 1/10000 dollars: `3000` means $0.30.

```powershell
poetry run python -m scripts.kalshi_demo_trial `
  --key-id-file C:/Users/Test/.secrets/kalshi-demo/key-id.txt `
  --private-key-file C:/Users/Test/.secrets/kalshi-demo/private-key.pem `
  --root C:/evidence/demo-trial `
  prepare --ticker DEMO-CONTRACT --limit-price 3000
```

Use the same three path arguments with `submit`, `reconcile` or `status` instead
of `prepare`. Preparation checks the empty primary account, saves the credential
fingerprint and initial cash, creates a fresh intent UUID and five-minute expiry,
and initializes the existing SQLite journal. It places no order. The private key
and API key ID are never written into trial evidence.

`submit` verifies credentials, refuses a previous reservation, checks unchanged
cash and repeats authenticated empty-account preflight. It requires a matching
active contract, a future closing time, a two-sided order book, an executable
ask within the limit, at least one visible contract and no more than five seconds
between account preflight completion and the submission admission decision.
The transport checks intent expiry again immediately before sending.

The full $1 ceiling is reserved: contract limit price plus the remainder as fee
reserve. There is no implied estimate that fees will actually consume that
reserve. Actual fill cost and fees replace the reservation only after successful
reconciliation. The existing policy retains filled exposure until settlement.

## Write-ahead reservation and ambiguous responses

The journal commits the risk reservation before the HTTP POST. Competing callers
must pass the same transactional reservation reducer; a duplicate UUID is refused.
After reservation, this trial can never submit again, even after an HTTP error,
lost response, process restart, zero fill or terminal cancellation. A process
failure between reservation and sending is intentionally indistinguishable from
an uncertain send. It retains risk rather than assuming the exchange saw nothing.

`reconcile` makes GET requests only. It reads every current account order/fill page
using the shared cursor-cycle checks and requires exactly the manifest's terminal
IOC order. Duplicate/foreign fills, mismatched quantities, prices, times or fees
are refused. Summed fill acquisition costs and fees must equal native cumulative
order totals; the primary position quantity and cash change must also agree.
An available acknowledgement must name that same order. A missing order is an
unresolved result, never permission to resend.

The same command now also reads all primary-account settlement pages. A final
credit requires exactly one settlement matching the acquired YES quantity,
cost basis and fees, with no NO holding. Its event, result, timing and per-contract
payout must agree with the finalized public market. Native revenue is in cents;
the validator converts it exactly and refuses inconsistent payout arithmetic.
The unsettled position must be absent and cash must equal initial cash minus
acquisition cost and fees plus the credited revenue. Only then does the existing
journal record `account_settlement`, release exposure and report realized profit
or loss. This account evidence is distinct from the rehearsal's public-outcome
`settlement` record and makes no claim about changes to contract rules.

Repeated reconciliation is idempotent for the credited amounts. A changed or
disappeared settlement cannot replace history; an HTTP error, missing credit or
cash mismatch propagates without releasing pending exposure. Reports expose
`confirmed_settlements`, and a settled trial still cannot submit another order.

Evidence files include the manifest, acknowledgement when received, cumulative
journal and separate native reconciliation captures. `status` replays local
accounting without network traffic. Reports distinguish a reserved attempt from
a confirmed native order and always set `can_reserve=false` and `live_ready=false`.

## Scope and limitations

This intentionally permits one initial demo order, not an ongoing trading account.
Any returned prior orders or current positions block preparation. Account activity
outside this process can race sequential reads; there is no atomic exchange
snapshot. Other subaccounts, archived orders/positions, external account activity
and a live strategy admission policy are outside this trial. Settlement requires
the native account credit to remain in the live data set; archived or temporarily
inconsistent evidence is an explicit failure, never an inferred payout.

The credential fingerprint binds key material rather than independently proving
a stable exchange user ID. Local files are not a tamper-proof security boundary:
someone with write access can edit/copy a manifest or replace a journal. Do not
delete or alter the trial to retry an ambiguous submission.

Native account money may have six fractional digits. Values enter the existing
1/10000-dollar journal only when exactly representable; excess precision causes
an explicit failure and leaves risk reserved. Fill aggregation retains exact
microdollar arithmetic. No amount is rounded to force reconciliation to pass.

Tests execute real signing, HTTP sockets, SQLite, CLI files and subprocesses.
An actual socket disconnect after POST exercises retained reservation and
GET-only recovery. Repeated submission after restart is refused before network
I/O. These local tests are distinct from the credentialed demo evidence.
Final-outcome tests cover binary wins/losses, scalar payouts, exact revenue,
cash/position mismatches, provisional markets, disappeared credits, conflicting
journal events and restart replay through real HTTP sockets and SQLite.
The integrated `make check` passed 3,330 tests, strict typing and guards, with
100% coverage of 15,758 statements and 3,002 branches. Forty added tests cover
account settlement decoding, journal transitions and the HTTP settlement lifecycle.

The current official contracts are
[Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2),
[Get Orders](https://docs.kalshi.com/api-reference/orders/get-orders), and
[Get Fills](https://docs.kalshi.com/api-reference/portfolio/get-fills), checked
October 13, 2026. Demo fills are not evidence of production liquidity or returns.
Account credits follow
[Get Settlements](https://docs.kalshi.com/api-reference/portfolio/get-settlements)
and its native schema in the [OpenAPI specification](https://docs.kalshi.com/openapi.yaml),
checked the same day.

## October 15 batch follow-up

The single-order trial remains the settled baseline. A separate bounded batch
now reconciles all known account orders and shares the durable submission path.
Four new demo fills are recorded in [the batch validation report](DEMO_BATCH_VALIDATION_20261015.md).
The old single-order observer is disabled; its one-order account assumption no
longer holds. The replacement task's first invocation failed during a local DNS
outage. After recovery, a Task Scheduler launch exited 0, saved fresh account
evidence and admitted no fifth order. The report preserves both observations.
