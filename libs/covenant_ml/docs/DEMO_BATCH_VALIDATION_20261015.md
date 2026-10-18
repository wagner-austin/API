# Bounded demo batch execution validation

Measured October 15, 2026. This is execution validation with mock funds, not a
forecasting experiment or evidence of profitable trading.

## Frozen protocol

The batch admits at most 20 attempts on distinct NHL spread contracts selected
by available YES asks, then ticker. No model selected these contracts. A scan
of 10,000 ordinary demo markets found no ask at or below the initially considered
five-cent limit; 20 fresh NHL books offered at least one contract at 55 cents or
less. The price was fixed at 55 cents before the plan was written.

Every attempt is one YES contract, immediate-or-cancel. The existing limits
remain $1 reserved per order, $3 total exposure and $5 cumulative gross loss.
Realized profits do not replenish that loss allowance. The plan permits new
reservations for 24 hours, from epoch milliseconds 1792042308590 through
1792128708590. A cap is not a promise of 20 submissions or fills.

The baseline is the previously reconciled, settled single-order demo trial.
Its original cash was $50 and its net realized mock profit was $0.6853.
The batch reports its own realized result separately from that baseline.
Credential ownership binds one immutable plan to one account and directory.
Recreating the service or invoking a fresh process does not reset the limits.

## Exchange observations

Four new orders filled one contract each and reconciled against authenticated
orders, fills, positions, settlements and stable account balances. Prices were
$0.52, $0.52, $0.52 and $0.55; fees were $0.0175, $0.0175, $0.0175 and $0.0174.
Total new exposure is $2.1799 and authenticated remaining cash is $48.5054.
All four remained unsettled at the recorded observation. Batch realized profit
was zero. The account also contains the earlier settled baseline order.

The first three immediate post-submission collections rejected changing cash
with DEMO-BATCH-002. The fourth rejected an incomplete order identity set with
DEMO-BATCH-001. Later, explicit read-only reconciliation accepted each complete
snapshot. Persisted reservations prevented a second submission of those intents.
The failed raw snapshots were not retained: their exception logs identify the
validation failure, while accepted native snapshots and submission evidence
are saved. This limits retrospective diagnosis of the rejected responses.

Saved admission state reports can_reserve=false: $2.1799 exposure plus a new
$1 reservation exceeds $3. A subsequent advance invocation failed at DNS
resolution before completing its preflight; that failure is not evidence of a
successful exchange-connected risk-gate run. The controlled integration tests
exercise that refusal through the actual production service.

After DNS recovered, the scheduled advance action completed successfully at
2026-10-15T07:04:04Z. It saved a fresh authenticated account snapshot, retained
exactly four attempt directories and five total account orders including the
baseline, and submitted no fifth batch order. This verifies the risk-gate refusal
through the scheduled exchange-connected path as well.

## Controlled validation

The full package make check passed 4,254 tests, strict typing across 694 files,
all guards, and 100% coverage of 19,218 statements and 3,992 branches. It took
119 seconds, under the 300-second package budget. Tests use real HTTP sockets,
signatures, files, SQLite and subprocesses with controlled exchange responses.

One test executes 20 sequential local order lifecycles, including full, partial
and zero fills. Another reconstructs 100 consumed attempts and proves no further
POST is admitted. Those are not 20 or 100 exchange trades. Cases also cover
exact fractional-cent fees, unknown account activity, inconsistent cash,
positions and credits, atomic journal refusal, connection loss after submission,
missing or altered evidence, process death, restart, expiry and gross loss caps.

The separate earlier 100-scenario execution matrix remains documented in
DEMO_EXECUTION_MATRIX.md. Neither suite measures exchange liquidity or skill.

## Evidence and reproduction

Code and tests are at 8a20b584a936bb5aae3661773c1c74d9b96a8639 in the API repo.
The root below is C:/Users/Test/PROJECTS/files/market-edge-20261012/.

- kalshi-demo-batch-20261015/plan.json: frozen account, limits and ticker sequence,
  SHA256 e1046ea3a9a32d4520fb541b8ad7ee45b5d1263a56fe283bcaba4a357265bc66.
- kalshi-demo-batch-20261015/execution.sqlite: atomic account journal.
- kalshi-demo-batch-20261015/attempts/001 through 004: intent, quote capture,
  acknowledgement and submission timing, with digests checked on reconciliation.
- kalshi-demo-batch-20261015/account-*.json: accepted native account snapshots.
- kalshi-demo-batch-20261015/reconcile-004.json: four confirmed fills and risk state.
  SHA256 c00a021a4560609837a474c86fcb590bbea13d583c109008ebdd7a5d594c2488.
  Native snapshot account-7ec63f4e-5537-4791-85bc-2f6efa4d9eec.json records
  $48.5054 cash at epoch milliseconds 1792047403290.
- demo-batch-inventory-20261015/frozen-tickers.json: selected universe, SHA256
  697ba93489c8bacb92d0db5161285ee968c46918224f89878f5888dbe32b313f.
- demo-batch-runtime-20261015/source-receipts.json: 722 isolated source files,
  verified against their recorded hashes after installation.
- C:/Users/Test/PROJECTS/files/demo-batch-final-check-20261015.log: full local check.

Run from the isolated runtime's libs/covenant_ml directory, using its own Python:

```powershell
.venv/Scripts/python.exe -m scripts.kalshi_demo_batch `
  --key-id-file C:/Users/Test/.secrets/kalshi-demo/key-id.txt `
  --private-key-file C:/Users/Test/.secrets/kalshi-demo/private-key.pem `
  --root C:/Users/Test/PROJECTS/files/market-edge-20261012/kalshi-demo-batch-20261015 `
  reconcile
```

status reads locally; reconcile only reads the exchange; advance first reconciles
the full account and may submit one newly reserved intent. It never resends a
reserved intent. After expiry it only reconciles. Do not delete ownership or
state to create a fresh budget. Credentials are private files, not documentation.

## Monitoring

The prior single-order observer is disabled because its account contract permits
only that original order. API-KalshiDemo-Batch-20261015 is registered to call the
tested advance CLI every 15 minutes for seven days from October 15, with a
two-minute execution limit and overlapping instances ignored. Its fixed plan
expires after 24 hours, so subsequent invocations can only reconcile. The task
definition is saved in kalshi-demo-batch-20261015/scheduled-task.xml.

The first scheduler invocation exited 1 while local DNS resolution was failing.
After recovery, an explicit Task Scheduler launch exited 0 and produced native
snapshot account-4154fac5-83bd-4e81-bffb-cf29c7429672.json, SHA256
07e9d290040ae489d0fcf773c7509df397768f3b23575dab737c2f67614c7a6e.
Its completion timestamp is 1792047845285 and cash remains $48.5054. The saved
scheduler-verification.json ties the task result, snapshot, order count and
attempt count together. This verifies the scheduled action; future timed ticks
and settlement observations remain prospective.

## Final status (2026-10-18)

The plan stopped admitting reservations at 2026-10-16T05:31:48Z. All four batch
orders had settled by 2026-10-16T03:58:53Z. NJ2 paid $1; DET2, FLA2 and WPG2 paid
nothing. The status observed at 2026-10-18T11:42:58Z reads expired, zero
exposure and no issues, with 16 of the 20 permitted attempts started and 4
reserved, confirmed, filled and settled. The batch's net realized mock result is
-$1.1799: $1 revenue against $2.11 cost and $0.0699 fees. Including the
baseline order, the account's net realized result is -$0.4946, gross realized
loss is $1.6424 and $3.3576 of the $5 loss allowance remains. Because the
expired plan could only reconcile, API-KalshiDemo-Batch-20261015 was disabled
at 2026-10-18T11:55:53Z, ahead of the Linux reimage of its host austinpc (MCPs
board task e2ffed0d), instead of running to October 22. The status is saved as
kalshi-demo-batch-20261015-final-status-20261018.json, SHA256
0dc063874285e61ac36cb78db9fd5607075dea105c114959cb1eb03ea98d44e7.

## Interpretation

Four new demo fills validate a multi-order exchange path and observed recovery
from inconsistent account views. They do not establish an edge, an independent
sample of returns, production fill rates or a reason to add real funds. Several
contracts share sports events. The four settled outcomes are one batch, not a
sample from which to estimate returns.
