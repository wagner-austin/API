# Bounded execution rehearsal

`python -m scripts.execution_rehearsal` exercises the accounting for the proposed
$20 Kalshi experiment. **It is a local rehearsal, not a trading client.** It sends
no requests to an exchange, accepts no credentials and always reports
`live_ready: false` and `orders_sent: 0`. Its input files are operator-supplied
evidence, not authenticated account observations. Passing this rehearsal does
not authorize or establish readiness for real-money orders.

The existing frozen weather campaign is independent. This code does not change
its plans, fitted models, isolated runtime or scheduler.

## Fixed risk contract

All money uses integers in **1/10000 dollars**. Quantities in receipts use
1/100 contracts; proposed purchases use whole contracts. The initial supported
direction is buying YES with a limit price. There is no borrowing, short-sale,
sale-to-close, order amendment or portfolio netting path.

| Constraint | Fixed value |
| --- | --- |
| Experiment capital recorded in metadata | $20 |
| Maximum limit-price acquisition cost plus fee reserve per order | $1 |
| Maximum combined unresolved, resting and filled-position exposure | $3 |
| Gross realized losses plus remaining exposure | At most $5 |
| Intent validity | At most five minutes |

The $20 value identifies the experiment budget; it does not inspect a bank or
exchange balance. The stricter $3 and $5 limits govern reservations. A gain never
replenishes the cumulative loss allowance. Contracts are not treated as hedges
against one another. A second purchase of an already exposed contract is refused.

Reservation commits before the CLI prints the inspection-only order payload.
An unresolved reservation retains its entire cost and fee allowance and blocks
all new reservations. Time passing, a restart or an expired intent never releases
it. Only admissible receipt evidence can reconcile it. There is no automatic
retry, "assume rejected" operation or reset command.

A resting partial fill retains at least its original reservation, including the
remaining fee allowance. A terminal cancellation releases only the unfilled
portion: filled cost and actual fees remain exposed until final settlement.
Fractional settlement proceeds round down to the accounting unit. Gross losses
are summed per completed order, not offset by profitable orders.

Actual fees or fill costs exceeding the proposal remain recorded and latch a
permanent halt. Later favorable fills cannot erase that halt. Changed settlement
rules also halt further reservations. A manual halt permits reconciliation but
has no resume operation. These local controls cannot retroactively prevent an
exchange from charging more than the assumed fee reserve.

## Commands

Run from `libs/covenant_ml` in its locked Poetry environment:

```powershell
poetry run python -m scripts.execution_rehearsal --ledger ./rehearsal.sqlite init
poetry run python -m scripts.execution_rehearsal --ledger ./rehearsal.sqlite status
poetry run python -m scripts.execution_rehearsal --ledger ./rehearsal.sqlite reserve --intent ./intent.json
poetry run python -m scripts.execution_rehearsal --ledger ./rehearsal.sqlite receipt --order ./order.json
poetry run python -m scripts.execution_rehearsal --ledger ./rehearsal.sqlite settlement --client-order-id 00000000-0000-0000-0000-000000000001 --evidence ./settlement.json
poetry run python -m scripts.execution_rehearsal --ledger ./rehearsal.sqlite halt --reason "Experiment stopped"
```

`init` requires a new file in an existing directory. Repeating it fails without
changing the journal. Every other command requires an existing database and
refuses to create a replacement when the path is wrong.

The intent JSON requires `client_order_id` (canonical UUID), `ticker`,
`contracts`, `limit_price`, `fee_reserve`, `created_ms` and `expires_ms`.
Both timestamps are epoch milliseconds; the actual clock must be inside the
window at reservation. The fee reserve is an explicit assumption supplied by
the caller; this command does not calculate or verify the current fee schedule.

`receipt` reads the native Get Order response envelope, including its `order`
object. It requires current `outcome_side`, `book_side`, type, status, identity,
fixed-point counts, limit price, maker/taker costs and fees, and update time.
Receipt identity must match the reservation. Cumulative quantities, costs and
fees cannot regress; terminal evidence is immutable. An identical terminal
receipt is safe to import again without adding another fill. Reusing an
exchange order ID for a different reservation is rejected.

`settlement` reads one normalized `Settlement` object using the existing strict
settlement codec. It requires a terminal order and a finalized binary or scalar
payout with matching identity and consistent chronology. Pending outcomes never
release exposure. A finalized receipt cannot subsequently be changed.

The inspection payload uses the current
[Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2)
schema: a YES `bid`, fixed-point price and quantity, immediate-or-cancel,
`taker_at_cross` self-trade prevention and cancellation on pause. The native
receipt adapter follows
[Get Order](https://docs.kalshi.com/api-reference/orders/get-order).
These schema references were checked October 13, 2026. No HTTP submission is
implemented by this command.

## Persistence and verification

SQLite `BEGIN IMMEDIATE` serializes each mutation. Every operation replays and
validates the full journal, applies the same pure policy reducer, appends its
event and verifies the resulting state inside the transaction. A failed write
rolls back. Another writer causes an explicit lock failure instead of a retry.

Metadata binds the schema, rehearsal mode and fixed ceilings. Sequential event
numbers and a SHA-256 chain detect changed or missing interior records. This is
an integrity check, not an authenticated audit service: someone with write access
can rebuild a chain, remove its tail, copy or replace a database. Starting another
database would start another local rehearsal. A future live executor must bind
one journal to one account and reconcile the account independently.

Tests run actual codecs, reducers, SQLite transactions and CLI subprocesses.
They exercise exact monetary boundaries, partial fills, scalar payouts,
unresolved reservations, duplicate identities, corruption, competing writers,
database-trigger readback corruption, rollback, restart persistence and actual
nonzero process exits. The existing Makefile, strict mypy, shared guard and
pytest/xdist coverage harness cover these files; no parallel harness was added.
The integrated `make check` passes 3,206 tests, strict mypy across 531 files,
all guards and 100% coverage of 15,335 statements and 2,908 branches. The first
complete check took 137 seconds. Sixty-one tests specifically exercise the new
rehearsal records, policy, journal and CLI.

## Prerequisites for a live pilot

The remaining execution work is an authenticated exchange adapter with complete
account/fill/balance reconciliation, account-bound journal identity, current
fee and market-rule validation, and a submission path that persists intent
before sending and resolves ambiguous responses without resubmission. Fresh
quotes and available quantity must be checked at submission. Admission must be
based on the account, including activity outside this process.

That path needs end-to-end tests against Kalshi's
[separate demo environment](https://docs.kalshi.com/getting_started/demo_env)
before a separately authorized live experiment. A credential-free local test
does not prove demo connectivity, actual queue behavior, available liquidity or
profitable execution. The order proposal is a measurement input, not a trading
signal or evidence of an edge.
