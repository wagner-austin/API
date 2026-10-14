# Archived equity prices and disclosure availability

Board task: 24b6d678-a19d-4178-b195-76b8cbe5cce3.
Measured 2026-10-14 using public, read-only historical responses.

`markets/equity_records.py` defines an immutable `EquityPrice` and its strict
JSON encoder/decoder. `markets/equity_history.py` decodes the captured daily
chart schema and measures a future adjusted close-to-close return. These
inputs extend the accession-bound disclosure selection in
[DISCLOSURE_INPUTS.md](DISCLOSURE_INPUTS.md). They do not establish a stock edge.

## Price and time contract

The decoder requires exactly one equity result, the requested symbol and
currency, daily granularity, and an explicit null provider error. Timestamp,
close and adjusted-close arrays must be nonempty and equally sized. Every
price must be finite and positive; every timestamp must be an integer, unique
and strictly increasing. Nulls, mismatched symbols, provider errors and
ambiguous arrays propagate failures. No price is filled, substituted or
silently discarded. Other unused chart fields remain in the native archive.

The native timestamp means **session start**, not close. The receipt must be
at least 24 hours after every session start. This deliberately conservative
completion check rejects today's unfinished daily bar. It is not an exchange
calendar or certification that the source contains every scheduled session.

`split_adjusted_close` stores the provider's `close`;
`distribution_adjusted_close` stores its `adjclose`. The raw corporate-action
ledger is retained in the source response. Neither the split ratio nor the
dividend cash amount is applied again. Provider-adjusted levels are not
historical then-traded cash prices and must not be multiplied by historical
share counts without a separate consistent adjustment convention.

`adjusted_forward_return` requires a single symbol/currency and ordered bars.
It takes an explicit information-availability timestamp and uses the first
bar whose session starts **strictly after** that timestamp, entering at that
bar's close. Information arriving during a session therefore waits until the
next returned session's close. Exact equality at session start also skips
that session. This avoids assigning the daily opening timestamp to a price
that is only known later that day.

The exit is `holding_bars` later in the returned history. Missing entry or
exit coverage returns explicit `None`, never a zero return. The function
does not certify calendar completeness, execution, spreads, costs or fill
availability. Its result is `exit.adjusted_close / entry.adjusted_close - 1`.
It does not add dividends or splits a second time, annualize or claim profit.
The caller owns historically correct issuer-to-security mapping; passing a
filing timestamp alone does not prove that a ticker belongs to that issuer.

## Actual-source integration and replay

Evidence directory:
`C:/Users/Test/PROJECTS/files/stock-price-feasibility-20261014`.
The native capture hashes and acquisition limitations are documented in
[STOCK_PRICE_FEASIBILITY.md](STOCK_PRICE_FEASIBILITY.md).

Both full successful native responses are checked into test fixtures, with
unchanged JSON contents. The 2009 window contains 83 daily rows and the 2020
window 64. The latter contains the actual August 7 dividend and August 31
4:1 split. Tests execute production decoding, encoding and return arithmetic,
including exact timing boundaries and invalid/missing data.

The full-source integration also reads 2,265 actual filing records and 146
Assets observations from the disclosure archive. As of October 28, 2009 at
13:30 UTC, the selected September 26 balance is the original $53.851 billion
statement, accession `0001193125-09-214859`, accepted at
`2009-10-28T00:18:29.000Z`. Later revisions cannot replace it at that cutoff.
Apple's known issuer mapping is an explicit development assumption here.

| Development accounting case | Entry close | Exit close | Adjusted return |
| --- | --- | --- | ---: |
| Original filing, 20 observed bars | 2009-10-28 | 2009-11-25 | 0.06127839557173709 |
| Dividend boundary, one bar | 2020-08-06 | 2020-08-07 | -0.022735745119033757 |
| Split boundary, one bar | 2020-08-28 | 2020-08-31 | 0.03391222135170402 |

These are deliberately inspected development examples, not a trading strategy,
an untouched holdout, or evidence that selecting filings earns those returns.
The arithmetic excludes transaction costs and describes provider-adjusted
research outcomes rather than a dollar ledger or guaranteed fills.

Two independent processes wrote `equity-first.json` and `equity-replay.json`.
Their bytes are identical, SHA-256:
`7f94361f35aa8fc337305cec22d2eb4a76f7bb448ccada1693e161b96fe33b3a`.
Each report includes the implementation digest, arithmetic fingerprint, all
ten source/receipt hashes and a RunRecord sidecar. The production report
verifier rechecked the payload and every source. Both report zero orders
and `advance_to_funded_trading: false`.

Real HTTP integration serves the complete capture through the production
client and `ResponseArchive`, compares online and offline decoding, and
checks that corrupted evidence and an actual HTTP 404 fail. Offline replay
makes no additional request. No mock replaces the HTTP implementation.

The full `make check` passed: 3,656 tests, strict typing across source, tests
and scripts, all guards, and 100% coverage of 17,372 statements and 3,498
branches. It completed in 106 seconds of the 300-second budget. The requested
PowerShell command preserved the exit code and printed the last 100 log lines;
the full log is `make-check.log` in the evidence directory.

## Failure contract and remaining research

- `EQUITY-001`: nonfinite or nonpositive price.
- `EQUITY-002`: provider error, absent explicit error status, ambiguous arrays.
- `EQUITY-003`: symbol, currency, instrument or interval mismatch.
- `EQUITY-004`: empty, unequal, unordered, duplicate or unfinished history.
- `EQUITY-005`: mixed return identities or unordered return observations.
- `EQUITY-006`: floating-point ratio overflow or underflow.

Missing or wrongly typed fields raise the shared validators' `TypeError`;
invalid timestamp, identity and bounded-integer errors retain their existing
traceable `MARKET-*` codes. Exceptions propagate without recovery branches.

A stock experiment still needs a frozen signal, universe, mapping, evaluation
period, cost convention and treatment of delistings and missing outcomes.
Current survivor tickers cannot stand in for a historical universe. No
purchase or real-money deposit is needed to validate these accounting inputs.
