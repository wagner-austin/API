# Authenticated demo account preflight

This command adds authenticated account reads to the local execution research.
It cannot submit orders, reach a production origin, or grant execution readiness.
No funding is needed. The separate frozen forecast study is unchanged.

## Credentials and command

Create a separate [Kalshi demo account](https://docs.kalshi.com/getting_started/demo_env)
and a demo API key. Save the key identifier in a text file and the downloaded
private key in a PEM file outside the repository. Supply file paths, never key
contents, in chat. Production credentials do not belong in this workflow.

From `libs/covenant_ml`, run:

```powershell
poetry run python -m scripts.kalshi_demo_preflight `
  --key-id-file C:/private/kalshi-demo-key-id.txt `
  --private-key-file C:/private/kalshi-demo-private.pem `
  --output C:/evidence/kalshi-demo-preflight.json
```

Both credential files must already exist; the output parent must exist and the
output file must be new. There is no host override or live switch. The command
uses `https://external-api.demo.kalshi.co`, primary subaccount zero, ten-second
request timeouts, no automatic retries, and no redirect following. Failures
propagate with a nonzero process exit; no successful evidence report is emitted.

## Exact contract

`KalshiSigner` loads and validates Ed25519 or RSA-2048 private keys. RSA uses
PSS/MGF1/SHA-256 with a 32-byte salt. Both algorithms sign timestamp milliseconds,
uppercase HTTP verb and path without the query. The
[official authentication documentation](https://docs.kalshi.com/getting_started/api_keys)
was checked October 13, 2026. The report contains a SHA-256 fingerprint of the
key identifier and public key, never private key bytes or authentication headers.
This fingerprint identifies credentials, not an independently verified user ID.

The reader collects balance, recent orders, unsettled positions and balance
again. Every endpoint is explicitly scoped to primary subaccount zero; the
balance includes all exchange shards. It requires:

- Explicit empty order, market-position and event-position arrays.
- Explicit empty continuation cursors. A missing cursor is insufficient evidence,
  even though the positions schema lists that field as optional.
- Identical balances, portfolio values and last-update times across both reads.
- Zero reported portfolio value, nonnegative balances and a collection interval
  between zero and thirty seconds, without a future balance update time.
- Agreement between the integer cent balance and the whole-cent part of the
  fixed-point dollar balance. Up to six fractional digits are retained exactly
  as integer **microdollars**, distinct from the rehearsal's 1/10000-dollar units.

An account with returned order history is rejected, including canceled or
executed orders. This intentionally conservative initial preflight is not the
reconciliation algorithm for an account already used by the experiment. No
pagination is skipped and accepted: a nonempty cursor is an explicit failure.
The API's archived history and other subaccounts are outside this inspection.
Success therefore does not establish that the account has never traded.

The endpoint schemas are
[Get Balance](https://docs.kalshi.com/api-reference/portfolio/get-balance),
[Get Orders](https://docs.kalshi.com/api-reference/orders/get-orders), and
[Get Positions](https://docs.kalshi.com/api-reference/portfolio/get-positions).
Unknown response fields are irrelevant to this narrow contract; required fields
are validated immediately. Missing or changed required evidence fails visibly.

## Meaning of a result

A successful report has `mode=demo_account_preflight`, `orders_sent=0`,
`execution_ready=false`, and `live_ready=false`. Sequential reads are not an
atomic account snapshot: concurrent activity or delayed exchange reporting can
escape comparison. The result is an observation, never a trade-admission token.
The saved report also has no exchange attestation or anti-tampering guarantee.

Socket integration tests run the real HTTP client, JSON validation, cryptographic
signing and CLI file workflow. Signatures are independently verified with real
public keys, and altered messages fail verification. Tests cover HTTP failures,
incomplete evidence, exact balances, populated accounts and exclusive output.
These tests do not claim that a real demo credential or remote account was used.
The integrated `make check` passed 3,248 tests, strict typing and repository
guards, with 100% coverage of 15,468 statements and 2,936 branches in 140 seconds.
Forty-two tests specifically cover signing, demo preflight and its CLI.

The [single-order demo trial](KALSHI_DEMO_ORDER.md) now supplies credential binding,
persisted intent before submission, native order/fill/position/cash reconciliation,
fresh quote admission and GET-only ambiguous-outcome recovery. Its actual demo
order filled and reconciled on October 13, 2026. This preflight command itself
remains read-only and cannot admit a second trial after that account activity.

Before real funding, the remaining work includes a production account identity,
ongoing account and settlement reconciliation, current fee/rule admission and
an explicitly authorized production execution path. The
[local rehearsal contract](EXECUTION_REHEARSAL.md) remains the source for the
proposed $20/$1/$3/$5 limits. Neither this command nor that rehearsal establishes
a profitable trading edge.
