# Durable demo submission evidence

Board task: 8bc76fee-6333-4aa7-92bc-48d36415f019.

The demo submission workflow now saves the complete parsed native market and
order-book responses before sending an order. This includes every supplied
book level and contract-rule field, not just the executable top of book. These
are parsed response contents, not the exchange's raw HTTP bytes. Immutable
records retain serialized native objects and integer timestamps.

## Write ordering and failures

The service validates ordered market/book receipt times, decision time, active
market identity, quote depth, price limit, the five-second account preflight
window and the intent's expiry. It reserves risk in SQLite, exclusively creates
`submission-capture.json`, flushes and fsyncs it, then invokes the transport.
The transport checks the quote/preflight deadline and clock ordering again after
that persistence, before signing and posting. It cannot guarantee an unchanged
book at the exchange when the request arrives.

After a parsed response returns, the service exclusively creates and fsyncs
`acknowledgement.json` and `submission-timing.json`. The latter binds SHA256
digests of both the capture and acknowledgement. Native acknowledgement
validation then runs, so a structurally parsed but inconsistent acknowledgement
is retained for investigation. An HTTP error or disconnect before parsed return
does not create an acknowledgement or completed timing record.

File collisions and write failures propagate. A reserved trial remains reserved
if persistence or transport fails; restarting never resends it. A capture alone
does not prove that a request reached the exchange. Missing evidence remains
missing, and account reconciliation is the separate source of fill accounting.

## Read-only replay

Use the existing credential-file arguments and trial root with:

```text
python -m scripts.kalshi_demo_trial --key-id-file <path> --private-key-file <path> --root <trial> evidence
```

This action performs no network requests. It verifies the capture intent against
the credential-bound trial, the timing identity and start against the capture,
both content digests, quote admission constraints, and the native acknowledgement.
The result includes the verified capture, timing and acknowledgement.

`started_ms` marks the local submission workflow before journal/file persistence.
`returned_ms` is the local time after transport response parsing. Their difference
includes persistence, signing, network and parsing. It is not a pure network
round-trip measurement, matching-engine timestamp or estimate of live slippage.
Wall-clock reversal is rejected; this is not a monotonic high-resolution timer.

## Evidence and limits

The real socket, signing, SQLite, filesystem and CLI tests cover successful
submission and separate-process evidence replay, disconnect after receipt of the
order, exclusive-file collisions, stale quotes after persistence, malformed
acknowledgement identity and cross-record mutations. They verify single-send
behavior and retained risk. Quote and timing codecs also reject chronology,
expiry, price and digest failures. Controlled local test orders do not measure
actual exchange fill rates or establish a trading edge.

Full `make check` passed on October 15 UTC: 4,079 tests, strict mypy across
673 files, all guards, 100% of 18,789 statements and 3,900 branches, in 107
seconds. Log: `C:/Users/Test/PROJECTS/files/demo-evidence-final-check-20261015.log`.

The older completed demo order has no pre-send capture and cannot be upgraded
retroactively. Its actual mock settlement remains documented separately. The
frozen weather collector and its scoring protocol are unchanged; no new demo or
real-money order was sent to build this recorder.
