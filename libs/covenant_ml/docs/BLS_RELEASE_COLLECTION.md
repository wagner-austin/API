# Official release capture and replay

The official CPI release command now acquires the native BLS summary, validates
its target month and scheduled publication, and writes a report with a RunRecord
bound to the exact captured response. Offline replay reads that envelope without
requesting the current page again. This is release observation, not exchange
settlement, first-publication attestation, a fill, or a profitable strategy result.

## Observed public execution

On 2026-10-14 at 08:46:51.433093 UTC, the production command acquired
`https://www.bls.gov/news.release/cpi.nr0.htm?`. The validated document was the
August 2026 release, published September 11 at 12:30 UTC, release identifier
`USDL-26-1496`, with annual CPI-U change of 3.4 percent. It cannot supply the
September outcome. The September frozen predictions remain unchanged and unscored.

Evidence root:
`C:/Users/Test/PROJECTS/files/bls-release-feasibility-20261014`.

- `cli-capture/`: immutable native response envelope, including decoded text,
  actual local receipt and verified SHA-256.
- `cli-first.json` and `cli-replay.json`: capture and separate-process replay
  reports, with byte-identical reports and RunRecord sidecars.
- Report SHA-256:
  `82bab1f7d354b64634e014ee6c445013fdb0dc112168c17dfff06d304e1cb814`.
- Implementation SHA-256:
  `28068575f121f85723db415680ca2ebc5c30112d67ef0de6eae81fc33ed6bad2`.
- Three verified paths: report, sidecar and source envelope. Zero orders.

The first live request returned 403 with the HTTP library's default client
identification. A transparent `covenant-market-research/1.0 (public CPI release
observation)` User-Agent succeeded. The collector now sends this fixed research
identification explicitly. The failed request produced no successful capture or
report. There is no retry, browser impersonation or replacement source.

## Command contract

From `libs/covenant_ml`, using a deployment whose implementation hash has already
been measured and declared:

```text
python -m scripts.cpi_release --target-month 2026-08
  --published-at 2026-09-11T12:30:00Z
  --archive C:/Users/Test/PROJECTS/files/bls-release-feasibility-20261014/cli-capture
  --out <new-report-path> --mode replay
  --implementation-sha256 28068575f121f85723db415680ca2ebc5c30112d67ef0de6eae81fc33ed6bad2
```

Line breaks above are for readability. Capture requires a new archive directory;
both modes require new report and sidecar destinations. `--source-url` permits an
explicit test origin; its default is `https://www.bls.gov`. The document path is
fixed. A different implementation must be declared deliberately, and will be
recorded in its new report. Never overwrite or relabel the original observation.

`ResponseArchive.capture_response` owns native acquisition, pacing, actual receipt
and exclusive persistence. Its JSON consumer continues to validate JSON at its
boundary. The release collector validates HTML using the existing strict parser.
Both formats share the same network and storage implementation. Dependencies come
from the existing markets `_test_hooks` factory, clock and pacing seams.

Errors propagate with explicit meanings:

- `CPI-008`: the observed publication differs from the declared publication.
- `CPI-009`: acquisition attempted to reuse an observation directory.
- `CPI-010`: acquisition began before publication, or its receipt clock regressed.
- `CPI-011`: implementation bytes differ from the declared deployment.
- `CPI-012`: the report or RunRecord already exists.
- Existing native parser, archive identity/hash and transport errors propagate.

A successful HTTP response is archived before native validation. A stale or
malformed response remains available for diagnosis, but cannot produce a completed
release report. Replaying a captured document validates it again against the
explicit target and publication. HTTP failures create no successful envelope.
The envelope preserves decoded response text; its digest is not a claim about
wire-level compression or original byte encoding.

## Validation and remaining integration

Twelve additional tests exercise actual local sockets and subprocesses: complete
native HTML, byte-identical replay, request identification, wrong target and
publication, output collisions, changed evidence, premature acquisition, clock
regression, missing endpoints, disconnected sockets and implementation drift.
The native HTML is the complete previously captured official document.

Full `make check` passed 3,806 tests, all guards and strict typing, with 100%
of 17,853 statements and 3,648 branches in 133/300 seconds. The final log is
`make-check-collection-final.log` under the evidence root. This check includes
the explicit research-client identification verified in the successful live run.

The prospective scorer must still join an observed September release to all 21
frozen contracts, verify native final settlement and unchanged rules, and score
the six original two-sided books while reporting the 15 missing sides. One
release remains one dependent event. This command does not infer settlements,
change forecasts or authorize trading. The separate
[scheduled deployment](BLS_RELEASE_SCHEDULE.md) records the subsequently installed
September acquisition and its real scheduled diagnostic.
