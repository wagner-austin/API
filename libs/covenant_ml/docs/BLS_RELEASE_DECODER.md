# Native BLS CPI release boundary

Board task: 66a6294a-7333-4158-9945-c59b1b0c2b45.

`bls_release.decode_bls_release` now validates the actual BLS summary or dated
release HTML and returns an immutable `CPIRelease`. The record contains the
required target month, UTC publication timestamp, USDL release identifier and
directly published one-decimal annual all-items CPI-U percentage. Its internal
JSON encoder and decoder enforce identity, finite decimal precision and a
publication month after the target.

The HTML collector uses the standard library parser with its real typed
callback signatures. It gathers release PRE text and the explicitly identified
Table A. Unrelated page content is not interpreted as data. Missing, duplicate,
unclosed or nested target structure raises a named error. Relevant duplicate
attributes are refused; a second Table A is not silently selected or merged.

The release header must uniquely identify the requested month and the native
8:30 a.m. Eastern publication date, including the correct weekday. Eastern
time is resolved through `America/New_York`, rather than assuming a fixed
UTC offset. The USDL year must match publication. Receipt before publication
fails. The annual table caption, all-items row, exact row/column references,
year/month header and signed one-decimal value must all validate. There is
no paragraph-number fallback, index-level calculation or core-index substitute.

Errors are divided into `CPI-001` publication identity, `CPI-002` percentage
precision, `CPI-003` HTML structure, `CPI-004` field cardinality, `CPI-005`
native header, `CPI-006` annual field semantics and `CPI-007` early receipt.
Shared malformed field and calendar exceptions also propagate. The parser
performs no network request; acquisition owns the original bytes and receipt.

## Actual native evidence

Both original files from `BLS_RELEASE_SOURCE.md` were read through the
production parser: the complete 90,388-byte current summary and the complete
1,372,789-byte dated archive, each using its own actual recorded receipt.
Both decode to:

```json
{
  "target_month": "2026-08",
  "published_at": "2026-09-11T12:30:00+00:00",
  "release_id": "USDL-26-1496",
  "annual_change": 3.4
}
```

Two independent offline processes persisted matching source-bound outputs
`parser-first.json` and `parser-replay.json` under
`C:/Users/Test/PROJECTS/files/bls-release-feasibility-20261014`. Each verified
14 evidence paths including prior source reports and RunRecords. Both reports
have SHA256:

`8f587d025d4d96918e6076f4a8ff4de5488bfb6fd2a8910ab28d0b34f5d93d51`

The checked-in HTML test fixture preserves the complete summary text. Its
128 CRLF pairs were normalized to LF by patch application; equality with the
original decoded text was verified. Raw original bytes remain unchanged in
the evidence archive, and the fixture is not represented as byte-identical
to that raw response. Its LF-byte SHA256 is
`e0d247045f73a0934879066f05d1cde57a7ca950d4bf0095865a8828d41cdd5f`.

## Tests and interpretation

Thirty-six focused tests execute the real codecs and native parser, including
the full actual summary and explicit malformed-input mutations. They cover
target mismatch, duplicate fields/tables/attributes, incomplete structure,
wrong weekdays, unsupported header times, early receipt, timezone
normalization, changed CPI identity, wrong annual headers, numeric precision,
May's native abbreviation and negative reported values. The mutation cases
are validation tests, not additional economic observations.

The focused run covers 100% of all 118 new statements and 40 branches. Strict
typing uses typed regular-expression matches rather than untyped findall
results, without casts, ignores or aliases. Shared package hooks remain the
I/O boundary for the acquisition workflow.

Full `make check` also passed: **3,794 tests**, strict mypy, all guards,
and **100% of 17,779 statements and 3,628 branches** in 160/300 seconds.

These outputs describe August only. The September result remains null, and
the parser explicitly refuses to treat this August document as September.
The first-observed-byte and correction limits in `BLS_RELEASE_SOURCE.md`
still apply. This parser does not itself capture future releases, schedule
acquisition, score native settlements or demonstrate a trading edge.
