# Accession-bound financial inputs

Board task: ea2386f0-b610-43b7-9857-302a85fc9927.

`markets.disclosure_records` defines immutable `Disclosure` and `FinancialFact`
records with internal encoders and validating decoders. `disclosure_history`
decodes native submission columns and one explicitly named company-facts
concept/unit, then selects a revision strictly before a decision timestamp.
These modules are pure; acquisition remains the shared `ResponseArchive` with
the existing package I/O hooks. They do not train a stock model or place orders.

## Native boundaries

Call `decode_submission_columns` on the explicitly selected
`filings.recent` object for a current submissions response, or on the root
of a named historical submissions response. Pass the issuer identity established
by the requested document. The decoder requires four aligned arrays:
`accessionNumber`, `form`, `filingDate`, `acceptanceDateTime`.
It rejects unequal lengths and duplicate accessions. Acceptance timestamps
must carry a timezone; a filing date never substitutes for an acceptance time.

`decode_company_facts(document, cik, taxonomy, tag, unit)` checks the native
issuer against the requested issuer and requires the exact taxonomy, tag and
unit. Missing tags or units fail; there is no alternative-tag selection or
currency conversion. An absent native start means an instantaneous balance;
an explicit null native start is malformed. Normalized facts always include
the start field, using null only for that instantaneous case. Numeric values
are finite floating-point research measurements, not exact-currency accounting.
Periods require start <= end <= filed. Missing and invalid values are errors.

## Selection contract

`select_disclosed_fact` takes facts for one issuer/concept/unit, complete
disclosure metadata, an exact period start/end, and an aware decision cutoff.
It validates every supplied fact's issuer/accession link and agreement on
form and filed date before selection. Callers must load required historical
submission files; an unlinked fact is an error, even if it would be excluded
by the requested period or cutoff. This makes incomplete metadata visible.

Duplicate accession/period facts, duplicate filing identities, mixed concepts
or units, and tied eligible acceptance timestamps fail explicitly. Valid
observations for other periods and observations accepted at or after the
cutoff are ineligible. The result is the latest eligible fact paired with its
disclosure, or an explicit None when none was accepted before the cutoff. It never chooses a
nearby period, invents an acceptance time or replaces missing evidence.

Acceptance is not a public-availability timestamp. The
[SEC webmaster documentation](https://www.sec.gov/files/about/webmaster-faq.htm)
states that dissemination lags acceptance, with no guaranteed maximum delay
or field recording the first public availability. This selector enforces
accession and acceptance chronology only. Historical acceptance-based
experiments must label their dissemination assumptions; prospective decisions
must occur after the captured source receipt and completed feature calculation.
Today's company-facts response also does not establish when that structured
representation first became available. The one-second boundary examples below
test selection arithmetic, not an executable one-second filing strategy.

| Code | Failure |
| --- | --- |
| DISCLOSURE-001 | Invalid native accession spelling |
| DISCLOSURE-002 | Missing normalized start, nonfinite value or invalid period chronology |
| DISCLOSURE-003 | Unequal native submission columns or duplicate accessions |
| DISCLOSURE-004 | Company-facts issuer differs from request |
| DISCLOSURE-005 | Mixed fact identity or duplicate filing metadata |
| DISCLOSURE-006 | Missing issuer-matched accession or inconsistent form/filed date |
| DISCLOSURE-007 | Duplicate accession/period observation or tied eligible acceptance times |
| DISCLOSURE-008 | Requested duration starts after its end |

Shared validators retain their existing traceable errors for malformed JSON,
dates, timestamps, bounded issuer identifiers and blank required strings.

## Real-source verification

Captured evidence is in
`C:/Users/Test/PROJECTS/files/stock-disclosure-feasibility-20261014`.
The current and additional historical Apple submissions decoded into 2,265
disclosures; the actual `us-gaap:Assets`/USD history decoded into 146 facts.
Every record passed encode/decode round-trip validation. The older submissions
source SHA-256 is
`4995251b37e05f59a66e0d56fbfcd326a6c0a8736cb5f3d5917a1862fd82c4d5`;
the current and company-facts hashes are in `STOCK_DISCLOSURE_FEASIBILITY.md`.

The exact-period selection for 2009-09-26 produced:

| Decision cutoff, UTC | Selected accession | Assets, USD |
| --- | --- | ---: |
| 2009-10-28 00:18:29 | None: acceptance equals cutoff | unavailable |
| 2009-10-28 00:18:30 | 0001193125-09-214859 | 53,851,000,000 |
| 2010-01-26 02:23:58 | 0001193125-10-012085 | 47,501,000,000 |
| 2010-01-26 02:25:59 | 0001193125-10-012091 | 47,501,000,000 |

Separate offline processes produced byte-identical `disclosure-first.json`
and `disclosure-replay.json`, SHA-256
`bc3ece517eb182b00709693be778ce2feaecfdde123b85fa3369723ca6a8af67`.
Their RunRecord and six original response/receipt source hashes were verified.

`tests/markets/fixtures/disclosure_apple_assets_2009.json` is a projection of
the captured source responses: three accessions, their four native submission
columns and their actual September 2009 Assets observations. It does not
invent acceptance times or revised values. Unit tests exercise those exact
revision boundaries and malformed-input failures. Integration tests serve
the native fixture over a real local HTTP server, run the production client
and archive, replay without another request, and verify both corrupted archive
failure and a real HTTP 404 without creating substitute evidence.

The final full `make check` passed: 3,628 tests, strict typing over source,
tests and scripts, all guards, and 100% coverage of 17,296 statements and
3,472 branches. The complete check took 107 seconds against its 300-second
budget. The initial guard rejection of a non-null assertion was corrected
to compare the complete expected fact and filing. Evidence log:
`stock-disclosure-feasibility-20261014/make-check-final.log`.

This establishes a reusable input boundary. A tradable historical security
mapping, corporate-action-aware return panel, frozen signal and chronological
out-of-sample evaluation remain necessary before any stock-market edge claim.
