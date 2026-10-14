# Twenty-issuer disclosure coverage measurement

Board task: cff6c385-d5f7-4018-917e-fafc315cce7d.
Panel declared at 2026-10-14T05:52:10.286259Z, before its source acquisition.
Evidence: `C:/Users/Test/PROJECTS/files/stock-panel-20261014`.

Actual submissions and company-facts responses were captured for all twenty
named nonfinancial issuers, with forty successful requests and forty receipt
files. This is a purposive current-company development panel, not a random
sample or historical investable universe. No prices, forward returns, stock
rankings or orders were requested in this measurement.

## Exact matching rule

For current liquidity, choose the latest reported period among 10-K, 10-K/A,
10-Q and 10-Q/A submissions, breaking period ties by acceptance timestamp.
Require exactly one USD instantaneous fact for each of `AssetsCurrent` and
`LiabilitiesCurrent` with that exact accession and period end.

For annual profitability inputs, choose the latest annual period among 10-K
and 10-K/A submissions, using the same tie rule. Require exactly one USD
instantaneous `Assets` observation and one `NetIncomeLoss` duration observation
ending on that period, both in the exact selected accession. The annual
duration must span 300 through 400 days. This is an explicitly coarse annual
duration constraint for this coverage probe, not a fiscal-calendar decoder.

All issuers passed native issuer and current ticker association checks.
Production disclosure decoders validated the facts; no alternative concept,
unit, filing or period supplied a missing observation. The table's check means
one exact observation for each of the two measures, not a profitability result.

| Symbol | Current period | Liquidity pair | Annual period | Annual pair |
| --- | --- | --- | --- | --- |
| AAPL | 2026-06-27 | matched | 2025-09-27 | matched |
| MSFT | 2026-06-30 | matched | 2026-06-30 | matched |
| GOOGL | 2026-06-30 | matched | 2025-12-31 | matched |
| AMZN | 2026-06-30 | matched | 2025-12-31 | matched |
| META | 2026-06-30 | matched | 2025-12-31 | matched |
| NVDA | 2026-07-26 | matched | 2026-01-25 | matched |
| ADBE | 2026-08-28 | matched | 2025-11-28 | matched |
| CRM | 2026-07-31 | matched | 2026-01-31 | matched |
| CSCO | 2026-07-25 | matched | 2026-07-25 | matched |
| ORCL | 2026-08-31 | matched | 2026-05-31 | matched |
| INTC | 2026-06-27 | matched | 2025-12-27 | matched |
| AMD | 2026-06-27 | matched | 2025-12-27 | absent |
| IBM | 2026-06-30 | matched | 2025-12-31 | matched |
| WMT | 2026-07-31 | matched | 2026-01-31 | matched |
| COST | 2026-08-30 | matched | 2026-08-30 | matched |
| TGT | 2026-08-01 | matched | 2026-01-31 | matched |
| PG | 2026-06-30 | matched | 2026-06-30 | matched |
| KO | 2026-07-03 | absent | 2025-12-31 | matched |
| PEP | 2026-09-05 | matched | 2025-12-27 | matched |
| JNJ | 2026-06-28 | matched | 2025-12-28 | matched |

The result is 76 of 80 exact fields: 19 liquidity pairs, 19 annual pairs and
18 issuers with all four inputs. Both exceptional pairs have zero matching
facts, not multiple competing facts. Neither exception was replaced with zero
or an older period. Matching does not by itself validate a positive ratio
denominator or economic comparability across industries and fiscal years.

## Traced exceptions

**AMD:** annual accession `0000002488-26-000021` is a 10-K/A. Its
[explanatory note](https://ir.amd.com/financial-information/sec-filings/content/0000002488-26-000021/amd-20251227.htm)
limits the amendment to correcting transposed selling-price and shipment
changes in management discussion and refers readers to the original filing.
The captured company-facts response contains neither requested financial
observation under this amendment's accession. This establishes a document
composition issue, not the absence of AMD financial statements. A reviewed
original-plus-amendment model could establish the applicable statements;
the present probe does not silently substitute the original accession.
The full issuer-hosted amendment and its receipt were captured, body SHA-256
`ac4a7d7a4ca0bb235f1a764718b4dcbeda6be248d843a33d966423891767e2cb`.

**Coca-Cola:** the requested facts have no observations at all under latest
quarter accession `0001628280-26-050503`. Their latest captured observations
instead come from the preceding April 3 quarter. This is more specific than
a period mismatch: the selected accession is absent from these fact series.
The actual [July 3 quarterly filing](https://www.sec.gov/Archives/edgar/data/21344/000162828026050503/ko-20260703.htm)
does show current assets of $32,876 million and current liabilities of
$25,200 million. Thus the original disclosure contains values omitted from
the captured structured series. This probe does not infer why the feed omits
them or silently merge manually extracted values into that feed. A strict
direct-filing parser would need its own context, units and period validation.
The full SEC filing and its receipt were captured, body SHA-256
`b588e02c3f829c9dd3531240c8bbc9790ca3aaa8f42b4f4d4addcf3ec7e07883`.

## Reproducibility and decision

Separate offline processes produced `coverage-first.json` and
`coverage-replay.json` with identical bytes, SHA-256:
`f034db535874d97ff412b31f904daa447894ff028ed348d248d3a079c0c36aac`.
The report includes every selected filing, accepted timestamp, source receipt
time, exact financial observation and missing status. Its RunRecord verifier
checked all 81 input files plus report and sidecar. The two explanatory HTML
documents were captured afterwards as additional investigation evidence; they
are not misrepresented as inputs to that earlier coverage report.

A complete twenty-company signal combining these four exact-source inputs is
not supported by this route as captured. An explicitly declared eighteen-company
prospective pilot is technically possible after validating denominators and
freezing its selection, weighting, costs, dates and outcome requirements.
That would be a new protocol, not a silent change to this twenty-company panel.

The useful next implementation is an accession- and period-bound feature
boundary plus a frozen prospective protocol. Public receipt timestamps must
precede actual signal computation and the decision; filing acceptance alone
does not prove public availability. Missing future outcomes, delistings and
corporate actions must remain visible. Sector concentration and different
financial period ages limit interpretation. No signal or investment edge was
established merely by obtaining these fields, and no funds are required for
this stage of research.
