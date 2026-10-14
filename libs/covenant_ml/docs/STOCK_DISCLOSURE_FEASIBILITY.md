# Stock-disclosure experiment: measured input feasibility

Measured 2026-10-14. Board task: e4cb5e1c-ae64-4829-89cb-797f86bba05b.
Evidence: `C:/Users/Test/PROJECTS/files/stock-disclosure-feasibility-20261014`.
This report assesses an input route; it contains no stock-return backtest,
trade, fitted return model or claim of profitable stock selection.

## Existing data cannot directly supply tradable stock labels

The actual local files, rather than their README's advertised counts, give:

| Input | Measured rows | Columns | Identity and availability limitation |
| --- | ---: | ---: | --- |
| US `us_data/american_bankruptcy.csv` | 78,682 | 21 | 8,971 company keys, every one anonymized as `C_*`; year spans 1999-2018; no security identifier or disclosure timestamp |
| Taiwan `taiwan_data/data.csv` | 6,819 | 96 | Bankruptcy label and 95 features; no company key, security identifier or disclosure timestamp |
| Polish `1year.arff` | 7,027 | 65 | 64 attributes and bankruptcy label; no identified security or publication timestamp |
| Polish `2year.arff` | 10,173 | 65 | Same schema limitation |
| Polish `3year.arff` | 10,503 | 65 | Same schema limitation |
| Polish `4year.arff` | 9,792 | 65 | Same schema limitation |
| Polish `5year.arff` | 5,910 | 65 | Same schema limitation |

`local-datasets.json` preserves hashes, full headers and these measurements.
The US file contains 73,462 alive and 5,220 failed rows; Taiwan contains 220
bankrupt and 6,599 non-bankrupt rows. US SHA-256:
`cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`.
These are row counts, not independent companies or trading opportunities.

The [original US dataset repository](https://github.com/sowide/bankruptcy_dataset)
explicitly says it cannot release the non-anonymized version under its source
license. Its README reports 8,262 companies, which differs from the 8,971
distinct keys measured in our file. Do not silently treat these versions as
identical. The upstream README and its receipt are archived locally.

## Reuse the machinery, not the bankruptcy label as a stock forecast

Code references are relative to the repository root:

- `libs/covenant_ml/src/covenant_ml/datasets/registry_configs.py` registers
  the three bankruptcy inputs. The US ordinary dataset excludes company name
  and year from its predictors. None of these target labels is a forward return.
- `libs/covenant_ml/src/covenant_ml/benchmarking/dataset.py` preserves company
  group codes and source hashes. Company-disjoint evaluation is useful, but
  its retained data do not establish filing-time availability.
- `libs/covenant_ml/src/covenant_ml/trainer_fit.py` calls `stratified_split`
  with a random state and optional groups. That is not a chronological stock
  return evaluation; using it unchanged would not prove a trading edge.
- `libs/covenant_ml/src/covenant_ml/markets/archive.py`, `reporting.py` and
  `provenance.py` already supply exact-response capture, replay, source hashes
  and environment-bound reports. Reuse these, not another ad hoc HTTP framework.
- `services/covenant-radar-api/src/covenant_radar_api/seeding/real_data.py`
  manufactures Taiwan identifiers and maps US numeric columns into proxy
  ratios; `real_data_raw.py` converts missing/invalid values to defaults.
  These seed paths do not qualify as strict point-in-time equity inputs.
- `libs/covenant_domain/src/covenant_domain/features.py` computes debt/EBITDA,
  interest coverage, current ratio and leverage changes. It assumes supplied
  financial metrics and covenant results, and has default-on-zero/history
  behavior. It is a domain reference, not an approved equity feature adapter.

No seed behavior was changed here, and no existing bankruptcy classifier was
represented as a stock-return model. A new input adapter must explicitly
reject missing or ambiguous evidence and match the project's strict types.

## Primary-source route verified with actual responses

The [SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
provide submissions and standard entity-wide financial facts without API keys.
We successfully captured Apple submissions and company facts from `data.sec.gov`
at 05:17:08Z and 05:17:40Z. Receipts bind URLs, request/receipt times and hashes:

- Submissions: `2821703cf04e1138923243a87cc5148673feb9d3792a5b6a7747d1e1a093bf7f`.
- Company facts: `73a86c6aedc31f77cac2ea4df5f80f0b3bd7e6eb58bb4e01444fbedf3afb9c43`.

`sec-feasibility.json` records the measured structure. The current submissions
response names Apple and AAPL and contains 1,001 recent filing rows with aligned
accession, form, period, filing-date, acceptance-time and primary-document
columns. It also identifies additional historical submission files.
The latest 10-Q in this response is accession `0000320193-26-000020`, period
2026-06-27, accepted 2026-07-31T14:01:02Z. The period end is not its publication
time. Current ticker metadata is not a historical security-identity table.

The `Assets` fact contains 146 observations across 70 period ends. The
2009-09-26 period appears seven times: the original 2009-10-27 filing reports
$53.851 billion, while a January 2010 filing reports $47.501 billion for that
same period. This directly demonstrates why grouping by period and taking a
latest value can leak later information. Join facts to accession-specific
acceptance times and choose only information available at the decision.
Older accessions require the additional submissions files; do not guess their
acceptance times from a filing date.

Current assets, current liabilities, current/noncurrent long-term debt,
operating income, interest expense, net income, operating cash flow and
depreciation/amortization tags are present somewhere in this company's history.
Presence is not proof of complete coverage at every date. Custom covenant
definitions and borrower-specific EBITDA adjustments require the filing text;
the aggregate standard-tag endpoint does not solve those semantics.

## Concrete next implementation and evaluation

First implement and test accession-bound financial observations: strict
submissions decoding; equal native array lengths; unique accession identities;
aware acceptance times; company-fact unit and period validation; and as-of
selection that rejects unavailable or conflicting facts. Use the captured
original/revised pair as a real regression fixture. This can be executed now
without a deposit or broker account and produces an auditable input, not alpha.

Before scoring returns, obtain a verified historical security mapping and
prices with splits, cash distributions, delisted securities, venue coverage
and acquisition provenance. No such matched return panel was established in
this pass. A present-day survivor list cannot define a historical universe.
[Alpaca's own documentation](https://docs.alpaca.markets/us/docs/historical-stock-data-1)
distinguishes its single-exchange IEX feed from consolidated SIP data; merely
obtaining a bar does not establish full-market execution or delisting coverage.
No account access, subscription or purchase was attempted.

Freeze a single candidate before seeing returns: nonfinancial common stocks,
one declared financial-disclosure change, entry at the next full regular
session after acceptance plus processing delay, and a 20-session holding
period. Declare the universe, tag definitions, missing-data exclusions,
corporate-action accounting, costs and benchmark before acquisition/scoring.
Train only on earlier disclosures and completed earlier holding periods;
purge overlapping return labels at split boundaries. Compare the candidate
with simple price/sector controls and an appropriate market benchmark, group
uncertainty by issuer and overlapping calendar exposure, and retain every
failed candidate. The hypothesis and exact thresholds still need freezing;
this paragraph is a design requirement, not a preregistered trading strategy.

The funding gate remains unchanged: identifiable financial observations and
a working model are prerequisites for testing an edge, not evidence of one.
This feasibility pass changes documentation only. Existing market code passed
3,608 tests and 100% statement/branch coverage locally and on the independent
fleet check at `7e974141c341d5729cd12b4ccd30de447f098610`.
