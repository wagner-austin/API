# Official CPI release input for the frozen forecast

Board task: 7a692f5a-0fa5-4c0b-a211-f2d299a854bc.
Observed 2026-10-14 before September CPI publication. No outcome or trade.

## Captured primary evidence

Evidence root: `C:/Users/Test/PROJECTS/files/bls-release-feasibility-20261014`.
All five direct requests returned HTTP 200. Each raw HTML file has an
independent receipt containing URL, request start, response receipt, status,
byte length and SHA256. No fallback source or authenticated access was used.

| File | Exact official source | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `summary.html` | [Current summary](https://www.bls.gov/news.release/cpi.nr0.htm) | 90,388 | `8b4d352b4f1ffc6604d953d304c151993c722f8a096a4030765183105ade9484` |
| `table.html` | [Current Table 1](https://www.bls.gov/news.release/cpi.t01.htm) | 93,040 | `b55519887981966df076de98905a8a66113569cc3534a91bb8efc0d1246564f7` |
| `calendar.html` | [2026 calendar](https://www.bls.gov/schedule/2026/home.htm) | 100,563 | `1bfb9d805fb7b419504962aa22773506b558534daac3b47d72c9de99fd16acac` |
| `archive-index.html` | [Release archive index](https://www.bls.gov/bls/news-release/cpi.htm) | 128,514 | `e522ffbbd037e18febae96b978ac1873617661e9e75a8f5f6ac6b4271ffd6c43` |
| `august-archive.html` | [August dated release](https://www.bls.gov/news.release/archives/cpi_09112026.htm) | 1,372,789 | `aee0cdff604c295d784e6302b2ba74130f10f535675a5a0fb7262c936edaaabd` |

The first four receipts are between `06:43:48.157563Z` and
`06:43:48.295707Z`; the dated release receipt is `06:44:21.064324Z`.
Current summary and dated archive both identify **August 2026**, published
September 11 at 8:30 a.m. Eastern, release identifier `USDL-26-1496`.
Both Table A and the separate Table 1 report **3.4% unadjusted twelve-month
all-items CPI-U change**. This is a parser-development observation, **not**
the September 2026 forecast outcome.

The captured official calendar schedules September CPI for **October 14,
2026 at 8:30 a.m. Eastern**, or **12:30 UTC**. This agrees with the next-release
notice in the current August summary. The frozen Kalshi books close one
minute earlier, at 12:29 UTC. No future result is present in these captures.

## Native contract established by inspection

The summary contains two `PRE` blocks. The first carries the embargo date,
publication identifier and uppercase target-month headline. Requiring exactly
one `PRE` in the entire page would reject this real release. Extract the unique
release header by its content and validate its date, weekday, target month,
time and publication identifier, rather than selecting an arbitrary paragraph.

Table A is explicitly identified by `id="cpi_pressa"`. Its all-items row
header is `id="cpi_pressa.r.1"` with text `All items`. The annual column is
`id="cpi_pressa.h.1.9"`, whose observed normalized text is
`Un- adjusted 12-mos. ended Aug. 2026`. The required data cell's headers are
exactly the row and that annual column. Its `datavalue` is `3.4`.

This is a directly published one-decimal percentage, so there is no reason
to recompute it from rounded index levels, use a seasonally adjusted monthly
change, or substitute the core, CPI-W or chained index. Require the table
caption's CPI-U identity, unique row/header/cell membership, matching target
year/month and a finite signed one-decimal value. Missing, duplicate,
conflicting or changed structure must fail with specific validation errors.

For the observed Table 1, the corresponding cell names row `cpipress1.r.1`,
column family `cpipress1.h.1.6` and annual span `cpipress1.h.2.6`. Its caption
identifies the target. It independently confirms the same published value
within the BLS release; it is not an independent statistical measurement.
The prospective acquisition route can use the summary's header and Table A
together; Table 1 is supporting evidence, not a runtime fallback.

The parser should return an immutable typed release containing target month,
publication timestamp, publication identifier and reported yearly percentage,
with a strict JSON codec. A receipt must follow publication. When scoring,
require the original target and release date, unchanged frozen probabilities,
all native contract identities and rules, and finalized binary settlements
agreeing with the published value at every strict-greater strike. A mismatch
fails the observation; it must not silently exclude a losing contract.

## Timing and first-release limits

Today’s successful requests establish an acquisition route, not a guarantee
that it will answer at release time. The summary URL changes with new monthly
releases, so target validation and immutable raw capture are essential.
An August response after September's scheduled release still fails September
identity; it cannot be called September data or replaced by a derived value.

A local receipt proves when this machine observed the bytes. Today's August
archive fetch does not attest the bytes at their first public appearance.
For September, capture the official release after the scheduled publication,
retain its actual timestamp and identifier, and preserve subsequent corrections
separately. Do not claim a cryptographically attested first publication or
silently merge corrected values into the frozen observation. The native final
settlement agreement is a required additional check, not proof of historical
data availability.

## Reproduction and next implementation

Two separate offline analyses verified all ten raw/receipt files and the
observed title, annual header and three matching 3.4% fields. Their source-bound
`first.json` and `replay.json` reports have identical SHA256:

`5627af0e1df19b7566d4948fd210d32458621ea6535c7e1307c111cd9d461355`

Each report verifies twelve evidence paths including its RunRecord. The
future target is explicitly September, with `future_outcome: null` and zero
orders. This source investigation adds no production parser or scheduler;
the next code step is a strict HTML boundary tested against these actual
current and archived documents, followed by source-bound settlement scoring.
The original inflation snapshot and failed trading-advancement gates remain
unchanged.
