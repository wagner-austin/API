# September 2026 inflation: frozen pre-release observation

Board task: af257a5f-c24b-44b7-a2b1-88815360e815.
Recorded on 2026-10-14 before the outcome was available.

This is an observational forecast comparison, not a trading advancement.
The historical experiment failed its advancement gates; those gates remain
unchanged. No order was submitted and no profit is claimed.

## Receipt evidence

Evidence directory: `C:/Users/Test/PROJECTS/files/inflation-prospective-20261014`.
The official Cleveland Fed yearly nowcast response was received at
2026-10-14T05:12:19.4914762Z. Its SHA-256 is
`bf0edf1132e22c7ba05729efe6ddcd3701cd1912d50ee36f546e13ce9bcf8c46`.
The Kalshi event inventory was received at 05:12:19.7796387Z, with SHA-256
`07c9dcf13c8f5af1185b71bb2ebca471453e5b1564c8aab670f113d9b1515208`.
Both raw files have receipts recording the request URL, start time, receipt
time, byte length and exact-byte hash.

The 21 native order books were captured through `ResponseArchive.capture`
between 05:12:59.077123Z and 05:13:07.706633Z. Each envelope preserves its URL,
receipt timestamp, full response text and verified body hash. All receipts
precede the native scheduled close of 2026-10-14T12:29:00Z by over seven hours.
These are local clock receipts, not independently timestamped attestations.

All 21 contracts were active and unresolved. The complete inventory had an
empty pagination cursor. Native identities, strictly-greater strikes and
the full primary settlement rule were checked; the official target actual
was absent. Source receipt order, market opening, closing and a fifteen-minute
maximum acquisition span were checked before writing predictions.

## Frozen probabilities

The unchanged `inflation_model` uses the last 60 available earlier monthly
forecast errors, decimal rounding to one reported decimal and half-count
smoothing. `forecast_before` selected the October 13 forecast for September:
**3.60066057218129%**. Its cutoff is midnight October 14, strictly excluding
same-day forecasts. The empirical errors and every prediction are saved in
`inflation_prospective_2026_09.json` beside this note.

Six of 21 order books had both sides. Fifteen had a missing side and receive
no invented midpoint. The six two-sided observations are:

| Strictly above | Model YES | Market midpoint | YES bid / ask |
| --- | ---: | ---: | ---: |
| 3.3% | 0.8770491803 | 0.985 | 0.98 / 0.99 |
| 3.4% | 0.7786885246 | 0.985 | 0.98 / 0.99 |
| 3.5% | 0.6475409836 | 0.895 | 0.89 / 0.90 |
| 3.6% | 0.3032786885 | 0.455 | 0.43 / 0.48 |
| 3.7% | 0.1393442623 | 0.095 | 0.09 / 0.10 |
| 3.8% | 0.0573770492 | 0.015 | 0.01 / 0.02 |

The saved book fields use ten-thousandths of a dollar and hundredths of a
contract. The 3.5% YES bid had only one contract available. Prices, sizes and
receipts are observations; they do not establish an executable edge or a fill.

## Reproduction and later scoring

Two separate offline Python processes applied the existing strict decoder,
`forecast_before`, `earlier_errors`, `inflation_probability`, native book
decoder and shared report writer to exactly the captured inputs. The outputs
`snapshot.json` and `replay.json` are byte-identical, with SHA-256
`cc899b99ebea41def0ea5b531db99fec50c6eb95af7d10b284477b420180d85c`.
The RunRecord and all 26 source hashes passed `read_verified_market_report`.
The checked-in JSON preserves the full prediction payload; original report
bytes and its RunRecord remain in the evidence directory.

The implementation at commit `7e974141c341d5729cd12b4ccd30de447f098610`
passed local `make check` and independent fleet job
`35ab53be-eacd-4e58-a6b6-7e699c512dbe`: 3,608 tests, strict typing, guards,
100% statement and branch coverage. This observation adds evidence only.

After the official first-release September CPI and native final settlement
exist, preserve those raw responses with new receipts. Verify their agreement
at every frozen strike. Compute Brier loss for the unchanged probabilities
and captured midpoint across all six two-sided books, and report all fifteen
missing-side exclusions. Preserve this file and the original predictions;
write the outcome evaluation separately. A mismatch must fail validation.

One release is one dependent event, not 21 independent trials. Quote receipts
are after 05:12Z, not the historical protocol's midnight decision time; do not
pool this observation as another identical historical decision. No simulated
profit, revised threshold, post-outcome contract selection or trading decision
belongs in this forecast-scoring result.

Sources: [official nowcast](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting)
and the native public Kalshi URLs preserved in the receipt files and envelopes.
