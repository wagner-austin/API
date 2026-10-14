# Fixed second-venue sports forecast: no trades after costs

Task `fe565781-a22d-4f62-8e39-cf026b9d1a51`, measured October 14, 2026 UTC.
The [frozen protocol](SPORTS_CROSS_VENUE_PROTOCOL.md) uses a Polymarket US
display midpoint, oriented to the selected Kalshi team, without fitting.

On the 94 eligible March 1–April 15 evaluation games, the second venue's
Brier score was 0.24608643617021278 against 0.24649388297872338 for Kalshi's
midpoint on the **same subset**. Lower is better; the difference is small and
is not a statistical superiority claim. Absolute forecast disagreement averaged
0.005 and never exceeded 0.020. No decision cleared the frozen executable-side
price, adverse-movement allowance, fee and three-cent margin requirement.

**Zero hypothetical trades, zero profit, zero orders. This candidate does not
advance.** This rejects the declared fixed-price-disagreement strategy on this
dataset, not all sports forecasting, maker strategies or other market families.
The zero daily-profit interval describes abstention; it says nothing about
uncertainty in the profitability of strategies that actually trade.

## Complete coverage accounting

| Stage | Count |
|---|---:|
| Original selected Kalshi games | 676 |
| Nonbinary settlements excluded | 1 |
| Training outcomes unavailable before evaluation excluded | 11 |
| Decisions carried into this comparison | 664 |
| US query pages, including terminal empty page | 17 |
| Unique US native market records | 7,795 |
| NHL moneylines in that inventory | 583 |
| Reviewed native team mappings | 32 |
| Selected games with no matching US game | 108 |
| Matched custom history requests | 556 |
| Empty histories | 39 |
| Wider-than-five-cent displays | 7 |
| Eligible US forecasts | 510 |
| Eligible original Kalshi candles | 361 |
| Missing original Kalshi candles | 303 |
| Jointly eligible decisions | 286 |
| January–February context games / dates | 192 / 35 |
| March–April evaluation games / dates | 94 / 27 |

The history captures contain 31,537 stored minute observations. Each joined
game uses both native team IDs and its event date, not a fuzzy name match.
Every one of the 664 decisions has its US exclusion/status, Kalshi candle
status and joint eligibility in the report. The counts on the two venues are
overlapping filters, so they must not be subtracted as independent exclusions.

The original gates require at least 100 context games, 100 evaluation games,
30 hypothetical trades, improved Brier and a positive lower daily-profit bound.
Only the context-count and improved-Brier gates passed. Primary and doubled-fee
profit are both zero.

## Evidence and limits

Evidence root: `C:/Users/Test/PROJECTS/files/sports-cross-venue-20261014`.
`inventory/` holds the 17 queried pages and `history/` the 556 custom windows.
The existing Kalshi inventory and 664 candles remain under
`C:/Users/Test/PROJECTS/files/sports-strategy-20261013`.

`final-result.json` and `final-replay.json` were produced by separate processes
and match byte for byte, SHA-256:
`5aa2d17514014e593d44f712e633cd88425299b1c3c3d7bf6b04ce08c8bbc2ea`.
Each has a RunRecord sidecar. The report binds 1,243 input files, including the
protocol and team map, to hashes. The protocol's frozen hash is
`e6886eeac623583de452ebb18dbd56d3e25b66a1787747064c0a20f9467217c4`;
the mapping's hash is
`e7714797319aaae526ddf8ed8163dc76e96306a8cb1e17adb7e5bddfa42fe785`.

The query is exhausted but is not proof of complete historical venue coverage:
hidden or delisted records may be absent. A 499-row page required continuing by
the requested 500-row offset, not by returned row count. The rejected offset-999
probe is preserved under `probe/`, outside the experiment's inputs. Duplicate
IDs or slugs abort the production inventory reader.

209 of the 286 jointly eligible records have US market creation metadata later
than the decision cutoff. Their retrospective histories therefore do not prove
contemporary US feed availability. This is exploratory evidence, not a repaired
untouched holdout. Display prices lack historical depth and fill evidence;
settlement-rule equivalence across venues is not established. The model submits
no US trades and uses Kalshi settlements only for its hypothetical accounting.

The small forecast differences explain why supplying a second market price did
not produce trades under these costs. Repeating this same test with a lower
margin chosen after seeing these outcomes would not supply independent evidence.
The next investigation should use information beyond the two sports prices;
the economics-release family identified in CROSS_MARKET_INVESTIGATION.md is
the next data-feasibility candidate.

## Validation

The complete `make check` passed in 101 seconds: 3,532 tests passed, strict
typing and guards passed, and coverage reached 100% of 16,799 statements and
3,306 branches. The real HTTP integration test exercises acquisition, offline
replay and corrupt-evidence rejection through the production CLI.

## Reproduce

From `libs/covenant_ml`, choose a new output path:

```powershell
poetry run python -m scripts.sports_cross_venue --kalshi-archive C:/Users/Test/PROJECTS/files/sports-strategy-20261013 --us-archive C:/Users/Test/PROJECTS/files/sports-cross-venue-20261014 --kalshi-url https://api.elections.kalshi.com/trade-api/v2 --us-url https://gateway.polymarket.us --team-map docs/sports_team_links.json --protocol docs/SPORTS_CROSS_VENUE_PROTOCOL.md --inventory-mode replay --history-mode replay --out C:/Users/Test/PROJECTS/files/sports-cross-venue-20261014/reproduced.json
```

Fresh acquisition selects `capture` explicitly for each US stage and uses a new
archive. Kalshi inputs are always the existing frozen archive. Existing requests
and outputs are refused; HTTP errors, identity drift and corrupt evidence
propagate. The CLI, native codecs and matching code reuse the existing archive,
dependency hooks, integer price validation, sports scorer and RunRecord writer.
