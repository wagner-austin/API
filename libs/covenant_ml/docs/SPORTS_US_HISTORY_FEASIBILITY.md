# Historical sports prices from a second venue

Measured 2026-10-14 UTC. Board task
`1924c421-2d75-4a36-8e98-6adbbfbab0e8`.

The public Polymarket US history endpoint returned a complete one-hour sequence
for each of ten sampled NHL games, including eight games in the existing
January–April Kalshi study. Six of those eight also have eligible Kalshi quotes
at the same decision time. This establishes sample data availability for a
comparison; it does not establish a trading edge or population-wide coverage.

## Acquisition and reproducibility

Evidence root:
`C:/Users/Test/PROJECTS/files/sports-us-history-20261014`.

`coverage.json` SHA-256:
`618e5aac6da60c227e3378bb1e0a6b47f22aabf81b7680012b7a3a9c0e1d0657`.

The report binds 25 response envelopes to exact request URLs, acquisition times,
body hashes and file hashes: five inventory pages, ten all-history probes and
ten custom one-hour requests. Existing `ResponseArchive.capture` performed all
requests with its real HTTP client and 0.3-second pacing. Its capture decoder
reverified every response hash when constructing the report. No credentials or
orders were involved. The saved source is `sources/price-history.md`, whose hash
is also in the report.

The deterministic sample was the first three NHL slugs from each first
100-record market-list page with `startDateMin` / `startDateMax` of January 1–3,
March 1–3 and April 1–3, plus the two already declared October 13 sports pairs.
The April page contained only two NHL slugs, giving eight historical games.
This is a feasibility sample, not a representative or exhaustive event sample.
Selection did not depend on prices or outcomes; no profit calculation was run.
The all-history responses include terminal prices, so these samples are already
exposed and must not become a purported untouched test set.

Requests used the public origin `https://gateway.polymarket.us`:

```text
GET /v1/markets?limit=100&offset=0&startDateMin=...&startDateMax=...
GET /v1/price-history?symbol=<native-slug>&fixedInterval=INTERVAL_ALL&fidelity=180
GET /v1/price-history?symbol=<native-slug>&timestamp.startTimestamp=<11:00 UTC game-date>&timestamp.endTimestamp=<12:00 UTC game-date>&fidelity=1
```

The January–April Kalshi counterpart was read entirely offline from
`C:/Users/Test/PROJECTS/files/sports-strategy-20261013` using the existing
`historical_selection` and `historical_quote` functions. The original contract
selection, cutoff and exclusions were preserved. Market tickers in the table
identify the selected team; Polymarket Yes can refer to its opponent.

## Observed coverage

Every custom response contains exactly 61 observations: one per minute from
11:00 through 12:00 UTC inclusive on the slug's calendar date. Validation checked
the entire timestamp sequence, uniqueness, bounds, both prices in [0, 1], and
non-crossed display spreads. There are 610 observations in total. Matching
eligible Kalshi candle timestamps equal the exact 12:00 UTC endpoint.

| Polymarket native slug | Yes / No display at 12:00 | Selected Kalshi ticker | Kalshi bid / ask |
|---|---:|---|---:|
| aec-nhl-chi-was-2026-01-03 | .269 / .746 | KXNHLGAME-26JAN03CHIWSH-CHI | .27 / .28 |
| aec-nhl-wpg-ott-2026-01-03 | .459 / .559 | KXNHLGAME-26JAN03WPGOTT-OTT | .54 / .55 |
| aec-nhl-tor-nyi-2026-01-03 | .507 / .510 | KXNHLGAME-26JAN03TORNYI-NYI | .49 / .51 |
| aec-nhl-veg-buf-2026-03-03 | .430 / .580 | KXNHLGAME-26MAR03VGKBUF-BUF | .58 / .59 |
| aec-nhl-fla-nj-2026-03-03 | .520 / .500 | KXNHLGAME-26MAR03FLANJ-FLA | Missing candle |
| aec-nhl-pit-bos-2026-03-03 | .480 / .530 | KXNHLGAME-26MAR03PITBOS-BOS | .53 / .54 |
| aec-nhl-edm-sj-2026-04-08 | .530 / .480 | KXNHLGAME-26APR08EDMSJ-EDM | Missing candle |
| aec-nhl-tor-nyi-2026-04-09 | .300 / .710 | KXNHLGAME-26APR09TORNYI-NYI | .70 / .71 |
| aec-nhl-pit-chi-2026-10-13 | .570 / .440 | Outside historical study | Not compared |
| aec-nhl-min-wpg-2026-10-13 | .500 / .510 | Outside historical study | Not compared |

The historical Yes team IDs are 1500 (Chicago), 1505 (Winnipeg), 1490 (Toronto),
1513 (Vegas), 1491 (Florida), 1497 (Pittsburgh), 1508 (Edmonton), and 1490
(Toronto), respectively. Each came from the record's `marketSides` long side,
not a guess from name order. Venue abbreviations differ: WAS/WSH and VEG/VGK
already occur in this small sample. A production join needs native IDs and
explicit orientation, not a string-only ticker conversion.

## Boundaries that matter

The [official history documentation](https://docs.polymarket.us/api-reference/price-history/get-price-history)
describes display prices derived from the order book, normally the Yes ask and
one minus the Yes bid. It does not promise historical size, executable fills,
or individual trades. Short custom windows return stored observations; the
all-history profile instead samples older markets daily. Use the custom range
for a decision-time comparison, not a daily point carrying that day's date.

Six sampled January/March records have March 7 `createdAt` values after their
game dates, while their history contains earlier observations. The endpoint
therefore supplies retrospectively available records whose original publication
provenance has not been established here. Their existence is useful for an
exploratory comparison, but is insufficient proof of what the US feed exposed
to a trader on the historical date.

The [market-list documentation](https://docs.polymarket.us/api-reference/markets/get-markets)
exposes start-date filters. Those select the market's start date, not necessarily
the game's date: April 1–3 returned April 8–9 games. An initial October query
also returned crypto records despite a `sportsMarketTypes` query parameter.
The subsequent sample filtered actual returned NHL identities locally. This
probe does not establish whether the discrepancy is parameter serialization or
server filtering; never rely on that filter as validation or claim full coverage
from these first pages.

Historical list records use `sportsMarketType="moneyline"`; the existing live
collector validates a different native value from its own endpoint. Its strict
validation remains unchanged. A historical adapter must validate the historical
contract explicitly, rather than weakening the live collector or introducing a
compatibility fallback. Settlement-rule equivalence between venues has not been
established by the identity matches in this note.

## Concrete next experiment

Proceed with a separate, explicitly exploratory cross-venue forecast comparison.
The first gate is a complete, unique native-ID inventory and an orientation map
for all 664 previously selected Kalshi decisions, with every unmatched event and
missing quote counted. Acquire a short custom range ending at each exact cutoff;
select only observations at or before it with a declared freshness bound.

Before scoring, freeze the join, timing rule and signal: compare the oriented
Polymarket display midpoint against the Kalshi midpoint as probability forecasts,
without fitting a new blend on the already examined March–April outcomes. Report
coverage and paired probability scores first. Any hypothetical Kalshi trade
calculation must preserve actual side orientation, the existing fee/slippage
assumptions and missing-data exclusions. Historical Polymarket execution cannot
be inferred from these responses.

A favorable exploratory result would justify prospective synchronized collection
and a separately frozen evaluation. The unresolved retrospective provenance and
absence of historical depth mean this feasibility result alone is not a reason
to add funds.
