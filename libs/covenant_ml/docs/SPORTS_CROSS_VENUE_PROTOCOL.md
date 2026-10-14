# Frozen cross-venue sports forecast screen

Declared 2026-10-14 before scoring this candidate. Task
`fe565781-a22d-4f62-8e39-cf026b9d1a51`. This is exploratory: the same Kalshi
outcomes and ten Polymarket sample histories have already been examined.

Use the original 664 eligible January 1–April 15, 2026 Kalshi specifications,
one lexicographically selected contract per game, with the existing 12:00 UTC
decision, binary settlement and training-settlement exclusions unchanged.
Inventory Polymarket `/v1/markets` with limit 500, offsets in multiples of 500,
endDateMin January 1 and endDateMax April 17 UTC, categories=sports. Continue
through the first empty page, including after short pages; reject repeated
native IDs or slugs. The observed 499-row page is not an end marker. This is
coverage of this declared API query, not proof that hidden/delisted markets
never existed. Preserve the failed offset-999 probe separately from run inputs.

Freeze a bijective 32-team native-ID map from the independently inspected team
names. Join by event calendar date and the unordered pair of native team IDs.
Reject ambiguous matches. Check both opposing Kalshi contracts; one selected
team alone is insufficient. Validate Polymarket's two sides and exactly one
long side, native IDs and NHL league, moneyline type, and slug identity.
No substring or fuzzy-name joins; no back-compatibility decoding.

Query a one-hour custom price-history range ending at the exact decision cutoff,
fidelity=1. Reject duplicate, reversed, out-of-window or malformed observations
and crossed prices. Use the last observation at or before the cutoff, at most
60 seconds old; no extrapolation or replacement from another timestamp. Exclude
empty, stale, boundary or wider-than-five-cent displays. Require market start
at or before the cutoff and game start strictly later. Report late-created
metadata explicitly: it does not establish contemporary feed availability.

The fixed forecast is (longPrice + 1 - shortPrice) / 2, complemented when the
selected Kalshi team is the Polymarket short side. No fit, blend search, model
selection or adjustment based on outcomes. January–February provides coverage
context; March 1–April 15 is the scored exploratory partition. Every decision
receives an exclusion or eligibility record; preserve missing Kalshi candles.

Reuse `score_sports_predictions`: one hypothetical contract per game only when
the forecast clears the executable-side Kalshi price plus one cent adverse
movement, rounded quadratic fees (coefficient .07), and a three-cent margin.
Report Brier scores against the same-subset Kalshi midpoint, hypothetical profit,
doubled-fee profit, trade count, daily interval and existing sample-size gates.
Historical display prices contain no depth or fill proof. Even passing gates
would license a fresh prospective demo experiment, never real-money funding.

Capture every input with URL, acquisition timestamp and SHA-256. Reproduce the
report offline with identical bytes and bind it to the frozen protocol, team map,
implementation and existing RunRecord provenance. Do not modify this protocol
after the candidate's results are scored.
