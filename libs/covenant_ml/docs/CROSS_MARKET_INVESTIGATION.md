# Cross-market investigation: October 13, 2026

Weather is one hypothesis, not evidence that other markets cannot be beaten.
This investigation compares alternatives and captures public production quotes.
It places no orders and does not change the frozen weather campaign or demo trial.

## What was measured

Evidence directory: `C:/Users/Test/PROJECTS/files/market-edge-20261013-cross-market`.
`capture-manifest.jsonl` records 47 response URLs, request start and completion
times in UTC, and SHA-256 hashes. All 47 hashes were checked against saved bytes.
The separate series inventory contains 15,011 series, including 3,908 Sports,
855 Economics, 276 Crypto, and 424 Climate and Weather series. These are series
counts, not counts of currently tradable contracts. Its SHA-256 is
`fe644f85c060ba54519036d81d1c5cc7f54215f2c4674d9a24ec83d0cae9f712`.

Selection: request all open events in KXNFLGAME, KXNHLGAME, KXCPIYOY,
KXNFLSPREAD and KXNHLSPREAD, with nested markets and limit 200. Each returned an
empty pagination cursor. These families were chosen before inspecting prices.
For cross-venue screening, use the ten October 13 NHL games returned by the
Polymarket US league endpoint, ordered by slug, and match them to Kalshi.
This is a bounded convenience sample, not a census of Polymarket US.

| Screen | Comparisons | Lowest purchase cost | Positive gross bound |
| --- | ---: | ---: | ---: |
| NFL: buy both team winners | 28 | $1.00 | 0 |
| NHL: buy both team winners | 51 | $1.00 | 0 |
| CPI: buy lower-threshold YES and higher-threshold NO | 630 | $1.01 | 0 |
| NFL: buy team winner YES and positive-spread NO | 359 | $1.03 | 0 |
| NFL: buy lower-spread YES and higher-spread NO, same team | 2,192 | $1.01 | 0 |
| NHL: buy team winner YES and positive-spread NO | 126 | $1.18 | 0 |
| NHL: buy lower-spread YES and higher-spread NO, same team | 83 | $1.07 | 0 |
| Kalshi / Polymarket US: opposing winner positions | 20 | $1.00 | 0 |

These are 3,489 overlapping comparisons, not independent trials. The CPI sample
has 63 contracts across three months. The spread samples contain 14 NFL events
and 29 NHL events. The first event snapshots were captured at approximately
22:53-22:54 UTC; the cross-venue books were captured later. Exact receipt times
are in the manifest. No frequency-of-opportunity estimate follows from one pass.

For an ordinary completed game, opposing winners pay $1 collectively. For
nested thresholds L < H, YES(L) + NO(H) pays at least $1, and can pay $2 in the
middle. The screen tests whether purchase cost is below that minimum payout;
it does not test whether a positive-cost middle bet has positive expected value.
NFL ties pay $0.50 per winner contract in the captured rules. Cancellation,
postponement and discretionary fair-price settlement require separate analysis:
the ordinary-completion bound is not asserted for those states, especially
across venues. No positive candidate reached that rule-equivalence gate.

Kalshi event summaries supply asks for the broad initial screen. Four subsequent
depth captures checked two NHL winner books and two near-minimum NFL spread
books. They still showed $1.01 purchase costs. This does not validate depth for
every initial comparison. Cross-venue screening used actual books on both
venues, with nonempty sides; YES asks on Kalshi are one minus the best NO bid.
The [Kalshi book documentation](https://docs.kalshi.com/getting_started/orderbook_responses)
describes that complement relationship.

Crypto: captured Coinbase Exchange level-one BTC-USD, ETH-USD and ETH-BTC books.
USD -> BTC -> ETH -> USD returned -0.142858 basis points before fees;
USD -> ETH -> BTC -> USD returned -0.896011 basis points before fees.
The formulas are `ETHUSD_bid / (BTCUSD_ask * ETHBTC_ask) - 1` and
`ETHBTC_bid * BTCUSD_bid / ETHUSD_ask - 1`, respectively. The three books have
different exchange timestamps spanning about 1.25 seconds; they are not atomic.
This is a negative snapshot screen, not an executable backtest. See the
[Coinbase book specification](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book).

Derived rows are saved in `sports-screen.json`, `cpi-screen.json`,
`sports-related-screen.json`, `cross-venue-screen.json`, and `crypto-screen.json`.
Raw responses remain the authority. Fees were not needed to reject these
gross-negative or gross-zero candidates. This does not assess maker orders,
rebate strategies, expected-value bets, or opportunities at other times.

## Literature and current mechanics

[Cheng, Yang and Zou](https://arxiv.org/html/2605.00864v1) report seven
single-market and 290 combinatorial NBA arbitrage episodes from historical
Polymarket books. Their method assumes zero NBA trading fees. Episode duration
is inferred from polling, and the authors acknowledge that measured duration
can exceed actual duration. Those results motivate measurement, not an assertion
that the same strategy is profitable now or that both legs could have filled.

Current [Polymarket International fees](https://docs.polymarket.com/trading/fees)
include a sports taker coefficient of 0.05 in `C * rate * p * (1-p)`.
At two hypothetical 50-cent fills that is 2.5 cents total per paired share,
before other costs. Use actual market parameters, not the category example,
for any proposed trade. International and US are separate venues; the
[operator's statement](https://polymarket.com/transparency) restricts US users
from trading on the international platform.

[Polymarket US](https://docs.polymarket.us/api-reference/introduction) provides
public REST market data without an API key, which this investigation accessed.
Its [fee schedule](https://docs.polymarket.us/fees), read October 13, uses a
standard taker coefficient of 0.0695, with exceptions. The captured NHL markets
also report 0.0695. The page announces removal of the maker rebate for its
lowest-volume tier on October 19. A small-account experiment must not rely on
today's temporary rebate persisting. No US account eligibility was verified.

[Moshrefi's sports calibration paper](https://arxiv.org/html/2607.14430v1)
examines about 23 million trades and finds time-dependent calibration and
parlay markup. Its reported prices exclude fees. Time-to-expiry is defined
using the market's close time, which the paper associates with game end.
Our implementation would need information available at the decision timestamp:
eventual game end is not a valid live feature. Nor does repeated observation
of one eventual outcome create millions of independent outcome observations.
These are design concerns for a replication, not a claim the descriptive
analysis itself is invalid. Multiplying leg prices is not generally a joint
probability without an adequate dependence model.

The four paper/fee pages above were also captured as HTML with hashes in
`source-manifest.jsonl`. The SEC documentation was read through the web tool;
a separate local HTML download was rejected by SEC automated-access controls
and is not claimed as an archived source.

## Comparison and research priority

| Priority | Hypothesis | Existing advantage and concrete missing evidence |
| --- | --- | --- |
| 1 | Sports: venue disagreement and probability calibration | Public books on two US venues are reachable; repeated scheduled events allow prospective testing. Need synchronized history, rule matching, and a forecast that improves after fees. This pass found no immediate arbitrage. |
| 2 | Economics: predict releases using earlier public information | CPI threshold relationships are measurable. Need vintage inputs and forecasts available before the decision, plus enough independent release dates. A ladder consistency check is not an inflation forecast. |
| 3 | Stocks: company disclosure changes | Covenant-related feature extraction is relevant to company fundamentals. Need tradable identifiers, contemporaneous filings, price history including delistings, and future-return evaluation. Existing bankruptcy/default classification is not evidence of equity alpha. |
| 4 | Crypto: structural conversion or venue differences | Continuous public books make measurement straightforward. The sampled triangular paths were already negative without fees. Cross-venue funding, transfer delay, and inventory costs remain unmeasured. |
| 5 | Options/futures: relative value or hedged event exposure | Requires instrument-specific settlement, multiplier, margin, spread and fee data. No options/futures edge was measured here; a $20 execution budget sharply restricts many strategies. |

This ordering is a judgment about the next experiment's feasibility and reuse,
not a statistical ranking of expected profits. All five remain unproven locally.
Stocks have an actionable data route: the [SEC submission and XBRL APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
expose filings and financial facts. Extract information as published, not the
latest revised financial history; compare a preregistered signal with simple
price/sector controls. A stock strategy also needs an explicit holding period
and risk-adjusted benchmark, rather than treating market appreciation as alpha.

Standard [equity options usually represent 100 shares](https://www.optionseducation.org/optionsoverview/options-basics).
A quoted $0.20 premium therefore typically consumes $20 before fees; that is
not a reason to purchase cheap options. Contract exceptions exist. Broker costs
also vary: [IBKR's US options schedule](https://www.interactivebrokers.com/en/pricing/commissions-options.php)
lists per-contract fees and order minimums. It is not a universal broker tariff.

The reviewed `covenant-radar-api` already registers covenant, weather and esports
domains. Its esports domain derives features from match snapshots and plugs
into the shared streaming worker. That is reusable infrastructure; this review
did not establish a calibrated sports model, a complete live sports feed, or
profitable esports execution. The research archive, strict codecs, chronological
evaluation and accounting work are also reusable. None supplies an edge by itself.

## Next experiment and funding gate

The [repeated sports capture workflow](SPORTS_CAPTURE.md) now implements the
public-data acquisition and exact offline replay stage of this experiment.

Next: a read-only prospective NHL venue-disagreement study. Start with full-game
winner markets, not player props or parlays. Declare the game universe before
quotes, reconcile teams and all settlement cases, archive both book sides and
receipt times, and calculate costs at 1, 5 and 20 paired shares subject to a
$20 total budget. Record absent depth as unavailable, never as a fill.

Separate two questions: (1) do minimum-payout opportunities survive actual fees
and conservative execution delay, and (2) does one venue's contemporaneous
price improve the other's outcome forecast? Use observed game clock/state or
scheduled pregame cutoffs, never eventual closing time. Fit only on earlier
games; group uncertainty by game/date; freeze one primary decision rule before
evaluation. Quote differences alone do not prove which venue is right.

For any candidate, re-observe depth after measured decision/request latency,
account for the possibility that only one leg fills, and verify settlement
equivalence before treating the positions as a hedge. A REST sample surviving
two reads is still not a guaranteed fill. Do not book simulated maker fills
merely because a trade touched the posted price.

Funding conclusion: this investigation needs no real-money deposit. A later
$20 live trial would test execution under a demonstrated candidate rule; it
would not establish profitability through a lucky win. No live trial is
authorized or initiated by this report. The existing demo execution and frozen
weather measurement continue under their existing scope.

This investigation changes documentation only. Separately, the previous weather
calibration fleet run passed its tests and coverage but exceeded the 300-second
runtime budget (345 seconds). That remains an unresolved validation issue;
it does not invalidate these archived market observations.
