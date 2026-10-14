# Public stock-price input: reachable, with measured coverage limits

Measured 2026-10-14. Board task: 32dbc2bc-1f25-4d74-828e-2949ad3e924c.
Evidence: `C:/Users/Test/PROJECTS/files/stock-price-feasibility-20261014`.
No orders, account access, subscription or deposit were required.

The public Yahoo chart endpoint returned actual Apple daily price histories
and corporate-action records. This supplies a bounded data-development route.
It does not establish a complete historical equity universe or profitable
stock strategy. The endpoint is an observed web-data interface; a supported
commercial API contract was not established by this probe.

## Captures and checks

| Captured request | Status | Rows | Source body SHA-256 |
| --- | ---: | ---: | --- |
| AAPL, October 2009 through January 2010 | 200 | 83 | `44a70b7e8a65f824b8f897fa3b95768f89be36145b89e7a9f02e1624d313c851` |
| AAPL, July through September 2020 | 200 | 64 | `9d7515cf0a75ae864966b1569ff61ebe785eda259417a20833beb1cd6c01fe8d` |
| TWTR, September through October 27, 2022 | 404 | unavailable | `f0b5f9feeac9f1fc93c31b1edc03fbc1fda4acb3ec5e233ebb39d6f8b66e1769` |

The successful captures use `aapl-verified.json` and `aapl-2020.json`; their
receipt files preserve complete URLs, request/receipt times, response status
and body hashes. Their intervals contain 83 and 64 unique ordered daily
timestamps. Open, high, low, close, volume and adjusted-close arrays match
those lengths. Every close and adjusted close in the selected windows was
present and positive. Native metadata identifies AAPL, USD and an equity.
`measurements.json` records these checks and selected action-date rows.

An initial curl probe saved `aapl-2009-2010.json`, but its status-receipt
formatter failed before recording a valid receipt. It is not used as the
verified capture. A separate explicit acquisition through the project's real
HTTP client produced `aapl-verified.json` and its complete receipt. This was
an acquisition correction, not a data fallback or substituted missing price.

The TWTR response explicitly reports `Not Found` and says no data was found,
with delisting as a possible reason. Its response and receipt were saved,
then the real HTTP error propagated. No alternate symbol, zero return, last
price or empty successful history was substituted. This one failure proves
the requested symbol/window was unavailable through this route; it does not
prove every delisted security is unavailable or identify its complete mapping.

## Corporate-action semantics matter

The 2020 response contains a 4:1 split dated August 31 and a dividend dated
August 7 of $0.205 per adjusted share. Apple's own
[July 30 announcement](https://www.apple.com/newsroom/2020/07/apple-reports-third-quarter-results/)
declared a $0.82 dividend and a four-for-one split, with split-adjusted trading
beginning August 31. Dividing $0.82 by four yields the response's $0.205.
Apple's [dividend history](https://investor.apple.com/dividend-history/default.aspx)
also identifies the split. These primary-source records support the adjustment
interpretation; this was not inferred from an apparent price jump alone.

The returned closes on August 28 and August 31 are approximately $124.8075
and $129.04. The first is already expressed on the post-split share basis.
Applying the 4:1 action a second time to that series would corrupt returns.
Adjusted close divided by close changes from approximately 0.967394 on
August 6 to 0.969138 on August 7, consistent with a dividend adjustment.

[Yahoo's adjustment explanation](https://help.yahoo.com/kb/SLN28256.html)
describes adjusted close as incorporating splits and distributions. Its
provider-help search result was available; direct help-page reads returned
429 and are not claimed as locally archived full documentation. The Apple
announcement was verified through the primary-source search result after a
direct web read timed out. Raw price and action responses were actually saved.

Neither historical `close` nor `adjusted_close` should silently be labeled the
then-traded cash price. Today's adjusted levels can include actions after the
historical decision. Use an explicitly documented return convention and retain
the original action ledger; do not mix adjusted per-share prices with historical
share counts, then add the same distributions again. The observations here
verify a split/dividend example, not all merger, spin-off or liquidation cases.

## Supported next experiment

The captured Apple responses and accession-bound financial inputs can support
a strict pipeline integration test with a known issuer, declared periods,
explicit action treatment and exact archived replay. That exercise should
verify joins and return arithmetic, not select a profitable-looking date.
Apple was selected for a schema probe, not from an investable historical
universe; these windows are development evidence and cannot become untouched
out-of-sample results after inspecting them.

A stock-edge study still needs a frozen signal and evaluation period,
historically valid security mapping, a complete declared universe and an
explicit policy for delistings and corporate actions. A new prospective cohort
can instead be fixed now and followed forward, retaining every member even if
it delists; it must not silently drop missing future outcomes. No such cohort
or return strategy was registered by this feasibility probe.

For broader coverage, [Nasdaq's corporate-action product description](https://www.nasdaq.com/products/data/equities/corporate-action-solutions)
explicitly covers listings, delistings, symbol changes and dividends. That is
a documented data route, not a claim that this account has access, that it is
free, or that a purchase is necessary before continuing development.

Funding is not the present input bottleneck. We can continue testing acquisition,
identity and accounting with the public evidence. The missing historical
universe and verified return methodology remain explicit rather than being
papered over with convenient surviving tickers.
