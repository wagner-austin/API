# Prospective financial-strength pilot: frozen protocol

Board task: 077c3e57-8770-4028-9320-d7e32e49cb58.
Declared 2026-10-14 before calculating signal rankings or observing outcomes.
This is a paper research experiment; it authorizes no orders or funded promotion.

## Scope and hypothesis

The hypothesis is that a simple financial-strength ranking outperforms the
equal-weight cohort over the declared future window after a stated cost
allowance. This is an exploratory hypothesis, not a replication of a published
quality factor, calibrated return forecast or demonstrated trading edge.

The fixed eighteen-company cohort is: AAPL, MSFT, GOOGL, AMZN, META, NVDA,
ADBE, CRM, CSCO, ORCL, INTC, IBM, WMT, COST, TGT, PG, PEP and JNJ.
These are precisely the complete-input companies from the previously declared
twenty-company coverage probe. AMD and KO remain listed as excluded before
ranking: the selected AMD annual amendment lacks the required facts, and the
captured KO structured series lacks the selected quarterly accession. No
replacement securities are introduced. This purposive, concentrated cohort
cannot support claims about all stocks or a historical investable universe.

Use only the forty native responses captured in `stock-panel-20261014`, as
identified by coverage report SHA-256
`f034db535874d97ff412b31f904daa447894ff028ed348d248d3a079c0c36aac`.
No fresh financial response, alternative tag, older period or amended rule
may silently replace this snapshot. Invalid input fails the pilot explicitly.

## Feature and selection rule

For each member, use the exact matched facts and filings from the coverage
report: current assets and liabilities from its selected current filing;
annual net income and end-of-year assets from its selected annual filing.
Require matched issuer, USD unit, accession, period and filing metadata within
each pair, positive denominators, nonnegative assets and finite arithmetic.
Annual income may be negative. Annual duration remains 300 through 400 days,
as declared by the coverage probe. Retain actual dates; do not pretend fiscal
periods align or that annual observations all have the same age.

Compute current ratio = current assets / current liabilities, and annual
return on assets = annual net income / end-of-year assets. The latter is
explicitly an ending-assets convention, not average-assets profitability.

For each feature, count cohort members with a strictly smaller value. Add
the two counts, giving equal weight to the two ordinal measures. Sort by
descending summed score, then ascending ticker for deterministic ties. Select
the first six. Each selected name receives one-sixth of the paper portfolio;
the benchmark holds all eighteen at one-eighteenth each. No fitted coefficient,
threshold search, sector adjustment, short position, leverage or rebalancing
is part of this pilot. Persist all eighteen features, component ranks and
selection indicators, including the twelve unselected members.

## Information timing and future outcomes

The signal computation and all source receipts must precede the decision at
2026-10-14T13:00:00Z. Record the actual computation completion timestamp and
refuse the experiment if this deadline is missed. SEC acceptance is retained
as metadata, not substituted for receipt or public availability.

Paper entry is the October 14, 2026 regular-session close. Paper exit is the
November 11, 2026 regular-session close, twenty ordinary weekday sessions
later. Require both exact dates and the full expected sequence of 21 bars for
every cohort member. Do not substitute a later first bar, shorten the horizon,
forward-fill a suspension, drop a delisting or turn a missing outcome into zero.
An unexpected closure or unavailable outcome makes this window incomplete
under the frozen rule and requires an explicit evidence report.

Use the archived provider distribution-adjusted close ratio minus one. Never
apply splits or dividends again. Keep raw action evidence and validate the
source's issuer, currency, daily interval and completed-bar receipt time.
Both endpoints must come from the same archived response for each name.
These are adjusted research returns, not broker fills or a cash-share ledger.

The primary statistic is the selected portfolio's equal-weight adjusted
return minus 0.002, then minus the gross equal-weight cohort return. Report
the gross selected return and its net absolute return separately. Repeat only
the declared cost sensitivity at 0.005 round trip; these are assumptions, not
measured spreads or a claim that a $20 account can execute this portfolio.
Do not change the primary statistic after seeing results.

## Interpretation and stopping rules

There is one future portfolio window, not eighteen independent experiments.
Report all constituent outcomes, baseline comparisons, missingness and sector
concentration without manufacturing significance from correlated stocks.
A positive result would justify further independent testing, not establish
repeatable alpha. A negative result stays negative; no retrospective changes
to dates, names, features, costs or weights are permitted for this pilot.

The frozen snapshot must state `advance_to_funded_trading: false`, retain
unobserved outcomes as null and record zero orders. No money is required.
Completing an outcome report requires actual future closes; a successful
code check cannot substitute for those observations. This protocol alone
does not install a scheduled collector or claim that future collection ran.
