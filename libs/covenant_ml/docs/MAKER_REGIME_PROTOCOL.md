# Descriptive maker regimes, declared October 14, 2026

Task `9879d3bd-872d-4606-aaf7-4ba05db9f1ae`. Definitions written before
computing regime results on the archived panel. Historical outcomes have already
been examined in earlier work; this is an exploratory diagnostic, not a new
holdout or an independently confirmed strategy.

Use only the six hourly candle closes at decision minus five, four, three, two,
one and zero hours. These are within the collector's existing six-hour query.
Each required timestamp must have both bid and ask. Do not carry a quote forward,
interpolate a missing close or substitute a nearby timestamp. A missing slot or
side makes the contract unclassified; report its usable quote count.

At each close take the midpoint. Measure the population standard deviation of
the five successive midpoint changes, without annualization. Split at one cent
(100 internal price units): strictly below is low variation; equality or above
is high variation. Direction is the sign of final minus initial midpoint:
rising, flat or falling. These are absolute changes in binary-contract probability
prices, not stock returns or weather forecast uncertainty. The thresholds are
descriptive choices, not calibrated predictors or economic optimality claims.

Emit all six variation/direction groups plus unclassified. Include all contracts,
exclusions, fills, costs and date-clustered uncertainty using the existing summary
implementation. A contract belongs to exactly one group. Multiple contracts on
one event date are not independent observations. Group intervals are exploratory
and have no multiple-comparison correction. Missing coverage can be selective.

Preserve every existing order, fill-proxy assumption, pooled statistic and
advancement gate. Never use these labels to select trades or tune a profitable
subset. Preserve the original frozen reports and write new diagnostic reports.
Native candle closure does not prove the last underlying quote update was fresh.

Test causal inputs with future price/quote, trade and settlement edits and with
future data appended, removed and reordered. Past features and order placement
must stay identical; future fills and profit may legitimately change. Positive
controls must change relevant past quotes and move the computed feature/decision.
Exercise the actual collector/parser, report writer and offline replay as well
as the pure functions. Keep the prospective campaign and scheduled runtimes frozen.
