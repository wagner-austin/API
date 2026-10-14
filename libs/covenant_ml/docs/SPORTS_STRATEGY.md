# Frozen sports calibration experiment

Board task: `2a002414-5d20-43e7-ad05-682840945e4e`.
Protocol frozen October 13, 2026, before requesting the evaluation-period
market list or prices. A schema probe inspected two June 14 contracts only;
June is outside both experiment partitions. This is a retrospective screening
experiment, not a claim of historically executable orders or future profits.

## Single primary hypothesis

An earlier-game fitted probability temperature improves later-game NHL
winner probability calibration enough to survive a conservative cost screen.
For market midpoint p, the forecast is p^t / (p^t + (1-p)^t).
Choose t from {0.5, 0.75, 1, 1.25, 1.5, 2} using minimum training Brier
score; ties prefer the exponent nearest 1, then the smaller exponent.
No evaluation-period parameter selection, additional strategies or retuning.

Training dates: January 1 through February 28, 2026. Evaluation dates:
March 1 through April 15, 2026. Date comes from the native event ticker,
not eventual close time. Choose one contract per event, the lexicographically
first ticker, before inspecting its price or outcome. The decision cutoff is
12:00 UTC on that date. Require a one-minute bid/ask candle ending exactly
at that cutoff; do not substitute the last trade, a later candle, a daily
candle or a carried-forward price. Require 0 < bid <= ask < 1 and spread
at most $0.05. Use the midpoint as the uncalibrated probability baseline.
Exclude non-binary settlements, absent candles, unavailable quotes and
contracts not open at the cutoff, reporting each exclusion separately.
Training settlements must precede the evaluation start.

At most one hypothetical one-contract position per game. Price a YES at
the ask and a NO at 1 minus the bid. Add $0.01 adverse price movement,
plus ceil-to-cent(0.07 * price * (1-price)) as the declared fee assumption.
Choose the side with the larger forecast minus total cost, only if the
remaining expected margin is at least $0.03; otherwise abstain. Hold through
settlement. Also report results with the fee assumption doubled, without
changing the primary strategy's selections. This does not reconstruct
historical fee overrides or queue position. Candles carry no available
depth, so all profit figures are explicitly hypothetical quote-based scores.

Report training/evaluation game counts, date coverage, selected exponent,
Brier scores against the market midpoint, trades, total hypothetical profit,
and a confidence interval on daily profit totals. Require at least 100
usable games in each partition, at least 30 evaluation trades, improved
evaluation Brier score and a positive lower 95% confidence bound on mean
daily profit (including abstention days with eligible games) before proposing
a subsequent prospective demo study. Passing this screen would authorize
neither real-money execution nor a profitability claim. An inability to obtain
required quotes is a measured data limitation, not a negative strategy result.

## Evidence sources and limits

Use the existing hashed ResponseArchive and explicit capture/offline replay.
Kalshi's [historical data documentation](https://docs.kalshi.com/getting_started/historical_data)
defines the archive cutoff, and the
[candlestick endpoint](https://docs.kalshi.com/api-reference/market/get-market-candlesticks)
documents bid/ask OHLC rather than historical depth. The current settled
market inventory is retrospective: deleted/missing contracts and historical
rule changes are not reconstructed. A positive screen requires further
prospective validation with contemporaneously recorded contract metadata.
