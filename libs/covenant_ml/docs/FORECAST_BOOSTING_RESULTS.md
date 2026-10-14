# Fixed chronological boosting comparison: challenger does not advance

The [protocol](FORECAST_BOOSTING_PROTOCOL.md) was committed as `7c20abd21`
before the historical challenger was fitted. Both actual offline runs completed
with all 359 declared evaluation contracts across 20 dates. The fixed LightGBM
model was worse than the forecast-augmented logistic baseline in both folds.
All three development criteria failed. No orders were submitted.

## Forecasts and costs

Lower Brier loss is better. The primary loss gives each event date equal weight.
Profits below are hypothetical one-contract quoted taker fills after the
declared fees; the stress column deducts another cent per trade.

| Arm | Date-mean Brier | Trades | P&L after fees | Extra-cent P&L |
|---|---:|---:|---:|---:|
| Raw market midpoint | 0.10679644 | 0 | $0.00 | $0.00 |
| Forecast-augmented logistic | 0.10675462 | 4 | $0.67 | $0.63 |
| Fixed LightGBM | 0.11666810 | 285 | -$6.94 | -$9.79 |

The logistic model's four trades all occurred in the first fold. They are not
enough to establish an edge. Its predictive loss is almost the same as the raw
market reference. A midpoint priced inside its own spread makes no trades under
the fee-plus-hurdle policy, which explains the raw-market zero-trade result.

| Fold | Available training rows / dates | Evaluation rows / dates | Labels purged this fold | Boosting minus logistic Brier |
|---|---:|---:|---:|---:|
| September 21-30 | 643 / 36 | 180 / 10 | 0 | +0.00455163 |
| October 1-10 | 805 / 45 | 179 / 10 | 18 | +0.01527532 |

The source report had already excluded September 20 unavailable labels; zero
new removals in fold 1 does not mean no earlier filtering occurred. Fold 2
excludes 18 September 30 labels unavailable at its first decision. Training
settlement maxima precede evaluation decision minima in both saved folds.

The pooled boosted-minus-logistic difference is **+0.00991348**. Its 95%
date-clustered interval is **[+0.00401533, +0.01581162]**, and its seven-lag
Newey-West interval is **[+0.00258553, +0.01724142]**. Both favor the logistic
baseline within this selected panel. The serial interval uses 20 ordered daily
means; its `n_units=3` is block metadata, not three independent event dates.

Boosting's fold P&L was -$1.25 and -$5.69; the extra-cent amounts were -$2.62
and -$7.17. Its mean loss did not improve in either fold, the pooled intervals
did not favor it, and stress profit was negative. No feature, hyperparameter,
threshold or subgroup was selected after these results.

## What this establishes

This is an executed baseline-versus-boosting comparison with chronological
training and costs, not merely a code test. It rejects this small, fixed
20-tree challenger as a development lead on these inputs. It does not reject
all boosted models, all weather information or all markets.

The dates and noon forecast cycle were selected using previous work. The
comparison therefore remains exploratory development, despite freezing this
particular fit. Twenty dates, initialization-based availability assumptions,
selected contract coverage and hypothetical execution limit generalization.
Neither model defines a $20 portfolio: shared collateral, executable depth,
latency and historical fee changes are not established. The existing prospective
campaign stays frozen, with no challenger substituted into it.

## Implementation and reproducibility

`scripts.forecast_boosting` verifies the source report and original native
market metadata through the existing evidence loader. It calls the existing
logistic and LightGBM factories, uses the same four fixed features, and never
passes evaluation data to fitting or early stopping. Separate modules own
boosted fitting, equal-date scoring and expanding-window evaluation.

Each run records all training identities, evaluation observations, predictions,
losses, fees, actions, logistic coefficients, native booster files, settings and
the actual LightGBM version in the shared RunFingerprint. Protocol copies and
model artifacts have absolute source paths and verified hashes.

Evidence root: `C:/Users/Test/PROJECTS/files/forecast-boosting-20261014`.
The invocation is preserved in `invocation.json`; `first.json` and `replay.json`
each verify **6,086 evidence paths**. All report fields except `sources` match
exactly, and both native model files match byte-for-byte. Whole report hashes
differ because they correctly name different model/protocol-copy paths.

- First report: `53de96b6c3dcdccb1cc22ce4d26cd748be137deb205fbf7a6d1fbb197375fb74`.
- Replay report: `9b15587b8b5a264c3c41f2ae851391014521ce0831842c83dda3a5776e6d887e`.
- Implementation: `684450772fd3a7d10830286895497e2e1e67cd69aacf0370455adee3d58c4f53`.
- Protocol: `9ebb0df5efe65f18b019050a879aa9d969dc9b5f8c46a94b4acab23b90c9a1bf`.

An additional calculation independently recomputed all 359 actions, fees and
cash payoffs for each of the three arms, using Decimal cent rounding rather
than the policy helper. All matched. Direct squared errors aggregated by date
also matched the reported primary loss to within 1e-15.

Full `make check` passed **3,860 tests**, strict mypy across 648 files, all guards,
and **100% of 18,079 statements and 3,684 branches**, in 118/300 seconds. Log:
`C:/Users/Test/PROJECTS/files/market-edge-20261012/boosting-final-make-check-20261014.log`.
The first full check caught date-list JSON typing errors, repaired before this
passing run and before historical fitting. Real-model tests cover future-label
and future-feature mutations, a past-label positive control, native reload,
chronological label purging, equal-date weighting, serial interval disagreement,
fee accounting, offline CLI replay and an actual failing subprocess.
