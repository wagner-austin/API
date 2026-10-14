# Earlier-only boosted selection: failure to improve later predictions

Measured October 14, 2026 for board task 46fc5d55-c6f6-41db-b8ab-0474f4ca009c.
The [protocol](FORECAST_SELECTION_PROTOCOL.md) was committed as `7f13ab05c`
before fitting historical observations. All outer dates had already been
examined; this is exploratory development, not fresh holdout confirmation.

## What the additional runs established

Twenty-four boosted candidates, two selected refits and six logistic fits ran
per execution. Selection used only the last ten dates inside each earlier
training partition. Later features and labels were excluded from selection.

| Outer period | Inner train / validation rows | Purged inner labels | Selected booster |
|---|---:|---:|---|
| September 21-30 | 446 / 180 | 17 | 400 rounds, 8 leaves, no weather |
| October 1-10 | 607 / 180 | 18 | 20 rounds, 4 leaves, no weather |

The first validation period favored longer training: market-only 4-leaf
validation Brier improved from 0.120232 at 20 rounds to 0.112358 at 400.
The second favored shorter training: the same comparison worsened from
0.126872 to 0.142094, while training loss decreased. Thus underfitting is a
plausible part of the first fixed model's weakness, but increasing training
does not reliably solve it. The second window shows worsening generalization
with additional fitting. Both windows' logistic validation scores beat every
boosted candidate; weather improved those logistic scores only slightly.

## Later-period results

Same 359 contracts on 20 represented dates as the fixed comparison. These are
correlated contracts, not 359 independent events. Costs and entry policy are
unchanged: one contract per opportunity, quoted taker side, assumed quadratic
fee coefficient 700 with cent ceiling, two-cent hurdle, extra-cent stress.

| Arm | Equal-date Brier | Trades | Net simulation | Extra-cent stress |
|---|---:|---:|---:|---:|
| Raw market | 0.106796442 | 0 | $0.00 | $0.00 |
| Logistic | 0.106754620 | 4 | +$0.67 | +$0.63 |
| Selected boosting | 0.122687721 | 208 | -$13.23 | -$15.31 |

Selected boosting lost $7.54 on 60 first-fold trades and $5.69 on 148
second-fold trades. Its pooled Brier contrast against logistic was +0.0159331
(worse), date-clustered 95% interval [0.0099793, 0.0218869] and seven-lag
Newey-West interval [0.0124717, 0.0193945]. All three advancement criteria fail.
There is no family-wise correction for the historical research program and
no prospective minimum-detectable-effect guarantee. The displayed serial
interval's n_units=3 is block metadata, not three independent event dates.

This is not a $20 portfolio: no shared collateral constraint, depth guarantee,
historical fee reconstruction or actual fill was measured. No orders were sent.

## Reproduction and checks

Evidence: `C:/Users/Test/PROJECTS/files/forecast-selection-20261014`.
Both `first.json` and `replay.json` verify 6,110 input/artifact paths. All report
fields except output-specific source paths match exactly; all 26 native model
files match byte for byte. Report hashes differ because their source paths do.

- First: `81b50ce3d15058c7adb25447ede5163084111d1bb8f5e27bb2e7bb9119dfa297`.
- Replay: `a29ce7c64593e6f194d06bd58e174a52fbac548e250ccab39d4a87298e349559`.
- Protocol: `6dd0cdb024d177cb4228bd26348914bea613175ecbfb2f625fe855c8cdcfa44b`.
- Implementation: `8b658d23bec2bc5cd7315de9f5d72e9a7e0941df4a4a6aeeb9eae60bfceacb95`.

The shared CLI now requires `--study fixed` or `--study selection`; all other
arguments retain their explicit source/protocol digests and date boundary.
Fixed study retains its original learner and scoring, with additional study
and selection metadata. Existing archived reports remain untouched.

Full `make check` passed: 3,868 tests, strict typing, guards, 100% coverage of
18,135 statements and 3,700 branches, 119/300 seconds. Tests include actual
native fits, real socket capture, subprocess replay/failure, future-feature
and future-label mutations preserving earlier selection, exact candidate
argmin accounting, label-availability purging and short-history refusal.
Log: `files/forecast-boosting-20261014/selection-second-check.log`.
