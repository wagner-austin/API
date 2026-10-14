# Earlier-game team ratings: reject advancement

The [declared rating strategy](SPORTS_RATINGS_PROTOCOL.md) selected a 50/50
blend of team-strength and market probabilities on the earlier games. On the
later games its forecast score worsened and its nine hypothetical trades lost
$3.16 after the declared costs. It fails advancement to a prospective demo
study. No live or demo orders were submitted by this experiment.

This is exploratory because the earlier temperature experiment already examined
the evaluation dates. Parameters were declared before computing ratings; they
were not changed after seeing this loss. Reject this specific implementation,
not team forecasting in general. It uses no lineup, goalie, injury, rest,
travel, goal-margin or independent sportsbook information.

## Results

| Measurement | Primary fitted blend | Predeclared ratings-only diagnostic |
| --- | ---: | ---: |
| Training games | 233 | Same observations |
| Evaluation games | 128 | Same observations |
| Evaluation dates | 39 | 39 |
| Rating weight | 0.5 | 1.0 |
| Market baseline Brier | 0.242362890625 | 0.242362890625 |
| Forecast Brier | 0.24540650278586457 | 0.2515626698810144 |
| Hypothetical one-contract trades | 9 | 55 |
| Hypothetical profit after costs | -$3.16 | -$2.54 |
| Same decisions with doubled fees | -$3.34 | -$3.64 |
| Mean daily profit 95% interval | [-$0.156665, -$0.005386] | [-$0.224940, $0.094683] |

The primary rule fails the trade-count, improved-Brier and positive-bound gates.
The diagnostic also fails the latter two gates and is not an alternative chosen
after the result. Both use the same quotes, adverse-price allowance and fee
assumptions. These are hypothetical quote-based scores, not executed returns.
The interval treats dates as independent; serial dependence may understate
uncertainty. The missing-quote and retrospective-metadata limitations from the
[temperature experiment](SPORTS_STRATEGY_RESULT.md) also apply.

## Information timing and identity checks

The existing inventory contains 35 team ticker abbreviations but 32 native team
IDs. Every abbreviation maps to exactly one ID; TB/TBL, NJ/NJD and LA/LAK are
the three alias pairs. Ratings use the native IDs without guessing from names.
There are 1,335 October 1-April 15 games, no incomplete contract pairs and one
non-binary game excluded. The 1,334 ordinary binary games are available for
updates; each is processed at most once, after both team contracts have settled
and before a later decision. All 361 quoted observations had
at least ten completed games for each team at their decision cutoff.

The evaluator was lifted from the temperature experiment so both candidates
share the same quote policy, costs, daily totals, intervals and gates. Replaying
the original temperature result after this change preserved every report field
except the implementation fingerprint. No frozen temperature choice changed.

## Evidence and reproduction

Board task: `43d59659-3a95-4d6a-8308-b890d883e921`.
Protocol SHA-256:
`6aabdcf13f7e0352a3d4c79f9f12658db7783b6a4f746d4125880ed82102cf1d`.

Evidence root: `C:/Users/Test/PROJECTS/files/sports-strategy-20261013`.
`ratings-initial.json` and `ratings-verified.json` match byte for byte across
separate processes, SHA-256
`591a91bc11f60480e67e5b498997418e9b2e3bf9074431a9de815b5936b8c505`.
Both bind the existing 668 request envelopes and have RunRecord sidecars.

From `libs/covenant_ml`:

```powershell
poetry run python -m scripts.sports_strategy --model ratings --archive C:/Users/Test/PROJECTS/files/sports-strategy-20261013 --market-url https://api.elections.kalshi.com/trade-api/v2 --mode replay --out C:/Users/Test/PROJECTS/files/sports-strategy-20261013/ratings-reproduced.json
```

The model argument is required: `temperature` and `ratings` are explicit
experiment choices. Capture and offline replay remain separate modes.

## Validation

The full local `make check` passed: 3,476 tests, strict typing and guards,
with 100% coverage of 16,573 statements and 3,228 branches. The complete
check took 206 seconds against its 300-second budget. Its log is
`ratings-make-check.log` in the evidence root. Remote audit is separate;
a local pass does not resolve the earlier remote runtime failures.
