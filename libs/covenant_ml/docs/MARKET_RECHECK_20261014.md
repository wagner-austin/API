# Recheck of market evidence, October 14, 2026

The second pass found one reproducibility defect and repaired it. It did not
change the previous negative trading result or uncover a profitable strategy.

## Independent code verification

Fleet job `8ad3893a-55ab-4c40-bf4f-e30b133a29c4` checked published commit
`21e8d84ba4320a4a447b3f307c8800fdf0701718` on serendipity. It passed 3,847 tests,
100% coverage and the full check in 238/300 seconds, ending at 11:12:44 UTC.
The original audit task subsequently reached verified completion.

This pass reread the grouped splitter, regression tests, forecast feature and
label timing, policy accounting and evidence verifier. The 38 focused split,
feature-causality, market no-lookahead and forecast-model tests passed again.
The feature tests include a past-data positive control; the market tests change
future quotes, trades and settlements while checking earlier placement and
features. These remain behavioral checks, not proof of historical availability.

## Protocol byte drift, reproduced and repaired

`read_verified_market_report` initially refused the 2025 transfer with
`[RECHECK-001] original source evidence changed`. Checking every source found
one mismatch: `docs/MAKER_TEMPORAL_TRANSFER_PROTOCOL.md`. Git's Windows checkout
conversion had changed LF to CRLF during the rebase after the original replay.
The expected digest was
`763fdf28497d32a4917838eb89b3dfec462dd3c4a396629a4114d2ee31f1247e`;
the changed bytes hashed to
`795c14c8e4c863a06229c421a425dcfc9b68fc00bbdf2ac45961f9f79d337db6`.
Removing only CRLF carriage returns recovered the expected digest exactly.

Restored those exact original protocol bytes. No report, plan, market response
or RunRecord was rewritten, and the verifier was not relaxed. The documentation
directory now pins `*_PROTOCOL.md` to LF through `.gitattributes`. A regression
test asserts the actual protocol's recorded SHA-256, so a future incompatible
checkout or edit fails the package suite. This is a narrow correction, not a
claim that arbitrary evidence paths are immutable.

The old transfer report names its protocol by a relative path; verification
must run from `libs/covenant_ml`. Future experiments must record absolute paths
and freeze their protocol in their own evidence directory before acquisition.
The original freeze commit was `f5052401b`; its published rebased counterpart
is `cc243c68e90dec52463e99d233d64d4725e2ef65`. The protocol content is unchanged.

## Result and accounting checks

After restoration both original transfer reports verified all 3,097 source
paths. A new actual CLI replay wrote
`C:/Users/Test/PROJECTS/files/maker-transfer-20261014/recheck-20261014.json`.
Its bytes equal the original report exactly, including SHA-256
`5c9e4bbe911741b24f236ce6c7877804278ce15d189d2ad67a74e7eb104d6ee9`.
For every execution arm, group market counts, fill counts, contract counts,
profit and friction-adjusted profit sum to the pooled values. The declared
two-cent-through high-falling group still has 139 fills, -$5.61 simulated P&L
and -$7.00 with additional friction.

Both 2026 reports verified all 5,910 paths again; their complete `results`
and `pre_entry_regimes` remain equal. The earlier statement about unchanged
market outputs after the two shared-library repairs still holds.

## Next comparison and its limits

The existing noon-cycle forecast report verified 6,081 evidence paths and
SHA-256 `07962971debfbb7e6696cd003aabd62d604c0378bdb8ed193a5c14f9b275ff27`.
Its 643 training observations cover 36 dates; 359 evaluation observations cover
20 dates. Contract count must not be presented as independent sample size.

The [boosting protocol](FORECAST_BOOSTING_PROTOCOL.md) freezes one small
LightGBM challenger against the forecast-augmented logistic baseline, with
two chronological refits, label-availability purging, fixed hyperparameters,
equal-date primary scoring and unchanged fee/friction diagnostics. No boosting
result is claimed by this document. These dates have already been examined;
the comparison is exploratory development and cannot become a fresh holdout
merely because the new model has not previously been fitted.

The operator subsequently requested execution during this same pass. The
[completed comparison](FORECAST_BOOSTING_RESULTS.md) records the actual runs,
independent replay, native model evidence and failed development criteria.
