# 2026-08-22 — the last two decorative knobs closed: max_features implemented, track_contributions removed

Agent-board task `d127aa61`, completing the config-not-reaching-training
class audit (8f03f32a). After this change, every field a ClearGBM config
can state is either honored by training or does not exist.

## max_features: implemented as a per-split feature budget

The documented semantic ("max features per split"), built in cleargbm_rs:
each node's split search considers a random subset of features, derived as
a pure function of (random_state, boosting round, node_id) via a per-node
RNG — stream-free, so the row-subsampling RNG never advances and
unsubsampled training is bit-identical to history. Histograms are still
built for all features (sibling subtraction needs complete parents); only
the search is restricted. Required-with-null config field (the num_leaves
contract), validated at construction and against n_features at train time,
serialized into the model JSON, threaded through cleargbm's Python layer
(which had dropped it at the Rust boundary since the field existed) and
the covenant_ml backend's fraction-to-count resolution.

## track_contributions: removed from every config surface

Contribution extraction is the covenant_ml explainer's post-hoc capability
over saved model JSON — it was never a training knob, and the Rust core
never saw the flag. A config field the trainer cannot honor has exactly
one honest state: absent. Removed from cleargbm's GradientBoostingConfig,
covenant_ml's ClearGBMConfig, the train-external wire parser (which now
tolerates and drops the stray key from older clients, with a test pinning
that behavior), the optimize builder, and every construction site.

## Equivalence gate

The four-arm benchmark under the new crate reproduces the 2026-08-21
single-pass manifest's quality metrics and leaf counts byte-for-byte on
every cleargbm arm and seed (manifest:
`BENCHMARK_MANIFEST_2026-08-22_knob_identity.json`). Both changes are
provably inert at defaults; every recorded manifest remains valid.

## The measurement: max_features on rw_matches grouped 5-fold CV

Same protocol as the weighted baseline (569,561 rows, 99 match groups,
16 features, production config), one delta: max_features=0.8 (12 of 16
features per split).

| arm | mean held-out AUC |
|---|---|
| all features (weighted baseline) | 0.7492 ± 0.0754 |
| max_features=0.8 | 0.7493 ± 0.0785 |

A statistical wash on the mean with real fold-level movement (fold 2
+0.012, fold 3 −0.012) — the knob demonstrably changes which splits win
without helping or hurting at this setting on this corpus. That is the
expected price signal for a regularizer on a corpus this large: its value
is as a tunable dial, not as a default. (Correction to an earlier claim:
the ClearGBM optimizer search space never sampled max_features — the
sampled max_features in the optimize builders belongs to RandomForest —
so no trial was mispriced by the old drop; adding it to the ClearGBM
search space is now possible and previously would have been meaningless.)

### Power: the wash is not a measurement, and the page's own prose proves it

Added by the power audit, board `1e4ab572`, 2026-09-09. "A statistical wash on
the mean" is a null, and a null needs the effect its instrument could have
resolved. The per-fold values are not published here, so the paired sd cannot
be computed exactly — **but this page states two of the five fold movements,
and those alone settle it.**

Fold 2 moved +0.012 and fold 3 moved −0.012 against a mean difference of
+0.0001. The three unpublished folds can only ADD to the sum of squared
deviations, so dropping them gives a strict lower bound:

| quantity | bound |
|---|---|
| mean paired difference | +0.0001 (0.7493 − 0.7492) |
| paired sd | **≥ 0.008486** |
| MDE at n = 5, `t_crit(4)·sd/√n` | **≥ 0.010536** |
| stated threshold of practical interest | 0.01 |
| replicates required | **≥ 6**, and only more as the true sd rises |

**The floor already exceeds the threshold, so `max_features=0.8` on
rw_matches is NOT TESTED — provably, in the best case the withheld data could
possibly describe.** The observed +0.0001 is at most 0.0095× its own detection
floor; the instrument is roughly 105 times too coarse to see it. No arrangement
of the three unpublished folds can rescue the verdict, because every
arrangement makes the sd larger.

**What this does NOT undermine.** Implementing the knob was correct regardless:
it was crossing the Rust boundary and being dropped, and a config value that
does not reach training is a defect at any effect size. The knob-sensitivity
test — train twice, assert the model changed — is also unaffected, because it
asserts a DIFFERENCE rather than a magnitude, which is exactly why it catches
this bug class where coverage and type checks do not. What is unsupported is
the quality verdict layered on top: "neutral at 0.8" describes a five-fold
design that could not have detected one AUC point, not the knob.

**The honest reading is the one the page already reaches by another route.**
It says the value of `max_features` "is as a tunable dial, not as a default",
and that the optimizer never sampled it. Both remain true and neither depends
on the wash. The claim to drop is that 0.8 was measured to be neutral; the
claim to keep is that nothing has yet measured it either way.

## The bug class, closed out

Four members found on 2026-08-21/22, all now dispositioned:
reg_lambda-et-al hardcoded (996bf364, +0.2 AUC pts on rw_matches),
scale_pos_weight decorative (c14dc11f, +1.3 pts), max_features dropped
(this change, implemented, neutral-at-0.8), track_contributions
unimplementable (this change, removed). The detector that catches the
class is knob-sensitivity testing — train twice with different values and
assert the model changed — now present for growth_strategy, num_leaves,
scale_pos_weight and max_features. Types, coverage and
completion-asserting tests all passed through every one of these defects.
