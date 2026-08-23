# 2026-08-23 — P3 Landing B: native categorical splits (many-vs-many)

Agent-board task `6e71afae` (ClearGBM program charter P3, final landing).
The third data-realism item: features can be declared categorical and
split by set membership over category codes, LightGBM's many-vs-many
mechanism, end to end from config to prediction.

## The mechanism

- **Config**: `categorical_features` is required-with-null on every
  surface. In cleargbm_rs and cleargbm-python it is a strictly ascending
  index list (the one canonical spelling of a set); in covenant_ml and on
  the Covenant-Radar wire it is a list of column NAMES resolved against
  the dataset's features, matching the monotonic-constraints precedent —
  an unknown name is an error, never a silent drop.
- **Binning**: a categorical column must hold non-negative integer codes
  (NaN = missing). Each distinct code gets its own bin in ascending code
  order; more distinct codes than `max_bins` is an error naming the
  feature and both counts — no silent rare-category grouping exists.
  `-0.0` normalizes to `0.0`. `FeatureBinning` is an enum (Numeric edges
  XOR Categorical map), so no feature can carry both.
- **Split search**: per node, the feature's non-empty categories sort by
  gradient-to-hessian ratio — the ordering Fisher (1958) proves
  sufficient for an optimal binary partition under a convex loss — and a
  prefix scan over the sorted order finds the best subset, with the NaN
  partition tried on both sides exactly as in the threshold scan. The
  sort key floors near-zero hessians at EPSILON rather than applying
  LightGBM's `cat_smooth` prior: no smoothing constant exists in the
  config, so none is silently applied (a future knob, stated-or-nothing).
- **Representation**: `SplitDecision` is an enum — `Threshold{split_bin}`
  XOR `CategorySubset{left_bins}` (a 256-bit bin mask) — so a threshold
  split can never carry a meaningless bin set and a categorical split can
  never carry a fabricated threshold. Node finalization translates the
  winning bins into raw category CODES via the per-run layout; the codes
  live on the node (`categories_goes_left`, model-wire field 10) because
  prediction has no binning.
- **Prediction**: membership in the node's left set routes left; any
  other value — other codes, unseen codes, non-integer values — routes
  right; missing follows the learned `nan_goes_left`.
- **Pairing rules**: a monotonic constraint on a categorical feature is
  rejected (codes have no order to constrain); indices are validated
  against `n_features` at train time.
- **SHAP**: the path explainer walks thresholds, so the covenant_ml
  decoder REFUSES a model containing a categorical node with a clear
  error rather than mis-attributing.

## The load-bearing test

Codes `[0, 1, 2, 3]` labelled `[1, 0, 1, 0]` — an alternating,
non-ordinal pattern NO threshold over code order can separate in one
split. A categorical depth-1 stump separates it perfectly and its root
carries exactly the `{0, 2}` vs `{1, 3}` partition (structural assert,
`threshold` = null); the numeric stump on identical data misclassifies
at least one code. Unseen code 9 lands bit-exactly in the non-member
leaf; the model JSON round-trips with identical predictions; both
growers learn the partition; training is deterministic.

## Equivalence gate: PASS, byte-for-byte

With `categorical_features = null` the four-arm benchmark reproduces the
2026-08-22 knob-identity manifest exactly — 112/112 quality values and
leaf counts across all 16 (model, variant, seed) arms — through both the
binning refactor (per-feature edge computation extracted; mixed
assignment loop) and the split-module refactor.
Manifest: `BENCHMARK_MANIFEST_2026-08-23_p3_categorical_identity.json`.

## Artifact retrains (19-field config + 10-field nodes), all EXACT

Second serde break of P3: config field 19 (`categorical_features`) and
tree-node field 10 (`categories_goes_left`) are required on the wire, so
all five service artifacts retrained again and reproduced their recorded
numbers exactly: active_cgbm 0.7790/0.7142 (round 16, spw 1.655), taiwan
production 0.9451 (98 trees) + sweep best 0.9364, us production 0.7848
(14 trees) + sweep best 0.8155.

## Gates

cargo fmt / clippy `-D warnings` / 1496 tests / 100.00% segment
coverage; cleargbm 227 passed, 100.00%; covenant_ml 2436 passed,
100.00%; covenant-radar-api 2588 passed, 100.00%. Files kept under the
600-line cap by real splits: split/mod.rs → threshold.rs + categorical.rs;
types/mod.rs → histogram.rs; the split and tree serde test batteries into
per-concern files; cleargbm-python's test_types_model into config and
model halves.
