# AMEX Competition Refactor Plan

## Status: ❌ NOT STARTED

## Goal

Beat 1st place score (0.80977) on Kaggle AMEX Default Prediction.

---

## Current State vs Target

| Aspect | Current | Target |
|--------|---------|--------|
| CV strategy | Random stratified split | GroupKFold by customer_ID |
| Aggregations | mean, std, min, max | + rank, diff, time windows |
| Metric | AUC | AMEX metric (0.5 * Gini + 0.5 * D@4%) |
| Ensemble | None | Weighted ensemble with OOF optimization |

---

## Implementation Phases

### Phase 1: AMEX Competition Metric

**Location:** `libs/covenant_ml/src/covenant_ml/metrics.py`

Add `compute_amex_metric()`:
- Gini coefficient with 20x weight for subsampled negatives
- Default rate at 4% threshold
- Combined score: 0.5 * Gini + 0.5 * D@4%

### Phase 2: GroupKFold Cross-Validation

**Location:** `libs/covenant_ml/src/covenant_ml/trainer.py`

Add `group_stratified_split()`:
- Ensures no customer appears in both train and validation
- Critical for time-series data with multiple observations per entity
- Uses existing `AutoPreprocessor` for per-fold preprocessing

### Phase 3: Competition Feature Engineering

**Location:** `libs/covenant_ml/src/covenant_ml/datasets/loaders/`

| Feature Type | Description |
|--------------|-------------|
| Window aggregations | last3, last6 observations per customer |
| Rank features | Per-customer and per-month percentiles |
| Diff features | Value changes between statements |

New files needed:
- `_polars_ranking.py` - Percentile rank computation
- `_polars_diff.py` - Row-to-row differences

### Phase 4: Ensemble Pipeline

**Location:** `libs/covenant_ml/src/covenant_ml/ensemble/`

```
ensemble/
├── types.py      # EnsembleConfig, OOFPredictions
├── weighted.py   # WeightedEnsemble class
└── optimizer.py  # optimize_ensemble_weights (scipy.optimize)
```

Optimize weights using OOF predictions to maximize AMEX metric.

### Phase 5: Competition Pipeline Script

**Location:** `services/covenant-radar-api/scripts/amex/`

Full pipeline:
1. Load training data with competition features
2. Train each model with GroupKFold CV
3. Optimize ensemble weights on OOF predictions
4. Generate test predictions
5. Create submission.csv

### Phase 6: LightGBM DART Configuration

Add DART-specific search space matching 1st place:
- Very low `feature_fraction` (0.02-0.1)
- `drop_rate` and `skip_drop` parameters
- Higher `lambda_l2` regularization (10-50)

---

## Expected Results

| Stage | Expected CV Score |
|-------|-------------------|
| Current (random split) | ~0.95 (inflated - leakage) |
| After GroupKFold | ~0.82-0.85 (realistic) |
| After rank/diff features | ~0.80-0.82 |
| After ensemble | ~0.80-0.81 |

**Target:** > 0.80977

---

## Reference

1st place solution: `amex_1st_place/` (at API root)
- `S2_manual_feature.py` - Feature engineering
- `S5_LGB_main.py` - LightGBM DART config
- `utils.py:61-80` - AMEX metric

---

## Coding Standards

Same as all covenant_ml code:
- No `Any`, `cast()`, `type: ignore`
- TypedDicts for all structured data
- Protocols for dynamic imports
- 100% test coverage
- No mocks

---

*Last updated: December 2025*
