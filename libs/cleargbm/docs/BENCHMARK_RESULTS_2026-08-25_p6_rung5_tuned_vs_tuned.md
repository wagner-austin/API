# Rung 5: tuned vs tuned — the P6 capstone (2026-08-25)

The charter decides "better than any other model" by measurement, so
the P6 farm's last rung asked the deployment question directly: each
dataset at its best-known feature preset, ClearGBM and LightGBM each
tuned by Optuna TPE over its OWN search space, at matched trial
budgets (100 trials; 40 on the heavy us-full, whose ~800 engineered
features price each trial at ~64s against the unprotected 60-minute
free-partition cap — the preflight guard refused the first, 240-minute
submission for lacking requeue+checkpointing, exactly as designed).
Eight members, 620 trials, 0 pruned, 0 failed. Best validation AUC:

| dataset (preset) | cleargbm | lightgbm | verdict | cleargbm dial |
|---|---|---|---|---|
| taiwan (ratios_only) | **0.9640** | 0.9581 | ClearGBM +0.0059 | none (denom 1) |
| us (full) | 0.8501 | **0.8911** | LightGBM +0.0410 | denom 64 |
| polish (none) | 0.9646 | **0.9658** | LightGBM +0.0012 | denom 64 |
| kaggle_gmc (log_only) | **0.8708** | 0.8699 | ClearGBM +0.0009 | denom 256 |

Honest readings, in order of importance:

- **The us gap is a tuning-surface asymmetry, and it is named.**
  LightGBM's search space tunes num_leaves, colsample_bytree,
  feature_fraction, reg_alpha and reg_lambda; ClearGBM's trial config
  PINS max_features=None, colsample_bytree=None, reg_lambda=0.0 —
  knobs the ENGINE has (landed and knob-sensitivity-tested in P3) that
  the tuning surface never samples. On ~800 engineered features,
  feature subsampling and L2 are exactly the levers that matter, and
  the four-point gap is what leaving them off the table costs. The
  successor work item is search-space parity: sample
  max_features/colsample_bytree, reg_alpha/reg_lambda, and the growth
  axis in ClearGBM's space. Until that is measured, us's tuned-vs-tuned
  entry stands as a LightGBM win.
- **Elsewhere the engines trade blows at the millipoint.** taiwan
  flips decisively to ClearGBM (0.9640 — the program's best taiwan
  number, +6.5 points over the pre-knob-closure era), kaggle flips to
  ClearGBM at its best-ever 0.8708, polish stays LightGBM's by 0.0012.
- **The dial behaves like a dial.** At 100 trials three of four
  winners chose a floor (us/polish denom 64, kaggle 256) and taiwan's
  winner chose NONE — a tuned axis the sampler uses where it pays and
  drops where it does not, which is the whole point of building it
  honestly instead of defaulting it.

Artifacts: per-member logs and optimal-configs (suffixed filenames,
the editable-install fix at work) in
`tools/hpc3/runs/results/p6-rung5/`; sweep doc
`sweep-cleargbm-p6-rung5.json`.
