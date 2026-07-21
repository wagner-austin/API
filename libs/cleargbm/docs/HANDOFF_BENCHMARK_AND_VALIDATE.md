# Handoff: ClearGBM benchmark + in-depth validation

**Author of handoff:** prior AI session (2026-07-20)
**For:** next AI session (fresh context — read this whole doc before starting)
**Repo root for this work:** `~/PROJECTS/api/libs/cleargbm/`
**Companion package (already integrated):** `~/PROJECTS/api/libs/covenant_ml/`

---

## Why this handoff exists

`ClearGBM` is a from-scratch gradient-boosting library:
- Python orchestration in `libs/cleargbm/src/cleargbm/`
- Rust core in `libs/cleargbm_rs/` (histogram building, tree construction, prediction, PyO3 bindings)
- Already wrapped as a `ClassifierBackend` in `covenant_ml/backends/cleargbm/backend.py`
- Also has a SHAP explainer at `covenant_ml/explainers/cleargbm_shap.py`

Existing README language implies "LightGBM-style" performance and correctness. **But no external benchmark exists.** The `scripts/benchmark.py` file only self-benchmarks ClearGBM on synthetic data (different `n_jobs`/`max_bins` configs) — it never trains LightGBM on the same data for comparison.

Austin needs a defensible answer to: **"How does ClearGBM compare to LightGBM on real data — accuracy, AUC, training time, inference time?"** And a deeper answer to **"Is the model actually correct?"** — checking that ClearGBM's outputs match a known-good implementation on the same inputs.

---

## Goals (in priority order)

1. **Correctness validation.** Prove ClearGBM produces sensible outputs. This is more important than "is it fast" — a broken model is worse than a slow one.
2. **Head-to-head benchmark against LightGBM.** Same dataset, same train/val/test split, same objective. Report accuracy + AUC + training wall-clock + inference wall-clock. Report honestly — if ClearGBM is 10x slower and 3% less accurate, publish that.
3. **README update.** Once numbers are in, write an honest "Benchmarks" section into `libs/cleargbm/README.md`. Delete any language that overclaims performance.
4. **(Optional) Blog post draft.** If the results are interesting, draft a HN-quality blog post about the build + benchmark process. Save to `~/PROJECTS/dashboards/blog/` (Austin's blog directory pattern from ivy/ corpus).

**Non-goals:**
- Do NOT try to make ClearGBM faster/more accurate to "win" against LightGBM. LightGBM is 8+ years of world-class C++ optimization. The value of ClearGBM is interpretability, testability, and pedagogy — not beating LightGBM at its own game. Benchmark honestly, report honestly.
- Do NOT publish the benchmark numbers publicly (README / blog) until Austin has reviewed them.
- Do NOT modify ClearGBM's public API in service of benchmarking. If a benchmark reveals an API gap, note it but don't fix it in this session.

---

## Datasets available (LOCAL first — no downloads needed)

### Primary: `american_bankruptcy.csv` (78,683 rows, binary classification)

- **Path:** `~/PROJECTS/api/libs/covenant_ml/tests/data/american_bankruptcy.csv`
- **Task:** Predict `status_label` (`alive` / `failed`) from 18 financial features
- **Structure:** Panel data (company × year). `company_name`, `status_label`, `year`, then `X1..X18`
- **Scale:** 78K rows is enough to see meaningful timing differences (small enough to iterate fast, large enough that both models will train non-trivially)
- **Class imbalance:** ~90% `alive` / 10% `failed` — realistic for a fintech binary problem. Use stratified splits.

**Recommended split:** Panel-aware split (time-based). Older years for train, newer years for val/test. This mirrors real production usage.

### Secondary: Kaggle Amex Default (small sample already local)

- **Path:** `~/PROJECTS/api/libs/covenant_ml/tests/datasets/fixtures/timeseries_amex_sample/`
- **Files:** `data.csv`, `labels.csv`, plus cached `features.parquet` / `labels.parquet` / `meta.parquet` in `.cache/`
- **Task:** Binary classification (default prediction) — same task as the Kaggle competition
- **This is a SAMPLE**, not the full Kaggle dataset. Small — meant for tests, not benchmarks. Fine as a secondary sanity dataset but the main benchmark should be `american_bankruptcy`.

### Full Kaggle Amex Default (would need download — SKIP unless benchmark is inconclusive)

- Registry entry `kaggle_amex_default` in `covenant_ml/backends/registry.py`
- The full Kaggle dataset is ~50GB and would need Kaggle API credentials
- Austin's `~/PROJECTS/api/amex_1st_place/` has the 1st-place solution code (LGB + NN ensemble) — uses same data
- **Don't download unless the local datasets don't produce usable results**

### Fallback: UCI datasets (if all above fail)

- UCI Adult (~48K rows, binary income prediction) — via `sklearn.datasets.fetch_openml('adult')`
- UCI Higgs Boson (~11M rows, binary particle physics) — larger, better timing signal, needs download
- Use only if local datasets have data-quality issues that block benchmarking

---

## Existing infrastructure to reuse (READ THIS BEFORE WRITING NEW CODE)

**Backend interface (both models implement it):**
- `covenant_ml/src/covenant_ml/backends/cleargbm/backend.py` — ClearGBM `ClassifierBackend`
- `covenant_ml/src/covenant_ml/backends/lightgbm/` — LightGBM `ClassifierBackend`
- Registry: `covenant_ml/src/covenant_ml/backends/registry.py`

Because both are wrapped in the same interface, benchmarking = load dataset → fit both backends with equivalent hyperparameters → predict → score. Should be a ~200-300 line script.

**Dataset loaders:**
- `covenant_ml/tests/datasets/loaders/test_timeseries_csv_loader.py` — shows how to load the Amex fixture
- Read the loader before writing your own — reuse the existing pattern.

**ClearGBM's own test surface:**
- `libs/cleargbm/tests/` — 17 test files covering ensemble, explain, histogram, losses, parallel, rust adapters, split, tree, types, buffers
- **`test_rust_adapters.py` and `test_rust_native_adapters.py`** — these test the Python↔Rust boundary. Read them to understand what the Rust core actually receives + returns.
- All tests pass at 100% branch coverage per repo convention. If any fail during your session, THAT IS THE VALIDATION PROBLEM — fix or report before benchmarking.

**Existing benchmark script:**
- `libs/cleargbm/scripts/benchmark.py` — self-benchmark on synthetic data. **Model your new benchmark on this file's structure** (NamedTuple result types, sys.stdout writes via `_write()` to avoid the bare-print guard rule, `argparse` for CLI, etc.). Don't reinvent the shape.

---

## Recommended methodology (do these in order)

### Phase 1: Correctness validation (deepest work — do NOT skip)

**Goal:** Prove ClearGBM is doing what it claims. Not "runs without errors" — actually producing correct outputs.

1. **Bit-for-bit or close-to-bit reproduction on a tiny problem.**
   - Take an 8-row, 2-feature XOR-ish dataset. Manually compute what the tree splits should be.
   - Train ClearGBM with `n_estimators=1, max_depth=3`.
   - Verify the learned tree matches what you computed by hand.
   - Report: does it match? If not, WHERE does it diverge?

2. **Sibling-subtraction check.**
   - README claims "sibling subtraction" for histogram computation. Verify: compute the parent histogram once, the left child histogram, then derive the right child histogram via subtraction. Check it matches the histogram computed independently on the right child's samples.
   - Location: `src/cleargbm/histogram.py` + `libs/cleargbm_rs/src/histogram/`
   - Report: is sibling subtraction actually happening? Is it correct?

3. **LightGBM-style O(K) split finding check.**
   - README claims "LightGBM-style O(K) split finding instead of O(n log n) sorting."
   - Verify: split finding operates on histogram bins (K = `max_bins`) not on sorted feature values (n = rows). Read `src/cleargbm/split.py` and `libs/cleargbm_rs/src/split/`.
   - Report: complexity claim accurate?

4. **Loss function correctness.**
   - `src/cleargbm/losses.py` implements binary log-loss with gradients + hessians.
   - Verify: gradients + hessians match textbook formulas (dL/dp = p - y for binary log-loss, second derivative = p(1-p)).
   - Compare against scipy or a written-out implementation.

5. **Predict-proba sanity.**
   - Train ClearGBM on `american_bankruptcy`, predict on holdout. Check:
     - Predictions are in [0, 1]
     - Mean predicted probability ≈ base rate of positive class in training data
     - AUC > 0.5 (i.e., the model actually learned something)
   - If AUC ≈ 0.5, the model isn't learning. Bigger problem than benchmarking.

6. **SHAP explainer sanity check.**
   - `covenant_ml/explainers/cleargbm_shap.py` — compute SHAP values on 100 test samples.
   - Verify: SHAP values sum to (prediction - base value) per instance (this is the SHAP additivity property).
   - Verify: mean absolute SHAP value per feature roughly ranks features by importance (should correlate with LightGBM's feature importances on same data).

7. **Determinism check.**
   - Train ClearGBM twice with the same seed, same data. Verify predictions are bit-identical.
   - If not deterministic → serious bug, report before continuing.

### Phase 2: Benchmark against LightGBM

**Goal:** Apples-to-apples comparison on `american_bankruptcy`.

1. **Split the data.**
   - Panel-aware: use years 1999-2013 for train, 2014-2015 for val, 2016-2018 for test (or whatever ranges the dataset covers — check the year range first).
   - Fall back to stratified random 70/15/15 if panel structure is unclear.
   - Log exact split sizes + class balance per split.

2. **Configure both models with equivalent hyperparameters.**
   - Match: `n_estimators`, `learning_rate`, `max_depth` (or `num_leaves`), `min_data_in_leaf`, `max_bins`, `regularization`.
   - Match objective (`binary`) and metric (`binary_logloss` or `auc`).
   - Match early stopping strategy (or disable in both).
   - **Document any hyperparameter that doesn't have an exact equivalent** in the README section you write later.

3. **Time each phase separately:**
   - Data preprocessing (bin edges computation for both)
   - Training (per epoch/iteration if possible, else total wall-clock)
   - Prediction on test set

4. **Measure quality:**
   - Accuracy (thresholded at 0.5)
   - AUC-ROC
   - AUC-PR (matters more for imbalanced classes)
   - Log-loss
   - Confusion matrix

5. **Sweep across dataset sizes:**
   - Train on 10K, 30K, 50K, 78K rows (subsampled from `american_bankruptcy`).
   - Report timing per size — shows if ClearGBM scales linearly, superlinearly, or has a threshold behavior.

6. **Sweep across `n_estimators`:**
   - 50, 100, 300, 500 trees. Do both models overfit at similar rates?

7. **Report format:**
   - Save results to `docs/BENCHMARK_RESULTS_2026-XX-XX.md`
   - Include a summary table + rationale + methodology + full raw numbers
   - Explicitly call out anything that surprised you

### Phase 3: README update

- Delete any performance-superiority language ("high performance" is fine as an *implementation* description with the specific mechanisms named; not fine as an implied comparison).
- Add a "Benchmarks" section citing the results doc.
- Add a "Non-goals" section: "Not designed to beat LightGBM/XGBoost. Value is interpretability + full type safety + Rust-backed correctness verification."
- Diff the README, show Austin BEFORE PUSHING.

### Phase 4 (optional): Blog post draft

If the benchmark story is interesting — e.g. "ClearGBM matches LightGBM on small datasets but falls off at scale," or "SHAP additivity holds bit-perfectly," or "sibling subtraction gave a 3x speedup on the histogram phase" — draft a blog post to `~/PROJECTS/dashboards/blog/cleargbm-vs-lightgbm/index.html` + `cleargbm-vs-lightgbm.md`.

Follow Austin's `ivy/` pattern for structure. Do NOT publish. Leave as a draft for his review.

---

## Auxiliary data / references

**`amex_1st_place`** at `~/PROJECTS/api/amex_1st_place/`:
- Kaggle Amex Default 1st-place solution (2022 competition, but README is set up for python==3.7.10)
- Uses LGBM + NN ensemble (`S5_LGB_main.py`, `S6_NN_main.py`, `S7_ensemble.py`)
- Has feature engineering pipelines (`S1_denoise.py` through `S4_feature_combined.py`)
- `input/` dir is empty (only `git.init`) — the actual competition data was never checked in
- **Value for this handoff:** you can crib the LightGBM hyperparameter configuration from `S5_LGB_main.py` if you want a "production-tuned LightGBM baseline" (it's the winning config). This may bias LightGBM in its favor for the benchmark, but that's actually what you want — see how ClearGBM does against a well-tuned LightGBM.

**ClearGBM docs to read:**
- `docs/architecture.md` — full architecture overview
- `docs/rust-core-transition-plan.md` — Phase 1-9 Rust migration plan, useful for understanding the Rust boundary
- README.md — current claims (some of which you're validating)

---

## What to hand back to Austin

At end of session, produce:

1. **`docs/BENCHMARK_RESULTS_2026-XX-XX.md`** — full benchmark writeup with methodology, numbers, and interpretation
2. **`docs/VALIDATION_REPORT_2026-XX-XX.md`** — correctness validation findings, with a red/yellow/green summary per check
3. **Draft README update** (as a diff, not yet applied) — new "Benchmarks" and "Non-goals" sections, and any tone-downs on existing claims
4. **`scripts/benchmark_vs_lightgbm.py`** — the runnable comparison script (follow existing `scripts/benchmark.py` structure)
5. **Blog post draft** (optional) — if the story is interesting
6. **Findings log** — anything surprising, any bugs found, any spots where the README claims don't match code reality. Save to `docs/FINDINGS_2026-XX-XX.md`.

---

## Guard rules to respect (this is a monorepo)

- **100% test coverage required** for anything you add. See `libs/cleargbm/Makefile` for `make check` — must pass.
- **No `Any`, no `cast`, no `# type: ignore`** — strict mypy per monorepo convention.
- **No mocks** in tests. Use `_test_hooks` DI pattern. See `libs/cleargbm/scripts/_test_hooks.py`.
- **No bare `print()`** — use `sys.stdout.write()` (there's a guard rule enforcing this). See `_write()` helper in `scripts/benchmark.py`.
- **No modifying `pyproject.toml`** without explicit permission — poetry lock re-resolution is expensive.

If `make check` fails after your changes, fix the underlying issue. Do not skip tests or add `noqa`.

---

## Time estimate

- **Phase 1 (correctness):** 3-5 hours. Depth matters more than speed.
- **Phase 2 (benchmark):** 2-4 hours (harness setup + runs + writeup).
- **Phase 3 (README):** 30-60 minutes.
- **Phase 4 (blog draft):** 2-3 hours (only if results are publishable).

Total: 6-12 hours of focused work. Use judgment — if Phase 1 uncovers a serious correctness bug, STOP and hand back to Austin before continuing to Phase 2.
