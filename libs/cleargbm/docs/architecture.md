# ClearGBM Architecture

ClearGBM is a strict-typed Python API over the Rust gradient-boosting core in [`cleargbm_rs`](../../cleargbm_rs/). All computation happens in Rust; the Python surface validates inputs at the boundary, marshals config into the Rust-side dict shape, and hands opaque model handles back to callers.

There is **no Python fallback** and **no per-primitive hook layer**. The Python side is not a math library.

## Package layout

```
libs/cleargbm/
├── pyproject.toml           # poetry backend; path-deps on ../cleargbm_rs
├── Makefile                 # lint + test + check targets
├── docs/
│   ├── architecture.md      # this file
│   ├── VALIDATION_REPORT_2026-07-20.md
│   ├── BENCHMARK_RESULTS_2026-07-20.md   # pre-refactor baseline
│   ├── BENCHMARK_RESULTS_2026-07-21.md   # post-refactor re-run
│   ├── BENCHMARK_MANIFEST_2026-07-20.json
│   ├── BENCHMARK_MANIFEST_2026-07-21.json
│   ├── HANDOFF_BENCHMARK_AND_VALIDATE.md  # completed handoff
│   └── RUST_ONLY_REFACTOR.md              # completed refactor plan
├── scripts/
│   ├── autotune.py          # small hyperparameter sweep (calls train_gradient_boosting)
│   ├── benchmark.py         # synthetic-data timing suite
│   └── guard.py             # monorepo guard runner
├── src/cleargbm/
│   ├── __init__.py          # empty; consumers import from ensemble/types
│   ├── _rust.py             # Protocol-typed __import__("cleargbm_rs")
│   ├── ensemble.py          # public API: train_gradient_boosting, predict_proba, predict_raw
│   ├── types.py             # re-export barrel for _types_*
│   ├── _types_json.py       # JSONValue + narrow_json_to_* validators
│   ├── _types_tree.py       # TreeNode, DecisionTree TypedDicts (decode target for JSON)
│   ├── _types_model.py      # GradientBoostingConfig, GradientBoostingModel, TrainingProgress
│   ├── _types_buffer.py     # buffer TypedDicts (still referenced by covenant_ml SHAP decode)
│   ├── _types_explain.py    # explanation TypedDicts (still referenced by covenant_ml)
│   ├── _types_tuning.py     # autotune TypedDicts (used by scripts/autotune.py)
│   ├── _hooks_guard.py      # guard script hooks (test DI)
│   └── py.typed             # PEP 561 marker
└── tests/
    ├── conftest.py          # config factory + fixtures (no hook-reset)
    ├── test_ensemble.py     # train + predict boundary tests
    ├── test_types.py        # TypedDict encode/decode roundtrips + require_* error paths
    ├── test_scripts_autotune.py
    ├── test_scripts_benchmark.py
    └── test_scripts_guard.py
```

**Total Python surface:** ~855 statements. 100% branch coverage. 182 tests.

## The Rust boundary

Every call into Rust routes through `cleargbm._rust`, which does exactly one dynamic import at module load and pins each callable to a `Protocol` type so mypy sees precise signatures:

```python
# cleargbm/_rust.py (excerpt)
class _TrainProto(Protocol):
    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64] | None,
        y_val: NDArray[np.int64] | None,
        config: dict[str, int | float | bool | list[int] | None],
        feature_names: list[str],
    ) -> PyGbmModelProto: ...

_native_mod = __import__("cleargbm_rs")
train_gradient_boosting_rs: _TrainProto = _native_mod.train_gradient_boosting_rs
```

This pattern satisfies the workspace's strict-typing rules (`disallow_any_expr`, etc.) without requiring `cast` or `type: ignore`. If `cleargbm_rs` is not installed, `_rust.py` raises `ImportError` at cleargbm import time.

## What the Rust core does

Everything performance-critical. Full list of primitives exposed at `cleargbm_rs`:

- `train_gradient_boosting_rs` — single-call full training loop (binning → boosting → early stopping)
- `predict_proba_model_rs` — batch probability prediction
- `predict_raw_model_rs` — batch raw log-odds prediction
- `py_gbm_model_to_json_rs` / `py_gbm_model_from_json_rs` — model serialization
- `py_gbm_model_feature_importances_rs` — split-count feature importance
- `py_gbm_model_n_trees_rs` / `py_gbm_model_n_classes_rs` — introspection
- Subprimitives (`build_tree_rs`, `build_histogram_rs`, `subtract_histogram_rs`, `predict_tree_rs`, `predict_ensemble_rs`, `precompute_feature_bins_rs`, `bin_samples_rs`, `compute_bin_edges_rs`, `sigmoid_rs`, `sigmoid_array_rs`, `binary_log_loss_*_rs`) — mostly unused now that training runs as a single native call; retained for the autotune script and for direct Rust callers.

For the Rust-side architecture, see [`cleargbm_rs/README.md`](../../cleargbm_rs/README.md).

## Variant trialing

Improvements land as *arms*, measured against a baseline, and the loser is
deleted. This section is the procedure; it is a precondition for any
performance or algorithm work, not documentation of one.

### The variant axis

`GradientBoostingConfig.growth_strategy` (`"depth_wise" | "leaf_wise"`) is the
first algorithmic variant parameter, paired with `num_leaves`. It crosses every
boundary as the same string — Python TypedDict, the dict handed to
`train_gradient_boosting_rs`, and the config serialized inside a saved model —
so a policy has exactly one spelling everywhere it appears. Contrast
`monotonic_constraints`, which is ints in Python and variant names in JSON;
that split is a wart, not a pattern to copy.

Three rules make the axis trustworthy:

- **No implicit default.** Both keys are required at every layer. A missing key
  is an error, never a silent `depth_wise` run — an arm that meant to name a
  policy and quietly got another one produces a mislabelled measurement, which
  is worse than no measurement.
- **The budget is paired with the policy, not ignored under the wrong one.**
  `leaf_wise` without `num_leaves` is rejected (best-first growth has no depth
  to bound its shape, so the budget is its only capacity limit), and
  `num_leaves` under `depth_wise` is rejected rather than accepted and unused.
  A run that reports a knob it did not honour is the same defect class as a
  missing policy.
- **Each rule has one owner.** Python validates each field's own type and
  range; the cross-field pairing is enforced once, in Rust
  (`GradientBoostingConfig::new`), so the two layers cannot drift.

`depth_wise` behaviour is unchanged by the axis: it still passes an unbounded
leaf count to the builder, so every manifest recorded before the axis existed
remains comparable.

### Best-first growth

`leaf_wise` is implemented in `cleargbm_rs/src/tree/leafwise.rs`, the sibling of
the depth-wise `builder.rs`. Both consume the same `BuildTreeInput` and produce
the same `Tree`; they differ only in the order nodes are chosen for splitting.

The loop keeps a frontier of splittable leaves, each carrying the best split it
would make, takes the largest gain, and evaluates only the two new children —
every other candidate's best split is unaffected by a split elsewhere in the
tree, which is what makes best-first affordable (Shi 2007; the shape LightGBM's
`SerialTreeLearner` uses). Sibling-subtraction is retained.

A node that cannot be split — depth budget reached, too few samples, or no
positive-gain split — is **removed** from candidacy rather than gain-poisoned.
Shi and LightGBM differ here, and the two only differ when a blocked leaf could
later become splittable. None can: depth never decreases, a node's sample count
never grows, and its histograms never change once built. The cheaper handling
is the equivalent one, not an approximation.

The cost is memory: each frontier candidate retains its histograms, so peak
histogram memory scales with the leaf budget rather than with tree depth.
Depth-wise holds one root-to-node path; leaf-wise holds the whole frontier.

The correctness anchor is
`test_leaf_wise_matches_depth_wise_when_the_budget_never_binds`. Given a budget
no tree of that depth can reach, both policies exhaust the same set of
splittable nodes, so their predictions must agree bit for bit. That is the
property making leaf-wise an ordering change rather than a different learner,
and it is what any future edit to either builder has to keep true.

### Two gate types

Every lever declares which gate it is under **before** measurement. Choosing
after the numbers arrive is how a quality regression gets reclassified as an
acceptable tradeoff.

| | Pure-perf lever | Algorithmic lever |
|---|---|---|
| Examples | prefetch, histogram buffer interleave, the Lever-1 ordered-gradients refactor | leaf-wise growth, GOSS, an added `gamma` term |
| Gate | Output **bit-identical** to baseline | Quality **may** change |
| Evidence | Extend the equivalence tests in `cleargbm_rs/src/histogram/tests/unit_tests.rs` — AUC and log-loss identical across ≥3 seeds | Paired per-seed comparison on identical company-disjoint splits (the ablation methodology) |
| Failure | Any bit difference kills the lever | Judged on the declared objective, with quality regression as the guarded downside |

For `leaf_wise` specifically, the objective is **fit time at statistically tied
quality**, not a quality gain. That is a measured expectation, not a guess: see
[`EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md`](EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md),
where XGBoost was used as an instrument (it implements both policies, so
everything but the policy is held constant) and leaf-wise growth monotonically
*hurt* AUC on the bankruptcy workload.

A variant arm should run on at least one large-n and one small-n dataset. On
the small ones in that experiment, `min_child_weight` stopped growth before any
leaf budget bound, so the policy never engaged and every arm came out
bit-identical — a null that is only visible if the sweep includes it, and is
otherwise silently averaged into a "no effect" conclusion.

### Running an arm

`covenant_ml.benchmarking` compares a *list* of arms, not a fixed pair.
`make_trainers` returns the ClearGBM baseline, `cleargbm@leaf_wise`, and
LightGBM; `make_baseline_trainers` returns only the two the pre-variant
manifests compared, as a separate entry point so a two-series manifest is never
mistaken for a three-arm run whose third arm failed.

Manifest schema 2 records `position: int` per result rather than
`ran_first: bool`. A boolean cannot describe an ordering over three arms, and
the ordering matters: whichever arm runs first at a seed gets the coolest CPU.
The runner rotates by one slot per seed, so over any `k` consecutive seeds each
of `k` arms occupies each slot exactly once. Two arms sharing a name is
rejected — a manifest is grouped by arm name, so duplicates would merge two
configurations into one series silently.

### The procedure

**branch → measure → collapse to the winner.** Variants are experiment arms,
not permanent forks. Lever 1 is the precedent: the `Option<>` fallback path was
deleted in the same week it lost, and deleting it bought 9% on its own. A
variant left in place after its arm loses is a second compute path that every
later change has to keep alive.

## What the Python surface does NOT do

The following used to live in cleargbm and were retired in Phase C (2026-07-21):

- Per-primitive hook layer (`_hooks_*.py`) — dispatch table for Python-fallback vs Rust-adapter routing. Retired: Rust is the only path.
- `_rust_adapters.py` / `_rust_native_adapters.py` — the wiring that put Rust callables into the hook table. Retired.
- `parallel.py`, `split.py`, `tree.py`, `histogram.py`, `losses.py`, `explain.py`, `buffers.py` — pure-Python implementations of GBM primitives. Retired: consumers who want an ML computation call Rust.
- `_test_hooks.py` re-export barrel — the hooks no longer exist.
- `train_gradient_boosting` (Python-loop version) — was a co-existing Python-orchestrated alternative to the native call. Retired; the sole `train_gradient_boosting` in `ensemble.py` calls Rust directly.

Total code deleted: ~10,000 lines (source + tests). Total tests: 555 → 182 (no coverage loss — the deleted tests exercised code that no longer exists).

## Testing

- **Test suite:** 182 tests, 100% branch coverage. Enforced by `make check`.
- **What's tested:** TypedDict encode/decode roundtrips + `require_*` error paths (test_types.py); `train_gradient_boosting` + `predict_proba` + `predict_raw` on a small synthetic dataset (test_ensemble.py); the three CLI scripts (autotune/benchmark/guard).
- **What's NOT tested here:** Rust correctness. That lives in `cleargbm_rs`'s own cargo tests (1,485 Rust tests, cargo llvm-cov segment coverage 100%).

## Consumers

The primary consumer today is [`covenant_ml`](../../covenant_ml/) via its `ClearGBMBackend`:

```
libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py
  ├── imports: cleargbm.ensemble.train_gradient_boosting, predict_proba
  ├── imports: cleargbm.types.GradientBoostingConfig
  └── imports: cleargbm_rs.py_gbm_model_{to_json,from_json,feature_importances,n_trees}_rs
```

`covenant_ml.explainers.cleargbm_shap` also depends on `cleargbm.types` (via a JSON-decode shim that translates Rust-side model JSON into the Python-side `GradientBoostingModel` TypedDict shape that SHAP's tree walker consumes).

## Related documents

- [`BENCHMARK_RESULTS_2026-08-19_growth_variants.md`](BENCHMARK_RESULTS_2026-08-19_growth_variants.md) — first `cleargbm@leaf_wise` measurement, and the EcoQoS throttling defect that invalidates earlier agent-run fit times
- [`BENCHMARK_RESULTS_2026-07-21.md`](BENCHMARK_RESULTS_2026-07-21.md) — post-refactor performance vs LightGBM
- [`VALIDATION_REPORT_2026-07-20.md`](VALIDATION_REPORT_2026-07-20.md) — correctness audit (bit-for-bit checks, sibling subtraction verification)
- [`RUST_ONLY_REFACTOR.md`](RUST_ONLY_REFACTOR.md) — the completed refactor plan
- [`HANDOFF_BENCHMARK_AND_VALIDATE.md`](HANDOFF_BENCHMARK_AND_VALIDATE.md) — the original scoping handoff
