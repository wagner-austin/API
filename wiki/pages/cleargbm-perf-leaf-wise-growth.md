---
title: ClearGBM perf — leaf-wise tree growth
tags: [ml, cleargbm, rust, performance, tree]
related:
  - "[[cleargbm-histogram-split-path]]"
  - "[[cleargbm-leaf-normalized-benchmarking]]"
source_paths:
  - libs/cleargbm/docs/EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-19_growth_variants.md
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-21_four_arm.md
  - libs/cleargbm_rs/src/tree/builder.rs
  - libs/cleargbm_rs/src/tree/leafwise.rs
  - libs/cleargbm_rs/src/tree/mod.rs
  - libs/cleargbm_rs/src/training/config.rs
source_git_blobs:
  "libs/cleargbm/docs/EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md": 570404c5fe5b67cd737dd3cddc41a29a614544e2
  "libs/cleargbm_rs/src/tree/builder.rs": 8f518f965e4615c42267996ec73c087b838be84c
  "libs/cleargbm_rs/src/tree/leafwise.rs": 0e540b746f0db6468f0cb6522ce43781b9a8c244
  "libs/cleargbm_rs/src/tree/mod.rs": f2c7332be3dae4cc20b758b4c11a561626ec4401
  "libs/cleargbm_rs/src/training/config.rs": 85fb08e891055d724122b6ff42e8d0af346e4639
fact_checked: "2026-08-21"
confidence: high
hubs: [libs]
---

# ClearGBM perf — leaf-wise tree growth

Replace ClearGBM's depth-first tree growth with LightGBM's leaf-wise (best-first) strategy: at every step, expand the leaf with the highest split gain across the whole tree, not the deepest-yet-unfinished branch. An earlier revision framed this as a capacity gain; the 2026-08-17 instrument measurement below reversed that — on this workload extra capacity *hurts*, and the prize is **reaching tied quality with fewer leaves, i.e. less tree-building work**. ClearGBM currently builds 1.523× LightGBM's leaves for a quality that is a statistical tie[^8][^1], and the quality ceiling is reachable at roughly half ClearGBM's leaf spend[^11].

**Confidence: medium.** Impact on this benchmark is ambiguous — quality is already a statistical tie with LightGBM[^1], so the accuracy ceiling is small. The *work* ceiling is not small: ClearGBM builds 1.52× LightGBM's leaves for that tied quality[^8], and leaf-wise is the change that closes it.

**Measured 2026-08-17, before building it:** using XGBoost as the instrument (it implements both growth policies), leaf-wise growth monotonically *hurt* quality on the bankruptcy workload — depthwise at 22.8 mean leaves beat lossguide at 31 and 47 leaves on AUC-ROC, AUC-PR and log-loss — and on two smaller datasets (Taiwan bankruptcy, German credit) all arms were identical because min-leaf regularization stopped growth at ~4.5 leaves before any budget bound. Full protocol, numbers and confounds: `libs/cleargbm/docs/EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md`[^11]. Consequence for this page: the prize is **work reduction at statistically tied quality** (the ceiling is reachable at ~23 leaves; ClearGBM spends ~47), not a capacity gain, and the variant should be judged on fit-time-at-tied-quality with quality regression as the guarded downside.

The interpretability objection that previously appeared here has been **withdrawn** — it rested on ClearGBM producing balanced trees, which measurement disproves[^7]. See § "Interpretability cost".

## IMPLEMENTED and measured (2026-08-20 / 2026-08-21)

Everything below the fold was written while this was a proposal; it is kept as the design record. What actually landed (commits `2a55899c` cleargbm_rs, `1a04bb1e` cleargbm, `bd55e8d6` covenant_ml, closing agent-board task `453c9234`):

- **The axis is `growth_strategy: depth_wise | leaf_wise` paired with `num_leaves: int | None`** — not the `max_leaves: Option<usize>` sketch below. The pairing is validated: `leaf_wise` without a budget is an error, a budget under `depth_wise` is an error, `num_leaves < 2` is an error. Depth-wise behaviour is byte-unchanged, so every pre-existing manifest stays comparable.
- **The builder is `libs/cleargbm_rs/src/tree/leafwise.rs`** — a flat frontier with ArgMax over gain (the LightGBM structure the design alternative below recorded), not the `BinaryHeap` sketch. Only the two children of a split are re-evaluated; sibling subtraction is retained. Blocked leaves are **removed** from candidacy (Shi 2007) rather than gain-poisoned; the module header proves the two are equivalent here (depth never decreases, a node's samples and histograms never change once built, so a blocked leaf can never become splittable).
- **The load-bearing test**: with an unreachable budget, leaf-wise and depth-wise must produce bit-identical predictions — growth order changes which node splits *first*, never which nodes are splittable. It does.
- **Measured twice, deterministically**: `BENCHMARK_RESULTS_2026-08-19_growth_variants.md` (three arms) and `BENCHMARK_RESULTS_2026-08-21_four_arm.md` (adds the `xgboost` arm). Leaf-wise reaches statistically tied quality at a third fewer leaves (~31 vs ~47) and is ~6% faster wall-clock; per leaf it is the *more* expensive builder (~1.4x), so the speed-up is entirely from building fewer leaves. Quality metrics reproduce bit-for-bit across the two dated runs — the deterministic-trialing property the board task demanded. The 2026-08-19 run also caught and fixed a measurement defect: Windows EcoQoS throttling stepped fit times 13x mid-run; `covenant_ml.benchmarking.power` now opts the process out and refuses to run throttled. **Follow-up 2026-08-21 (`BENCHMARK_RESULTS_2026-08-21_interleave.md`):** interleaving the histogram accumulators cut both ClearGBM arms ~16% and put per-leaf cost *under* LightGBM (0.984x); the leaf-wise arm now fits in 1.40x LightGBM's wall clock, and the remaining gap is the leaf-count policy, not the builder.

## What today's code does (depth-first)

`libs/cleargbm_rs/src/tree/builder.rs::build_tree` walks the tree via a LIFO stack[^2]:

```rust
// Line 209
let mut stack: Vec<PendingNode> = Vec::new();
stack.push(PendingNode { sample_indices, depth: 0, parent_id: None, ... });

// Line 218
while let Some(pending) = stack.pop() {
    // ... split this node ...
    // Line 364, 371: push right child then left child
    stack.push(PendingNode { /* right */ });
    stack.push(PendingNode { /* left */ });
}
```

Because Rust's `Vec::pop` is LIFO and the left child is pushed second, the left child is processed first — full depth-first traversal of the left subtree before touching the right. Every leaf gets one split before any other leaf gets two[^3].

## What leaf-wise would change (design record — superseded by the implementation above)

Replace the `Vec<PendingNode>` LIFO stack[^2] with a **max-heap ordered by best-available split gain**. At every step:

1. Compute the best split for every currently-open leaf (already done today by `find_best_split_from_histogram`; result carries a gain).
2. Push each open leaf's `(gain, PendingNode)` onto a `BinaryHeap`.
3. Pop the highest-gain leaf. Expand it. Compute the best split for each of its two children and push them back.
4. Stop when the heap is empty OR when `n_leaves >= max_leaves` OR when the highest available gain is `≤ min_gain_to_split`.

The `Vec<PendingNode>`[^2] → `BinaryHeap<(NotNan<f64>, PendingNode)>` swap is small in LOC. **Design alternative, recorded 2026-08-17 from the reference implementation:** LightGBM does not use a heap. `SerialTreeLearner::Train` keeps a flat `best_split_per_leaf_` array sized `num_leaves`, recomputes best splits only for the two leaves the previous split created, and selects with a plain ArgMax over the array; `max_depth` is enforced by writing `kMinScore` into a capped leaf's cached gain so it can never win the ArgMax (gain poisoning), leaving the selection path constraint-free. Shi's thesis, the origin of best-first induction, instead *removes* a constraint-blocked node from the node list. The two differ when a blocked leaf could later become splittable, and a leaf-wise arm must pick one and name it in the config surface. Primary-source pages for both, with line citations into the captured `serial_tree_learner.cpp` and the thesis PDF, are in the tech-wiki (`~/PROJECTS/tech-wiki`): `lightgbm-leaf-wise-growth-argmax-loop` and `shi-2007-best-first-tree-induction`. At `num_leaves` ≈ 31 the flat-array ArgMax is O(leaves) per split with no heap maintenance, and it composes with the sibling-histogram cache concern below because the recompute set is exactly the two new leaves. Whichever structure is chosen, the real work is[^2]:

- **A new stopping criterion.** Depth-first uses `max_depth` (a per-branch limit); leaf-wise needs `max_leaves` (a whole-tree limit) — otherwise leaf-wise can produce arbitrarily deep single-branch trees on adversarial data.
- **Config plumbing.** `GradientBoostingConfig` today has `max_depth`[^4] but no `max_leaves`. Add `max_leaves: Option<usize>` (`None` = disable the leaf-wise cap; falls back to `max_depth`).
- **Sibling-histogram cache invariant.** Today the depth-first code takes advantage of the fact that the sibling of a just-processed node is next on the stack; it caches the parent's histogram for the sibling-subtraction trick[^5]. Leaf-wise breaks the sibling-adjacency invariant — the sibling might not be the next leaf popped. The cache eviction policy has to change: hold the parent histogram as long as EITHER child is still an open leaf.

## Config surface (design record — the landed surface is `growth_strategy` + `num_leaves`, see above)

New field on `libs/cleargbm_rs/src/training/config.rs::GradientBoostingConfig`[^4]:

```rust
pub struct GradientBoostingConfig {
    // ... existing 12 fields ...
    /// Growth strategy: "depth_first" (default, current behavior) or "leaf_wise".
    growth_strategy: GrowthStrategy,
    /// Maximum number of leaves per tree (leaf-wise only). None = unbounded.
    max_leaves: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GrowthStrategy { DepthFirst, LeafWise }
```

Validation: when `growth_strategy == LeafWise`, either `max_leaves` OR `max_depth` must be set (raise `InvalidParameter` on the double-`None` case), mirroring the existing `max_depth < 1` guard[^4]. Both defaults preserve backward compatibility.

This config surface, extending the validated-config shape at[^4], is now specified as a prerequisite on agent-board task `453c9234` ("deterministic variant-trialing state", 2026-08-17), which also requires the benchmark factory to compare `cleargbm` vs `cleargbm@leaf_wise` vs `lightgbm` in one manifest and declares the gate type for this change: algorithmic, judged by paired per-seed quality on identical company-disjoint splits, NOT bit-identity. Implementation should route through that task rather than landing ad hoc.

Serde on `GrowthStrategy` follows the same pattern as `MonotonicConstraint` in `libs/cleargbm_rs/src/split/serde_impl.rs` — a three-variant enum serialized as a string[^9].

## Testing strategy

1. Unit tests in `libs/cleargbm_rs/src/tree/tests/` for the heap-based traversal: on a fixed synthetic dataset, assert the leaf-wise tree has the expected shape (heavier splits down the informative branch).
2. Regression guard: same synthetic dataset, `growth_strategy = DepthFirst` produces the pre-refactor tree (byte-identical serialization).
3. Cross-strategy proptest: for random data, both strategies converge to the same training loss within FP tolerance at `n_estimators → ∞`.
4. covenant_ml integration test: verify a `LeafWise`-trained model roundtrips through JSON and produces stable predictions.

## Interpretability cost

**Superseded 2026-07-24 — the premise was false.** This section previously claimed depth-first "produces balanced trees where every leaf sits at roughly the same depth", and that switching to leaf-wise would therefore cost interpretability. Direct measurement of a trained model refutes the premise[^7], so the objection does not apply to this codebase.

What a tree dump at `max_depth=5` actually shows[^7]:

- **Not balanced.** Root-to-leaf path lengths range 4–6, not a uniform depth.
- **Not full.** A full binary tree at `max_depth=6` has 64 leaves; ClearGBM measures 57.9 there, and 47.15 on the authoritative benchmark run[^8]. Stopping criteria retire branches early.
- **Not oblivious.** 13 distinct features appear at depth 5 — each node picks its own split, unlike CatBoost's symmetric trees where a whole level shares one test.

So depth-wise growth here yields exactly the irregular shape the section warned leaf-wise would introduce[^7]. The shape is already irregular; leaf-wise would change *which* branches get deep, not whether any do.

The rule-count comparison runs the other way from what was assumed: ClearGBM emits **47–58 leaves per tree against LightGBM's 31**[^8] — half again as many rules to read for statistically tied quality. On a rules-to-read measure, the depth-wise tree is the *less* readable of the two, and leaf-wise growth — which reaches equivalent loss with fewer effective splits — would likely improve it.

Nothing in the interpretability machinery depends on tree shape: `export_model_json`, split-count feature importance, and monotonic constraints all walk arbitrary trees[^10]. (TreeSHAP is **not** part of this codebase — the explain surface is a `FeatureContribution` type, and importance is split-count only, explicitly not gain-weighted[^10]. An earlier revision of this page listed TreeSHAP here; that was wrong.) The genuine interpretability lever is **oblivious trees** (uniform per-level splits), which is a different change from leaf-wise and is not what this page proposes.

Given ClearGBM's positioning as *Gradient Boosting You Can See Through*[^6], the honest framing is that today's growth strategy does not deliver that property, and leaf-wise does not take it away.

[^1]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md` § "Quality metrics" — every quality metric within seed std between cleargbm and lightgbm on the benchmark dataset.
[^2]: `libs/cleargbm_rs/src/tree/builder.rs:199,209,218` — comment "Build tree using depth-first stack", `stack: Vec<PendingNode>`, and the `while let Some(pending) = stack.pop()` loop entry.
[^3]: `libs/cleargbm_rs/src/tree/builder.rs:363-374` — right child pushed at line 364, left child pushed at line 371 (comment: "Push children to stack (right first so left is processed first)").
[^4]: `libs/cleargbm_rs/src/training/config.rs:18,50,105-110` — `pub max_depth: usize` in the params struct, `max_depth: usize` in the validated config, validation `if max_depth < 1_usize { return Err(...) }`.
[^5]: `libs/cleargbm_rs/src/tree/nodes.rs:34` — `cached_histograms: Option<Vec<HistogramBuffer>>` on `PendingNode` (declared at `libs/cleargbm_rs/src/tree/nodes.rs:18`, used from `builder.rs`); carried across the sibling boundary via the depth-first ordering.
[^6]: `libs/cleargbm/README.md` header line — *Gradient Boosting You Can See Through*.
[^7]: `libs/cleargbm/src/cleargbm/ensemble.py:195` — `export_model_json(model)`, the API this reading came from. Tree dump of a model trained on the bankruptcy dataset at `max_depth=5`, read 2026-07-24: 13 distinct `feature_index` values among depth-5 internal nodes; root-to-leaf path lengths 4–6. **The dump itself was not committed**, so the counts are reproducible via that function on the same dataset and config but are not re-readable from a stored artifact.
[^8]: `libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json` § `results[]` — 47.15 vs 30.96 leaves (leaf ratio 1.523×), being the across-seed mean of `results[].mean_leaves` over seeds `[42, 43, 44]`; recomputed from the manifest 2026-08-05, reproduces exactly. The earlier 57.9 vs 31.0 figures at `max_depth=6` are from the pre-rebuild harness and are narrated at [[cleargbm-leaf-normalized-benchmarking]] § "Measured tree-size divergence"; they are not in this manifest.
[^9]: `libs/cleargbm_rs/src/split/serde_impl.rs:8,11-27` — `use super::{MonotonicConstraint, ...}`, the `// MonotonicConstraint Serialization` banner, `impl Serialize for MonotonicConstraint`, and a string-visitor deserializer. This is the pattern a `GrowthStrategy` enum would copy.
[^11]: libs/cleargbm/docs/EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md — protocol, per-arm table, the min_child_weight hessian-vs-count confound, and the small-dataset null. Measurement logic: `libs/covenant_ml/src/covenant_ml/growth_policy/` (unit tested, 100% statement and branch); entry points `libs/covenant_ml/scripts/experiment_growth_policy_xgb_instrument.py` + `experiment_growth_policy_multi_dataset.py`, seeds 42/43/44 default. The scripts were first written under `libs/cleargbm/scripts/`, a package whose environment has none of xgboost, lightgbm or scikit-learn and cannot resolve the dataset path, so they were moved rather than left where they could not run.
[^10]: Verified 2026-07-31 against `libs/cleargbm/src` and `libs/cleargbm_rs/src` (excluding vendored `.venv`): `export_model_json` at `libs/cleargbm/src/cleargbm/ensemble.py:195`; split-count importance at `libs/cleargbm_rs/src/training/importance.rs:15`, whose module doc at `:4-10` states it counts "the split feature at any internal node across all trees, then normalize so …" and "deliberately does not depend on gain-per-split", noting gain-weighted importance as "a future enhancement"; monotonic constraints at `libs/cleargbm_rs/src/split/mod.rs` + `split/serde_impl.rs`. A case-sensitive grep for `SHAP` and `shap_values` across both source trees returns **no match** — the explain surface is `FeatureContribution` in `libs/cleargbm/src/cleargbm/_types_explain.py:34`.
