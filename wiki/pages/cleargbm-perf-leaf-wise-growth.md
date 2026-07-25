---
title: ClearGBM perf — leaf-wise tree growth
tags: [ml, cleargbm, rust, performance, tree]
related:
  - "[[cleargbm-histogram-split-path]]"
  - "[[cleargbm-leaf-normalized-benchmarking]]"
source_paths:
  - libs/cleargbm_rs/src/tree/builder.rs
  - libs/cleargbm_rs/src/tree/mod.rs
  - libs/cleargbm_rs/src/training/config.rs
fact_checked: "2026-07-24"
confidence: medium
hubs: [libs]
---

# ClearGBM perf — leaf-wise tree growth

Replace ClearGBM's depth-first tree growth with LightGBM's leaf-wise (best-first) strategy: at every step, expand the leaf with the highest split gain across the whole tree, not the deepest-yet-unfinished branch. This is a **capacity gain, not a speed gain** — leaf-wise reaches equivalent loss with fewer effective splits, so at matched `n_estimators` the model has more useful capacity per tree.

**Confidence: medium.** Impact on this benchmark is ambiguous — quality is already a statistical tie with LightGBM[^1], so the accuracy ceiling is small. The *work* ceiling is not small: ClearGBM builds 1.52× LightGBM's leaves for that tied quality[^8], and leaf-wise is the change that closes it.

The interpretability objection that previously appeared here has been **withdrawn** — it rested on ClearGBM producing balanced trees, which measurement disproves. See § "Interpretability cost".

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

## What leaf-wise would change

Replace the `Vec<PendingNode>` LIFO stack with a **max-heap ordered by best-available split gain**. At every step:

1. Compute the best split for every currently-open leaf (already done today by `find_best_split_from_histogram`; result carries a gain).
2. Push each open leaf's `(gain, PendingNode)` onto a `BinaryHeap`.
3. Pop the highest-gain leaf. Expand it. Compute the best split for each of its two children and push them back.
4. Stop when the heap is empty OR when `n_leaves >= max_leaves` OR when the highest available gain is `≤ min_gain_to_split`.

The `Vec<PendingNode>` → `BinaryHeap<(NotNan<f64>, PendingNode)>` swap is small in LOC. The real work is:

- **A new stopping criterion.** Depth-first uses `max_depth` (a per-branch limit); leaf-wise needs `max_leaves` (a whole-tree limit) — otherwise leaf-wise can produce arbitrarily deep single-branch trees on adversarial data.
- **Config plumbing.** `GradientBoostingConfig` today has `max_depth`[^4] but no `max_leaves`. Add `max_leaves: Option<usize>` (`None` = disable the leaf-wise cap; falls back to `max_depth`).
- **Sibling-histogram cache invariant.** Today the depth-first code takes advantage of the fact that the sibling of a just-processed node is next on the stack; it caches the parent's histogram for the sibling-subtraction trick[^5]. Leaf-wise breaks the sibling-adjacency invariant — the sibling might not be the next leaf popped. The cache eviction policy has to change: hold the parent histogram as long as EITHER child is still an open leaf.

## Config surface

New field on `libs/cleargbm_rs/src/training/config.rs::GradientBoostingConfig`:

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

Validation: when `growth_strategy == LeafWise`, either `max_leaves` OR `max_depth` must be set (raise `InvalidParameter` on the double-`None` case). Both defaults preserve backward compatibility.

Serde on `GrowthStrategy` follows the same pattern as `MonotonicConstraint` in `libs/cleargbm_rs/src/split/serde_impl.rs` — a three-variant enum serialized as a string.

## Testing strategy

1. Unit tests in `libs/cleargbm_rs/src/tree/tests/` for the heap-based traversal: on a fixed synthetic dataset, assert the leaf-wise tree has the expected shape (heavier splits down the informative branch).
2. Regression guard: same synthetic dataset, `growth_strategy = DepthFirst` produces the pre-refactor tree (byte-identical serialization).
3. Cross-strategy proptest: for random data, both strategies converge to the same training loss within FP tolerance at `n_estimators → ∞`.
4. covenant_ml integration test: verify a `LeafWise`-trained model roundtrips through JSON and produces stable predictions.

## Interpretability cost

**Superseded 2026-07-24 — the premise was false.** This section previously claimed depth-first "produces balanced trees where every leaf sits at roughly the same depth", and that switching to leaf-wise would therefore cost interpretability. Direct measurement of a trained model refutes the premise, so the objection does not apply to this codebase.

What a tree dump at `max_depth=5` actually shows[^7]:

- **Not balanced.** Root-to-leaf path lengths range 4–6, not a uniform depth.
- **Not full.** A full binary tree at `max_depth=6` has 64 leaves; ClearGBM measures 57.9 there, and 47.15 on the authoritative benchmark run[^8]. Stopping criteria retire branches early.
- **Not oblivious.** 13 distinct features appear at depth 5 — each node picks its own split, unlike CatBoost's symmetric trees where a whole level shares one test.

So depth-wise growth here yields exactly the irregular shape the section warned leaf-wise would introduce. The shape is already irregular; leaf-wise would change *which* branches get deep, not whether any do.

The rule-count comparison runs the other way from what was assumed: ClearGBM emits **47–58 leaves per tree against LightGBM's 31**[^8] — half again as many rules to read for statistically tied quality. On a rules-to-read measure, the depth-wise tree is the *less* readable of the two, and leaf-wise growth — which reaches equivalent loss with fewer effective splits — would likely improve it.

Nothing in the interpretability machinery depends on tree shape: `export_model_json`, split-count feature importance, TreeSHAP and monotonic constraints all walk arbitrary trees. The genuine interpretability lever is **oblivious trees** (uniform per-level splits), which is a different change from leaf-wise and is not what this page proposes.

Given ClearGBM's positioning as *Gradient Boosting You Can See Through*[^6], the honest framing is that today's growth strategy does not deliver that property, and leaf-wise does not take it away.

[^1]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md` § "Quality metrics" — every quality metric within seed std between cleargbm and lightgbm on the benchmark dataset.
[^2]: `libs/cleargbm_rs/src/tree/builder.rs:199,209,218` — comment "Build tree using depth-first stack", `stack: Vec<PendingNode>`, and the `while let Some(pending) = stack.pop()` loop entry.
[^3]: `libs/cleargbm_rs/src/tree/builder.rs:363-374` — right child pushed at line 364, left child pushed at line 371 (comment: "Push children to stack (right first so left is processed first)").
[^4]: `libs/cleargbm_rs/src/training/config.rs:18,50,105-110` — `pub max_depth: usize` in the params struct, `max_depth: usize` in the validated config, validation `if max_depth < 1_usize { return Err(...) }`.
[^5]: `libs/cleargbm_rs/src/tree/builder.rs:32` — `cached_histograms: Option<Vec<HistogramBuffer>>` on `PendingNode`; carried across the sibling boundary via the depth-first ordering.
[^6]: `libs/cleargbm/README.md` header line — *Gradient Boosting You Can See Through*.
[^7]: Tree dump of a model trained on the bankruptcy dataset at `max_depth=5`, read from `export_model_json` (2026-07-24): 13 distinct `feature_index` values among depth-5 internal nodes; root-to-leaf path lengths 4–6.
[^8]: [[cleargbm-leaf-normalized-benchmarking]] § "Measured tree-size divergence" (57.9 vs 31.0 leaves at `max_depth=6`) and § "Authoritative measurement (2026-07-24)" (47.15 vs 30.96 leaves, leaf ratio 1.523×).
