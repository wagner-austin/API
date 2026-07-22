---
title: ClearGBM perf — leaf-wise tree growth
tags: [ml, cleargbm, rust, performance, tree]
related: [[cleargbm-histogram-split-path]]
sources:
  - libs/cleargbm_rs/src/tree/builder.rs
  - libs/cleargbm_rs/src/tree/mod.rs
  - libs/cleargbm_rs/src/training/config.rs
fact_checked: 2026-07-21
confidence: medium
---

# ClearGBM perf — leaf-wise tree growth

Replace ClearGBM's depth-first tree growth with LightGBM's leaf-wise (best-first) strategy: at every step, expand the leaf with the highest split gain across the whole tree, not the deepest-yet-unfinished branch. This is a **capacity gain, not a speed gain** — leaf-wise reaches equivalent loss with fewer effective splits, so at matched `n_estimators` the model has more useful capacity per tree.

**Confidence: medium.** Impact on this benchmark is ambiguous — quality is already a statistical tie with LightGBM[^1], so the ceiling is small. **Do this LAST** in the perf roadmap, after column-major + uint8 + SIMD have landed and the speed gap is closed. Leaf-wise also trades interpretability (unbalanced trees are harder to read as rules) for capacity, which cuts against ClearGBM's core value prop.

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

Depth-first produces balanced trees where every leaf sits at roughly the same depth — readable as short rule paths. Leaf-wise produces unbalanced trees where one branch may be 10 splits deep and its sibling a single leaf. Rule extraction becomes harder to reason about, and the wiki page on any consumer-facing "explainable model" story would need a caveat.

Given ClearGBM's marketing is *Gradient Boosting You Can See Through*[^6], this trade-off is worth flagging to a human before shipping. Do the perf work in the other three pages first; revisit leaf-wise only if the speed gap doesn't close enough.

[^1]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md` § "Quality metrics" — every quality metric within seed std between cleargbm and lightgbm on the benchmark dataset.
[^2]: `libs/cleargbm_rs/src/tree/builder.rs:199,209,218` — comment "Build tree using depth-first stack", `stack: Vec<PendingNode>`, and the `while let Some(pending) = stack.pop()` loop entry.
[^3]: `libs/cleargbm_rs/src/tree/builder.rs:363-374` — right child pushed at line 364, left child pushed at line 371 (comment: "Push children to stack (right first so left is processed first)").
[^4]: `libs/cleargbm_rs/src/training/config.rs:18,50,105-110` — `pub max_depth: usize` in the params struct, `max_depth: usize` in the validated config, validation `if max_depth < 1_usize { return Err(...) }`.
[^5]: `libs/cleargbm_rs/src/tree/builder.rs:32` — `cached_histograms: Option<Vec<HistogramBuffer>>` on `PendingNode`; carried across the sibling boundary via the depth-first ordering.
[^6]: `libs/cleargbm/README.md` header line — *Gradient Boosting You Can See Through*.
