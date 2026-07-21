//! Feature importance extraction from a trained gradient boosting model.
//!
//! Split-count importance: for each feature, count how many times it appears as
//! the split feature at any internal node across all trees, then normalize so
//! importances sum to 1.0. Matches sklearn's default "weight"-style importance;
//! deliberately does not depend on gain-per-split, which is not stored on tree
//! nodes today.
//!
//! A future enhancement would be gain-weighted importance (LightGBM's default
//! "gain" importance), which requires threading split gain onto internal nodes
//! at training time — out of scope for this file.

use super::model::GradientBoostingModel;

/// Returns per-feature split-count importance, normalized to sum to 1.0.
///
/// # Args
///
/// * `model` - The trained `GradientBoostingModel`.
///
/// # Returns
///
/// A `Vec<(String, f64)>` with one entry per feature in `model.feature_names()`,
/// in feature-index order. Importances sum to 1.0 within FP precision when at
/// least one split exists across the ensemble. If the model contains no
/// internal (split) nodes — every tree is a single leaf — all importances are
/// 0.0.
///
/// # Examples
///
/// A model with three features that split 5 times on feature 0, 3 times on
/// feature 2, and never on feature 1 will return
/// `[("f0", 5/8), ("f1", 0.0), ("f2", 3/8)]`.
#[must_use]
pub fn feature_importances(model: &GradientBoostingModel) -> Vec<(String, f64)> {
    let feature_names = model.feature_names();
    let n_features = feature_names.len();
    // Counting in f64 from the start avoids `usize as f64` casts (banned by
    // the workspace `as_conversions` lint) and lets the normalization loop stay
    // in the same numeric domain end-to-end.
    let mut counts: Vec<f64> = vec![0.0_f64; n_features];

    for tree in model.trees() {
        for node in tree.nodes() {
            if node.is_leaf() {
                continue;
            }
            let feat_idx = match node.feature_index() {
                Some(idx) => idx,
                None => continue,
            };
            if feat_idx < n_features {
                counts[feat_idx] += 1.0_f64;
            }
        }
    }

    let total: f64 = counts.iter().sum();

    let mut result: Vec<(String, f64)> = Vec::with_capacity(n_features);
    if total <= 0.0_f64 {
        for name in feature_names {
            result.push((name.clone(), 0.0_f64));
        }
        return result;
    }

    for (i, name) in feature_names.iter().enumerate() {
        let normalized = counts[i] / total;
        result.push((name.clone(), normalized));
    }
    result
}
