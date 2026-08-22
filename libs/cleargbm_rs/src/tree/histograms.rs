//! Histogram building and split-finding used by [`super::builder`].
//!
//! Concentrates the per-node histogram construction — including the
//! rayon-parallel cross-feature dispatch and the sibling-subtraction
//! trick — and the split search that turns those histograms into a
//! chosen `SplitResult`. This is the "histogram" half of the tree
//! builder; the "structural" half lives in [`super::nodes`].

use rayon::prelude::*;

use crate::error::ClearGbmError;
use crate::histogram::{reorder_grad_hess_into, subtract_histogram, HistogramRequest};
use crate::hooks::Hooks;
use crate::split::{find_best_split_from_histogram, MonotonicConstraint, SplitResult};
use crate::types::{HistogramBuffer, SplitConfig};

/// Minimum per-feature work below which rayon dispatch is skipped for the
/// serial path. Empirically, rayon's steal-work + join overhead is ~1-5µs on
/// modern x86, and one histogram build over N samples takes ~N/2 nanoseconds
/// with the unrolled tight loop. Below this threshold the dispatch overhead
/// dominates the work; above it parallelism pays off.
const RAYON_PER_FEATURE_MIN_SAMPLES: usize = 4096;

/// Reusable scratch for the node-scoped ordered gradient/hessian streams.
///
/// Every node build permutes gradients and hessians into position-space
/// before its per-feature histogram passes. Allocating that scratch per node
/// cost a heap round-trip and — worse — a `vec![0.0; n]` zero-fill that the
/// reorder pass immediately overwrote: roughly 1.5 GB of pointless memset
/// across a 200-tree fit on the benchmark workload. One pair of buffers is
/// allocated per tree at root size instead; each node overwrites exactly the
/// prefix it reads, so no zeroing between nodes is needed.
pub(super) struct OrderedScratch {
    /// Ordered-gradient scratch; length is the tree's root sample count.
    ordered_gradients: Vec<f64>,

    /// Ordered-hessian scratch; same length.
    ordered_hessians: Vec<f64>,
}

impl OrderedScratch {
    /// Creates scratch sized for a tree whose root holds `n_samples` samples.
    ///
    /// # Args
    ///
    /// * `n_samples` - Sample count at the tree's root; every node's sample
    ///   set is a subset, so this bounds every per-node prefix.
    ///
    /// # Returns
    ///
    /// Scratch ready for [`build_feature_histograms`] and
    /// [`compute_child_histograms`].
    pub(super) fn new(n_samples: usize) -> Self {
        Self {
            ordered_gradients: vec![0.0_f64; n_samples],
            ordered_hessians: vec![0.0_f64; n_samples],
        }
    }

    /// Borrows the first `n` elements of both buffers, mutably.
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds the root sample count the scratch was created
    /// with — a caller bug, since every node's sample set is a subset of the
    /// root's.
    fn prefixes(&mut self, n: usize) -> (&mut [f64], &mut [f64]) {
        (
            &mut self.ordered_gradients[..n],
            &mut self.ordered_hessians[..n],
        )
    }
}

/// Configuration for building feature histograms.
pub(super) struct BuildHistogramConfig<'a> {
    /// Sample indices.
    pub(super) sample_indices: &'a [u32],
    /// Gradients for all samples.
    pub(super) gradients: &'a [f64],
    /// Hessians for all samples.
    pub(super) hessians: &'a [f64],
    /// Bin assignments in flat column-major u8 layout, length `n_samples * n_features`.
    pub(super) bins: &'a [u8],
    /// Sample count (row count of the original feature matrix).
    pub(super) n_samples: usize,
    /// Number of features.
    pub(super) n_features: usize,
    /// Number of bins.
    pub(super) n_bins: usize,
    /// Dependency injection hooks.
    pub(super) hooks: &'a Hooks,
}

/// Builds histograms for all features.
///
/// Always builds from scratch by dispatching to the histogram hook with a
/// per-feature contiguous bin slice. The sibling-subtraction cache is the
/// caller's concern: the depth-wise builder takes the cached vector off its
/// pending node and uses it directly instead of calling here, so no clone
/// of an already-owned cache ever happens.
///
/// # Column-major fast path
///
/// The bins slice is column-major, so the per-feature bin column is
/// `bins[feat_idx * n_samples..(feat_idx + 1) * n_samples]` — a contiguous
/// `n_samples`-long byte slice. The full `sample_indices`, `gradients`, and
/// `hessians` are threaded through directly; the histogram builder does the
/// per-index gather. This eliminates the three per-(node, feature) allocations
/// the pre-refactor row-major path required.
///
/// # Parallelism
///
/// Fresh histogram builds fan out across features via rayon's
/// `into_par_iter`. Each feature's histogram is an independent walk over
/// `sample_indices` gathering into its own bin column — no shared mutable
/// state. Order is preserved by `map`+`collect`, so `histograms[feat_idx]`
/// still indexes the right feature.
pub(super) fn build_feature_histograms(
    config: &BuildHistogramConfig<'_>,
    scratch: &mut OrderedScratch,
) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    // Reorder gradients + hessians into position-space ONCE for this node,
    // then dispatch the per-feature hook (which reads them sequentially).
    // Amortization = 2 gathers per sample per node (one on gradients, one on
    // hessians) instead of 2 gathers per sample per feature. For 18 features
    // that's an 18x reduction in the input-side gather count. See wiki page
    // `lightgbm-construct-histogram-inner`.
    let n_at_node = config.sample_indices.len();
    let (ordered_gradients, ordered_hessians) = scratch.prefixes(n_at_node);
    reorder_grad_hess_into(
        config.sample_indices,
        config.gradients,
        config.hessians,
        ordered_gradients,
        ordered_hessians,
    );

    let ordered_g: &[f64] = ordered_gradients;
    let ordered_h: &[f64] = ordered_hessians;
    let build_feat = |feat_idx: usize| -> Result<HistogramBuffer, ClearGbmError> {
        let feat_col_start = feat_idx * config.n_samples;
        let feat_col_end = feat_col_start + config.n_samples;
        let feat_bins = &config.bins[feat_col_start..feat_col_end];
        (config.hooks.build_histogram)(HistogramRequest {
            sample_indices: config.sample_indices,
            ordered_gradients: ordered_g,
            ordered_hessians: ordered_h,
            bins: feat_bins,
            n_bins: config.n_bins,
        })
    };

    let results: Vec<Result<HistogramBuffer, ClearGbmError>> =
        if config.sample_indices.len() >= RAYON_PER_FEATURE_MIN_SAMPLES {
            (0_usize..config.n_features)
                .into_par_iter()
                .map(build_feat)
                .collect()
        } else {
            (0_usize..config.n_features).map(build_feat).collect()
        };

    let mut histograms = Vec::with_capacity(config.n_features);
    for r in results {
        match r {
            Ok(h) => histograms.push(h),
            Err(e) => return Err(e),
        }
    }

    Ok(histograms)
}

/// Finds best split across all features.
pub(super) fn find_best_split_across_features_internal(
    histograms: &[HistogramBuffer],
    config: &SplitConfig,
    n_regular_bins: usize,
    monotonic_constraints: Option<&[MonotonicConstraint]>,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let mut best_split: Option<SplitResult> = None;

    for (feature_idx, histogram) in histograms.iter().enumerate() {
        let constraint = monotonic_constraints
            .and_then(|constraints| constraints.get(feature_idx).copied())
            .unwrap_or(MonotonicConstraint::None);

        let maybe_split = match find_best_split_from_histogram(
            histogram,
            feature_idx,
            config,
            n_regular_bins,
            constraint,
        ) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };

        if let Some(split) = maybe_split {
            let is_better = best_split
                .as_ref()
                .is_none_or(|current| split.gain() > current.gain());

            if is_better {
                best_split = Some(split);
            }
        }
    }

    Ok(best_split)
}

/// Configuration for computing child histograms.
pub(super) struct ChildHistogramConfig<'a> {
    /// Left child sample indices.
    pub(super) left_indices: &'a [u32],
    /// Right child sample indices.
    pub(super) right_indices: &'a [u32],
    /// Gradients for all samples.
    pub(super) gradients: &'a [f64],
    /// Hessians for all samples.
    pub(super) hessians: &'a [f64],
    /// Bin assignments in flat column-major u8 layout, length `n_samples * n_features`.
    pub(super) bins: &'a [u8],
    /// Sample count (row count of the original feature matrix).
    pub(super) n_samples: usize,
    /// Number of features.
    pub(super) n_features: usize,
    /// Number of bins.
    pub(super) n_bins: usize,
    /// Parent histograms.
    pub(super) parent_histograms: &'a [HistogramBuffer],
    /// Dependency injection hooks.
    pub(super) hooks: &'a Hooks,
}

/// Computes child histograms using the sibling-subtraction trick.
///
/// Builds the smaller child's histogram from scratch, then derives the
/// larger child by subtraction from the parent. The two-child work is
/// parallelized across features via rayon.
pub(super) fn compute_child_histograms(
    config: &ChildHistogramConfig<'_>,
    scratch: &mut OrderedScratch,
) -> Result<(Vec<HistogramBuffer>, Vec<HistogramBuffer>), ClearGbmError> {
    let n_left = config.left_indices.len();
    let n_right = config.right_indices.len();
    let left_is_smaller = n_left <= n_right;

    let smaller_indices = if left_is_smaller {
        config.left_indices
    } else {
        config.right_indices
    };

    // Reorder smaller-child gradients+hessians once (mirrors
    // build_feature_histograms), then per-feature histogram build.
    // Only the smaller child is built from data — the larger child is
    // derived by subtraction from parent (see wiki page
    // `lightgbm-sibling-subtraction-trick`), so the reorder cost pays off
    // n_features times just as in the root-histogram case.
    let n_at_smaller = smaller_indices.len();
    let (ordered_gradients, ordered_hessians) = scratch.prefixes(n_at_smaller);
    reorder_grad_hess_into(
        smaller_indices,
        config.gradients,
        config.hessians,
        ordered_gradients,
        ordered_hessians,
    );

    let ordered_g: &[f64] = ordered_gradients;
    let ordered_h: &[f64] = ordered_hessians;
    let build_pair =
        |feat_idx: usize| -> Result<(HistogramBuffer, HistogramBuffer), ClearGbmError> {
            let feat_col_start = feat_idx * config.n_samples;
            let feat_col_end = feat_col_start + config.n_samples;
            let feat_bins = &config.bins[feat_col_start..feat_col_end];

            let smaller_hist = match (config.hooks.build_histogram)(HistogramRequest {
                sample_indices: smaller_indices,
                ordered_gradients: ordered_g,
                ordered_hessians: ordered_h,
                bins: feat_bins,
                n_bins: config.n_bins,
            }) {
                Ok(h) => h,
                Err(e) => return Err(e),
            };

            let parent_hist = match config.parent_histograms.get(feat_idx) {
                Some(h) => h,
                None => {
                    return Err(ClearGbmError::FeatureIndexOutOfBounds {
                        index: feat_idx,
                        n_features: config.n_features,
                    })
                }
            };

            let larger_hist = match subtract_histogram(parent_hist, &smaller_hist) {
                Ok(h) => h,
                Err(e) => return Err(e),
            };

            Ok((smaller_hist, larger_hist))
        };

    let pairs: Vec<Result<(HistogramBuffer, HistogramBuffer), ClearGbmError>> =
        if smaller_indices.len() >= RAYON_PER_FEATURE_MIN_SAMPLES {
            (0_usize..config.n_features)
                .into_par_iter()
                .map(build_pair)
                .collect()
        } else {
            (0_usize..config.n_features).map(build_pair).collect()
        };

    let mut left_histograms = Vec::with_capacity(config.n_features);
    let mut right_histograms = Vec::with_capacity(config.n_features);
    for pair in pairs {
        let (smaller_hist, larger_hist) = match pair {
            Ok(p) => p,
            Err(e) => return Err(e),
        };
        if left_is_smaller {
            left_histograms.push(smaller_hist);
            right_histograms.push(larger_hist);
        } else {
            left_histograms.push(larger_hist);
            right_histograms.push(smaller_hist);
        }
    }

    Ok((left_histograms, right_histograms))
}
