//! Histogram building and split-finding used by [`super::builder`].
//!
//! Concentrates the per-node histogram construction — including the
//! sibling-subtraction trick — and the split search that turns those
//! histograms into a chosen `SplitResult`. This is the "histogram" half
//! of the tree builder; the "structural" half lives in [`super::nodes`].
//!
//! Both growth policies carry per-node histograms as [`NodeHistograms`]:
//! the float path's per-feature buffers, or the quantized path's packed
//! integer histograms with their decode scales. Every dispatch in this
//! file branches once on that enum; the float arms are the historical
//! code, operation for operation, which is what keeps quantized-off runs
//! bit-identical.

use crate::error::ClearGbmError;
use crate::histogram::quantized::{
    select_hist_width, subtract_quantized, unpack_acc16, unpack_acc32, QuantizedNodeHistograms,
};
use crate::histogram::{reorder_grad_hess_into, subtract_histogram, NodeHistogramRequest};
use crate::hooks::Hooks;
use crate::split::{
    find_best_categorical_split_from_histogram, find_best_split_from_histogram,
    find_best_split_from_quantized_histogram, MonotonicConstraint, QuantizedScanScales,
    SplitResult,
};
use crate::types::{HistogramBuffer, SplitConfig};

use super::builder::QuantizedTreeData;
use super::categorical::CategoricalLayout;

/// One node's histograms, in whichever representation the run trains
/// under.
///
/// The float variant is the historical `Vec<HistogramBuffer>`; the
/// quantized variant carries the packed integer histograms together with
/// the round's decode scales, so the split search is self-contained.
#[derive(Debug, Clone)]
pub(super) enum NodeHistograms {
    /// Per-feature f64 histograms (the historical path).
    Float(Vec<HistogramBuffer>),
    /// Per-feature packed integer histograms plus decode scales.
    Quantized {
        /// The packed histograms at this node's width.
        histograms: QuantizedNodeHistograms,
        /// The round's decode scales.
        scales: QuantizedScanScales,
    },
}

/// Reusable scratch for the node-scoped ordered gradient/hessian streams.
///
/// Every node build permutes gradients and hessians into position-space
/// before its per-feature histogram passes. Allocating that scratch per node
/// cost a heap round-trip and — worse — a `vec![0.0; n]` zero-fill that the
/// reorder pass immediately overwrote: roughly 1.5 GB of pointless memset
/// across a 200-tree fit on the benchmark workload. One pair of buffers is
/// allocated per tree at root size instead; each node overwrites exactly the
/// prefix it reads, so no zeroing between nodes is needed.
///
/// The quantized path does not use it: its single walk gathers each row's
/// 2-byte pair directly instead of staging ordered f64 streams.
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
    /// Row-major bin matrix (sample-major rows of `n_features` bytes),
    /// length `n_samples * n_features`.
    pub(super) bins_rows: &'a [u8],
    /// Number of features.
    pub(super) n_features: usize,
    /// Number of bins.
    pub(super) n_bins: usize,
    /// The round's quantized streams, when quantized training is on.
    pub(super) quantized: Option<QuantizedTreeData<'a>>,
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
/// # Float path
///
/// Reorders gradients + hessians into position-space ONCE for this node,
/// then dispatches the per-feature hook (which reads them sequentially).
/// Amortization = 2 gathers per sample per node instead of 2 gathers per
/// sample per feature. See wiki page `lightgbm-construct-histogram-inner`.
///
/// # Quantized path
///
/// Selects this node's component width from its sample count, then
/// dispatches the quantized hook, which walks the node once gathering
/// each row's `int8` pair directly — no ordered-stream staging.
pub(super) fn build_feature_histograms(
    config: &BuildHistogramConfig<'_>,
    scratch: &mut OrderedScratch,
) -> Result<NodeHistograms, ClearGbmError> {
    match config.quantized {
        None => {
            let n_at_node = config.sample_indices.len();
            let (ordered_gradients, ordered_hessians) = scratch.prefixes(n_at_node);
            reorder_grad_hess_into(
                config.sample_indices,
                config.gradients,
                config.hessians,
                ordered_gradients,
                ordered_hessians,
            );

            let histograms = match (config.hooks.build_node_histograms)(NodeHistogramRequest {
                sample_indices: config.sample_indices,
                ordered_gradients,
                ordered_hessians,
                bins_rows: config.bins_rows,
                n_features: config.n_features,
                n_bins: config.n_bins,
            }) {
                Ok(h) => h,
                Err(e) => return Err(e),
            };
            Ok(NodeHistograms::Float(histograms))
        }
        Some(quantized) => {
            let width = select_hist_width(config.sample_indices.len(), quantized.n_quant_bins);
            let histograms = match (config.hooks.build_node_histograms_quantized)(
                crate::histogram::quantized::QuantizedNodeHistogramRequest {
                    sample_indices: config.sample_indices,
                    packed_int8: quantized.packed_int8,
                    bins_rows: config.bins_rows,
                    n_features: config.n_features,
                    n_bins: config.n_bins,
                    width,
                },
            ) {
                Ok(h) => h,
                Err(e) => return Err(e),
            };
            Ok(NodeHistograms::Quantized {
                histograms,
                scales: QuantizedScanScales {
                    grad_scale: quantized.grad_scale,
                    hess_scale: quantized.hess_scale,
                },
            })
        }
    }
}

/// Finds the best split across the features a node may consider.
///
/// `allowed_features`, when present, is the per-node `max_features` mask —
/// masked-out features are skipped entirely. `None` scans every feature in
/// the same order as before the mask existed. Features marked categorical
/// in `categorical` run the many-vs-many subset search instead of the
/// threshold scan; the best split overall wins on gain regardless of kind.
///
/// Under quantized histograms every feature takes the integer threshold
/// scan: config validation refuses `quantized_gradient_bins` together
/// with `categorical_features`, so no categorical layout can reach the
/// quantized arm.
pub(super) fn find_best_split_across_features_internal(
    histograms: &NodeHistograms,
    config: &SplitConfig,
    n_regular_bins: usize,
    monotonic_constraints: Option<&[MonotonicConstraint]>,
    allowed_features: Option<&[bool]>,
    categorical: Option<&CategoricalLayout>,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let mut best_split: Option<SplitResult> = None;

    match histograms {
        NodeHistograms::Float(feature_histograms) => {
            for (feature_idx, histogram) in feature_histograms.iter().enumerate() {
                if let Some(mask) = allowed_features {
                    if !mask[feature_idx] {
                        continue;
                    }
                }
                let n_categories = categorical.and_then(|layout| layout.n_categories(feature_idx));
                let constraint = monotonic_constraints
                    .and_then(|constraints| constraints.get(feature_idx).copied())
                    .unwrap_or(MonotonicConstraint::None);

                let maybe_split = match n_categories {
                    Some(n_cats) => match find_best_categorical_split_from_histogram(
                        histogram,
                        feature_idx,
                        config,
                        n_cats,
                        n_regular_bins,
                    ) {
                        Ok(s) => s,
                        Err(e) => return Err(e),
                    },
                    None => match find_best_split_from_histogram(
                        histogram,
                        feature_idx,
                        config,
                        n_regular_bins,
                        constraint,
                    ) {
                        Ok(s) => s,
                        Err(e) => return Err(e),
                    },
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
        }
        NodeHistograms::Quantized {
            histograms: quantized,
            scales,
        } => {
            // Dispatch once on the node's width; the generic scan then
            // reads packed entries in place, at that width, with no
            // per-feature materialization.
            let request = QuantizedScanRequest {
                config,
                n_regular_bins,
                monotonic_constraints,
                allowed_features,
                scales: *scales,
            };
            best_split = match quantized {
                QuantizedNodeHistograms::B16(features) => {
                    match scan_quantized_features(features, unpack_acc16, &request) {
                        Ok(s) => s,
                        Err(e) => return Err(e),
                    }
                }
                QuantizedNodeHistograms::B32(features) => {
                    match scan_quantized_features(features, unpack_acc32, &request) {
                        Ok(s) => s,
                        Err(e) => return Err(e),
                    }
                }
            };
        }
    }

    Ok(best_split)
}

/// The per-node inputs the quantized cross-feature scan needs.
struct QuantizedScanRequest<'a> {
    /// Split configuration.
    config: &'a SplitConfig,
    /// Number of regular bins (excluding the NaN bin).
    n_regular_bins: usize,
    /// Per-feature monotonic constraints, when any.
    monotonic_constraints: Option<&'a [MonotonicConstraint]>,
    /// The per-node feature mask, when one is active.
    allowed_features: Option<&'a [bool]>,
    /// The round's decode scales.
    scales: QuantizedScanScales,
}

/// Scans every allowed feature's packed histogram for the best split.
///
/// Generic over the entry width so one body serves both packed forms;
/// masked-out features are skipped exactly as in the float loop.
fn scan_quantized_features<T: Copy>(
    features: &[Vec<T>],
    unpack: fn(T) -> (i64, i64, usize),
    request: &QuantizedScanRequest<'_>,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let mut best_split: Option<SplitResult> = None;
    for (feature_idx, bins) in features.iter().enumerate() {
        if let Some(mask) = request.allowed_features {
            if !mask[feature_idx] {
                continue;
            }
        }
        let constraint = request
            .monotonic_constraints
            .and_then(|constraints| constraints.get(feature_idx).copied())
            .unwrap_or(MonotonicConstraint::None);

        let maybe_split = match find_best_split_from_quantized_histogram(
            bins,
            unpack,
            feature_idx,
            request.config,
            request.n_regular_bins,
            constraint,
            request.scales,
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
    /// Row-major bin matrix (sample-major rows of `n_features` bytes),
    /// length `n_samples * n_features`.
    pub(super) bins_rows: &'a [u8],
    /// Number of features.
    pub(super) n_features: usize,
    /// Number of bins.
    pub(super) n_bins: usize,
    /// The round's quantized streams, when quantized training is on.
    pub(super) quantized: Option<QuantizedTreeData<'a>>,
    /// Parent histograms.
    pub(super) parent_histograms: &'a NodeHistograms,
    /// Dependency injection hooks.
    pub(super) hooks: &'a Hooks,
}

/// Computes child histograms using the sibling-subtraction trick.
///
/// Builds the smaller child's histogram from scratch, then derives the
/// larger child by subtraction from the parent. On the quantized path
/// the smaller child builds at its own width, the subtraction runs in
/// packed integers, and the sibling lands at the width its own count
/// selects — LightGBM's mixed-width `Subtract` dispatch.
pub(super) fn compute_child_histograms(
    config: &ChildHistogramConfig<'_>,
    scratch: &mut OrderedScratch,
) -> Result<(NodeHistograms, NodeHistograms), ClearGbmError> {
    let n_left = config.left_indices.len();
    let n_right = config.right_indices.len();
    let left_is_smaller = n_left <= n_right;

    let smaller_indices = if left_is_smaller {
        config.left_indices
    } else {
        config.right_indices
    };

    match config.parent_histograms {
        NodeHistograms::Float(parent_histograms) => {
            // Reorder smaller-child gradients+hessians once (mirrors
            // build_feature_histograms), then per-feature histogram build.
            // Only the smaller child is built from data — the larger child is
            // derived by subtraction from parent (see wiki page
            // `lightgbm-sibling-subtraction-trick`), so the reorder cost pays
            // off n_features times just as in the root-histogram case.
            let n_at_smaller = smaller_indices.len();
            let (ordered_gradients, ordered_hessians) = scratch.prefixes(n_at_smaller);
            reorder_grad_hess_into(
                smaller_indices,
                config.gradients,
                config.hessians,
                ordered_gradients,
                ordered_hessians,
            );

            let smaller_histograms =
                match (config.hooks.build_node_histograms)(NodeHistogramRequest {
                    sample_indices: smaller_indices,
                    ordered_gradients,
                    ordered_hessians,
                    bins_rows: config.bins_rows,
                    n_features: config.n_features,
                    n_bins: config.n_bins,
                }) {
                    Ok(h) => h,
                    Err(e) => return Err(e),
                };

            let build_pair = |(feat_idx, smaller_hist): (usize, HistogramBuffer)| -> Result<
                (HistogramBuffer, HistogramBuffer),
                ClearGbmError,
            > {
                let parent_hist = match parent_histograms.get(feat_idx) {
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
                smaller_histograms
                    .into_iter()
                    .enumerate()
                    .map(build_pair)
                    .collect();

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

            Ok((
                NodeHistograms::Float(left_histograms),
                NodeHistograms::Float(right_histograms),
            ))
        }
        NodeHistograms::Quantized {
            histograms: parent_quantized,
            scales,
        } => {
            // The quantized data travels alongside the parent histograms;
            // one cannot exist without the other in a quantized run.
            let Some(quantized) = config.quantized else {
                return Err(ClearGbmError::ShapeMismatch {
                    expected: "quantized tree data alongside quantized parent histograms"
                        .to_string(),
                    got: "no quantized data on the child-histogram request".to_string(),
                });
            };

            let smaller_width = select_hist_width(smaller_indices.len(), quantized.n_quant_bins);
            let smaller_histograms = match (config.hooks.build_node_histograms_quantized)(
                crate::histogram::quantized::QuantizedNodeHistogramRequest {
                    sample_indices: smaller_indices,
                    packed_int8: quantized.packed_int8,
                    bins_rows: config.bins_rows,
                    n_features: config.n_features,
                    n_bins: config.n_bins,
                    width: smaller_width,
                },
            ) {
                Ok(h) => h,
                Err(e) => return Err(e),
            };

            let larger_count = if left_is_smaller { n_right } else { n_left };
            let larger_width = select_hist_width(larger_count, quantized.n_quant_bins);
            let larger_histograms =
                match subtract_quantized(parent_quantized, &smaller_histograms, larger_width) {
                    Ok(h) => h,
                    Err(e) => return Err(e),
                };

            let smaller = NodeHistograms::Quantized {
                histograms: smaller_histograms,
                scales: *scales,
            };
            let larger = NodeHistograms::Quantized {
                histograms: larger_histograms,
                scales: *scales,
            };
            if left_is_smaller {
                Ok((smaller, larger))
            } else {
                Ok((larger, smaller))
            }
        }
    }
}
