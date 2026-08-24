//! Tree-builder tests for the quantized histogram path.
//!
//! Exercise both growth policies end to end on quantized data, the
//! quantized validation arms, the quantized hook's error propagation,
//! and the child-histogram invariant that quantized parent histograms
//! require quantized tree data.

use crate::error::ClearGbmError;
use crate::histogram::quantized::{QuantizedNodeHistogramRequest, QuantizedNodeHistograms};
use crate::hooks::Hooks;
use crate::split::QuantizedScanScales;
use crate::tree::histograms::{
    compute_child_histograms, ChildHistogramConfig, NodeHistograms, OrderedScratch,
};
use crate::tree::{
    build_tree_leaf_wise_with_leaf_assignment, build_tree_with_leaf_assignment, BuildTreeInput,
    QuantizedTreeData, TreeBuildConfig,
};
use crate::types::SplitConfig;

/// Eight rows, one feature, bins 0/1 split 4-4, gradients separable.
struct QuantFixture {
    sample_indices: Vec<u32>,
    gradients: Vec<f64>,
    hessians: Vec<f64>,
    bins_rows: Vec<u8>,
    packed_int8: Vec<i8>,
    bin_thresholds: Vec<Vec<f64>>,
}

fn fixture() -> QuantFixture {
    // Quantized pairs hand-built: gradient +2 for bin-0 rows, -2 for
    // bin-1 rows, hessian 4 everywhere; scales 0.5 / 0.25 decode them
    // to gradient +/-1 and hessian 1 - matching the float arrays.
    let n = 8_usize;
    let mut gradients = Vec::with_capacity(n);
    let mut bins_rows = Vec::with_capacity(n);
    let mut packed_int8 = Vec::with_capacity(2_usize * n);
    for i in 0_usize..n {
        let in_low = i < 4_usize;
        gradients.push(if in_low { 1.0_f64 } else { -1.0_f64 });
        bins_rows.push(if in_low { 0_u8 } else { 1_u8 });
        packed_int8.push(4_i8);
        packed_int8.push(if in_low { 2_i8 } else { -2_i8 });
    }
    QuantFixture {
        sample_indices: (0_u32..8_u32).collect(),
        gradients,
        hessians: vec![1.0_f64; n],
        bins_rows,
        packed_int8,
        bin_thresholds: vec![vec![0.5_f64, 1.5_f64]],
    }
}

fn tree_config(max_depth: usize, max_leaves: usize) -> Result<TreeBuildConfig, ClearGbmError> {
    let split_config = propagate!(SplitConfig::new(
        2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64
    ));
    TreeBuildConfig::new(max_depth, max_leaves, 0.0_f64, 0.0_f64, split_config)
}

fn build_input<'a>(fx: &'a QuantFixture, config: &'a TreeBuildConfig) -> BuildTreeInput<'a> {
    BuildTreeInput {
        sample_indices: &fx.sample_indices,
        gradients: &fx.gradients,
        hessians: &fx.hessians,
        bins_rows: &fx.bins_rows,
        n_samples: 8_usize,
        n_features: 1_usize,
        n_regular_bins: 2_usize,
        bin_thresholds: &fx.bin_thresholds,
        config,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
        quantized: Some(QuantizedTreeData {
            packed_int8: &fx.packed_int8,
            grad_scale: 0.5_f64,
            hess_scale: 0.25_f64,
            n_quant_bins: 4_usize,
        }),
    }
}

#[test]
fn test_depth_wise_quantized_build_splits_and_uses_float_leaf_values() -> Result<(), ClearGbmError>
{
    // The split must separate the bins, and each leaf's value must come
    // from the ORIGINAL float gradients: -sum(g)/sum(h) = -1 and +1.
    let config = propagate!(tree_config(3_usize, 0_usize));
    let fx = fixture();
    let input = build_input(&fx, &config);
    let (tree, leaf_values) =
        propagate!(build_tree_with_leaf_assignment(&input, &Hooks::default()));
    assert_eq!(tree.n_leaves(), 2_usize);
    assert!((leaf_values[0_usize] + 1.0_f64).abs() < 1e-12_f64);
    assert!((leaf_values[7_usize] - 1.0_f64).abs() < 1e-12_f64);
    Ok(())
}

#[test]
fn test_leaf_wise_quantized_build_matches_depth_wise_here() -> Result<(), ClearGbmError> {
    // One available split: both policies must produce the same leaves.
    let depth_config = propagate!(tree_config(3_usize, 0_usize));
    let leaf_config = propagate!(tree_config(3_usize, 4_usize));
    let fx = fixture();
    let depth_input = build_input(&fx, &depth_config);
    let leaf_input = build_input(&fx, &leaf_config);
    let (depth_tree, depth_values) = propagate!(build_tree_with_leaf_assignment(
        &depth_input,
        &Hooks::default()
    ));
    let (leaf_tree, leaf_values) = propagate!(build_tree_leaf_wise_with_leaf_assignment(
        &leaf_input,
        &Hooks::default()
    ));
    assert_eq!(depth_tree.n_leaves(), leaf_tree.n_leaves());
    assert_eq!(depth_values, leaf_values);
    Ok(())
}

#[test]
fn test_oversized_bins_times_rows_is_refused() -> Result<(), ClearGbmError> {
    // n_samples * n_quant_bins past u32::MAX would overflow the packed
    // 32-bit half; the validator must name the knob. n_samples is a
    // declared shape, so no giant allocation is needed to reach it.
    let config = propagate!(tree_config(3_usize, 0_usize));
    let fx = fixture();
    let mut input = build_input(&fx, &config);
    input.n_samples = 2_000_000_000_usize;
    input.quantized = Some(QuantizedTreeData {
        packed_int8: &fx.packed_int8,
        grad_scale: 0.5_f64,
        hess_scale: 0.25_f64,
        n_quant_bins: 4_usize,
    });
    match build_tree_with_leaf_assignment(&input, &Hooks::default()) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an overflowing width bound must be refused".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "quantized_gradient_bins");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_short_packed_stream_is_refused() -> Result<(), ClearGbmError> {
    let config = propagate!(tree_config(3_usize, 0_usize));
    let fx = fixture();
    let mut input = build_input(&fx, &config);
    let short_stream = vec![0_i8; 3_usize];
    input.quantized = Some(QuantizedTreeData {
        packed_int8: &short_stream,
        grad_scale: 0.5_f64,
        hess_scale: 0.25_f64,
        n_quant_bins: 4_usize,
    });
    match build_tree_with_leaf_assignment(&input, &Hooks::default()) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a short packed stream must be refused".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

/// A quantized histogram hook that always fails.
fn quant_error_hook(
    _: QuantizedNodeHistogramRequest<'_>,
) -> Result<QuantizedNodeHistograms, ClearGbmError> {
    Err(ClearGbmError::EmptyInput {
        context: "injected quantized".to_string(),
    })
}

#[test]
fn test_quantized_hook_error_propagates_from_the_root_build() -> Result<(), ClearGbmError> {
    let config = propagate!(tree_config(3_usize, 0_usize));
    let fx = fixture();
    let input = build_input(&fx, &config);
    let hooks = Hooks::with_quantized_histogram_builder(quant_error_hook);
    match build_tree_with_leaf_assignment(&input, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "the injected quantized error must propagate".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert!(context.contains("injected quantized"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_quantized_hook_error_propagates_leaf_wise_too() -> Result<(), ClearGbmError> {
    let config = propagate!(tree_config(3_usize, 4_usize));
    let fx = fixture();
    let input = build_input(&fx, &config);
    let hooks = Hooks::with_quantized_histogram_builder(quant_error_hook);
    match build_tree_leaf_wise_with_leaf_assignment(&input, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "the injected quantized error must propagate".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert!(context.contains("injected quantized"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_child_histograms_refuse_quantized_parent_without_data() -> Result<(), ClearGbmError> {
    // A quantized parent histogram with no quantized data on the request
    // violates the pairing invariant and must be refused by name.
    let fx = fixture();
    let left: Vec<u32> = vec![0_u32, 1_u32];
    let right: Vec<u32> = vec![2_u32, 3_u32];
    let parent = NodeHistograms::Quantized {
        histograms: QuantizedNodeHistograms::B16(vec![vec![]]),
        scales: QuantizedScanScales {
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
        },
    };
    let hooks = Hooks::default();
    let config = ChildHistogramConfig {
        left_indices: &left,
        right_indices: &right,
        gradients: &fx.gradients,
        hessians: &fx.hessians,
        bins_rows: &fx.bins_rows,
        n_features: 1_usize,
        n_bins: 3_usize,
        quantized: None,
        parent_histograms: &parent,
        hooks: &hooks,
    };
    match compute_child_histograms(&config, &mut OrderedScratch::new(4_usize)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "quantized parents without quantized data must be refused".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_smaller_child_hook_error_propagates_in_subtraction() -> Result<(), ClearGbmError> {
    // The quantized hook fires again for the smaller child during
    // sibling subtraction; its error must surface from there too.
    let fx = fixture();
    let left: Vec<u32> = vec![0_u32, 1_u32];
    let right: Vec<u32> = vec![2_u32, 3_u32];
    let parent = NodeHistograms::Quantized {
        histograms: QuantizedNodeHistograms::B16(vec![vec![]]),
        scales: QuantizedScanScales {
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
        },
    };
    let hooks = Hooks::with_quantized_histogram_builder(quant_error_hook);
    let config = ChildHistogramConfig {
        left_indices: &left,
        right_indices: &right,
        gradients: &fx.gradients,
        hessians: &fx.hessians,
        bins_rows: &fx.bins_rows,
        n_features: 1_usize,
        n_bins: 3_usize,
        quantized: Some(QuantizedTreeData {
            packed_int8: &fx.packed_int8,
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
            n_quant_bins: 4_usize,
        }),
        parent_histograms: &parent,
        hooks: &hooks,
    };
    match compute_child_histograms(&config, &mut OrderedScratch::new(4_usize)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "the injected quantized error must propagate".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert!(context.contains("injected quantized"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_quantized_split_scan_error_propagates_across_features() -> Result<(), ClearGbmError> {
    // An n_regular_bins beyond the histogram makes the integer scan
    // error; the cross-feature dispatcher must surface it intact.
    use crate::tree::histograms::find_best_split_across_features_internal;
    let histograms = NodeHistograms::Quantized {
        histograms: QuantizedNodeHistograms::B16(vec![vec![
            crate::histogram::quantized::QuantAcc16::ZERO,
        ]]),
        scales: QuantizedScanScales {
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
        },
    };
    let config = propagate!(SplitConfig::new(
        2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64
    ));
    match find_best_split_across_features_internal(&histograms, &config, 5_usize, None, None, None)
    {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an oversized n_regular_bins must be refused".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_regular_bins");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_quantized_feature_mask_skips_features_in_the_scan() -> Result<(), ClearGbmError> {
    // A mask allowing only feature 1 must make the quantized scan skip
    // feature 0 entirely; with feature 1 unsplittable the node yields no
    // split at all.
    use crate::tree::histograms::find_best_split_across_features_internal;
    let splittable = vec![
        crate::histogram::quantized::QuantAcc16 {
            packed: (10_i32 << 16_u32) + 5_i32,
            count: 5_u32,
        },
        crate::histogram::quantized::QuantAcc16 {
            packed: (-10_i32 << 16_u32) + 5_i32,
            count: 5_u32,
        },
    ];
    let flat = vec![
        crate::histogram::quantized::QuantAcc16 {
            packed: 10_i32,
            count: 10_u32,
        },
        crate::histogram::quantized::QuantAcc16::ZERO,
    ];
    let histograms = NodeHistograms::Quantized {
        histograms: QuantizedNodeHistograms::B16(vec![splittable, flat]),
        scales: QuantizedScanScales {
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
        },
    };
    let config = propagate!(SplitConfig::new(
        2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64
    ));
    let mask = vec![false, true];
    let masked = propagate!(find_best_split_across_features_internal(
        &histograms,
        &config,
        2_usize,
        None,
        Some(&mask),
        None,
    ));
    assert!(masked.is_none(), "feature 0 must be skipped by the mask");
    let unmasked = propagate!(find_best_split_across_features_internal(
        &histograms,
        &config,
        2_usize,
        None,
        None,
        None,
    ));
    assert!(unmasked.is_some(), "feature 0 splits when unmasked");
    Ok(())
}

#[test]
fn test_subtract_error_propagates_from_child_histograms() -> Result<(), ClearGbmError> {
    // A hook returning 32-bit child histograms under a 16-bit parent
    // makes the packed subtraction refuse; the child-histogram dispatch
    // must surface that error.
    fn wide_child_hook(
        request: QuantizedNodeHistogramRequest<'_>,
    ) -> Result<QuantizedNodeHistograms, ClearGbmError> {
        let bins = vec![
            vec![crate::histogram::quantized::QuantAcc32::ZERO; request.n_bins];
            request.n_features
        ];
        Ok(QuantizedNodeHistograms::B32(bins))
    }
    let fx = fixture();
    let left: Vec<u32> = vec![0_u32, 1_u32];
    let right: Vec<u32> = vec![2_u32, 3_u32];
    let parent = NodeHistograms::Quantized {
        histograms: QuantizedNodeHistograms::B16(vec![vec![
            crate::histogram::quantized::QuantAcc16::ZERO;
            3_usize
        ]]),
        scales: QuantizedScanScales {
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
        },
    };
    let hooks = Hooks::with_quantized_histogram_builder(wide_child_hook);
    let config = ChildHistogramConfig {
        left_indices: &left,
        right_indices: &right,
        gradients: &fx.gradients,
        hessians: &fx.hessians,
        bins_rows: &fx.bins_rows,
        n_features: 1_usize,
        n_bins: 3_usize,
        quantized: Some(QuantizedTreeData {
            packed_int8: &fx.packed_int8,
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
            n_quant_bins: 4_usize,
        }),
        parent_histograms: &parent,
        hooks: &hooks,
    };
    match compute_child_histograms(&config, &mut OrderedScratch::new(4_usize)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a wider child under a narrower parent must be refused".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_b32_histograms_scan_through_the_width_dispatch() -> Result<(), ClearGbmError> {
    // The 32-bit arm of the scan dispatch, driven directly: a splittable
    // two-bin B32 histogram must yield the same split the 16-bit form
    // would.
    use crate::tree::histograms::find_best_split_across_features_internal;
    let bins = vec![
        crate::histogram::quantized::QuantAcc32 {
            packed: (10_i64 << 32_u32) + 5_i64,
            count: 5_u32,
        },
        crate::histogram::quantized::QuantAcc32 {
            packed: (-10_i64 << 32_u32) + 5_i64,
            count: 5_u32,
        },
    ];
    let histograms = NodeHistograms::Quantized {
        histograms: QuantizedNodeHistograms::B32(vec![bins]),
        scales: QuantizedScanScales {
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
        },
    };
    let config = propagate!(SplitConfig::new(
        2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64
    ));
    let result = propagate!(find_best_split_across_features_internal(
        &histograms,
        &config,
        2_usize,
        None,
        None,
        None,
    ));
    let Some(split) = result else {
        return Err(ClearGbmError::TreeConstructionFailed {
            reason: "the 32-bit histogram must split".to_string(),
        });
    };
    assert_eq!(split.left_count(), 5_usize);
    assert_eq!(split.right_count(), 5_usize);
    Ok(())
}

#[test]
fn test_b32_scan_error_propagates_through_the_dispatch() -> Result<(), ClearGbmError> {
    // The 32-bit arm's error propagation: an oversized n_regular_bins
    // surfaces through the width dispatch intact.
    use crate::tree::histograms::find_best_split_across_features_internal;
    let histograms = NodeHistograms::Quantized {
        histograms: QuantizedNodeHistograms::B32(vec![vec![
            crate::histogram::quantized::QuantAcc32::ZERO,
        ]]),
        scales: QuantizedScanScales {
            grad_scale: 1.0_f64,
            hess_scale: 1.0_f64,
        },
    };
    let config = propagate!(SplitConfig::new(
        2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64
    ));
    match find_best_split_across_features_internal(&histograms, &config, 5_usize, None, None, None)
    {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an oversized n_regular_bins must be refused".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_regular_bins");
            Ok(())
        }
        Err(e) => Err(e),
    }
}
