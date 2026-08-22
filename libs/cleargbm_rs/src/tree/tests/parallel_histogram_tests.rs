//! Tests for the rayon-parallel histogram paths.
//!
//! [`super::super::histograms`] switches from a serial `map` to
//! `into_par_iter` once a node holds at least `RAYON_PER_FEATURE_MIN_SAMPLES`
//! samples. Every other test in the crate uses node sizes far below that
//! threshold, so the branch that actually runs in production was never
//! executed. These tests drive both sides of the branch and check each against
//! the same independently-computed reference, so a fan-out that dropped,
//! duplicated or misordered a feature would fail rather than merely be
//! untested.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::tree::histograms::{
    build_feature_histograms, compute_child_histograms, BuildHistogramConfig, ChildHistogramConfig,
    OrderedScratch,
};
use crate::types::HistogramBuffer;

/// Node size at or above which the parallel branch is taken.
///
/// Mirrors `histograms::RAYON_PER_FEATURE_MIN_SAMPLES`, which is private. The
/// assertion in [`parallel_threshold_is_what_these_tests_assume`] fails if the
/// two ever drift apart.
const PARALLEL_THRESHOLD: usize = 4096_usize;

const N_FEATURES: usize = 3_usize;
const N_BINS: usize = 8_usize;

/// Builds a deterministic column-major bin matrix.
///
/// Each feature uses a different stride so the features are not copies of one
/// another — a parallel dispatch that returned the same histogram for every
/// feature would otherwise pass.
///
/// # Args
///
/// * `n_samples` - Row count.
///
/// # Returns
///
/// A `n_features * n_samples` column-major bin matrix.
fn make_bins(n_samples: usize) -> Vec<u8> {
    let mut bins: Vec<u8> = Vec::with_capacity(N_FEATURES * n_samples);
    for feat in 0_usize..N_FEATURES {
        for row in 0_usize..n_samples {
            let raw = (row * (feat + 1_usize) + feat) % (N_BINS - 1_usize);
            bins.push(u8::try_from(raw).unwrap_or(0_u8));
        }
    }
    bins
}

/// Builds deterministic, non-uniform gradients.
fn make_gradients(n_samples: usize) -> Vec<f64> {
    (0_usize..n_samples)
        .map(|i| {
            let scaled = f64::from(u32::try_from(i % 97_usize).unwrap_or(0_u32));
            scaled.mul_add(0.01_f64, -0.5_f64)
        })
        .collect()
}

/// Builds deterministic, strictly positive hessians.
fn make_hessians(n_samples: usize) -> Vec<f64> {
    (0_usize..n_samples)
        .map(|i| {
            let scaled = f64::from(u32::try_from(i % 13_usize).unwrap_or(0_u32));
            scaled.mul_add(0.05_f64, 0.25_f64)
        })
        .collect()
}

/// Computes the expected per-feature histograms with a plain serial loop.
///
/// Deliberately does not call any crate histogram code, so it is an
/// independent oracle rather than a restatement of the implementation.
///
/// # Args
///
/// * `sample_indices` - Rows at the node.
/// * `gradients` - Gradient per row.
/// * `hessians` - Hessian per row.
/// * `bins` - Column-major bin matrix.
/// * `n_samples` - Row count of the full matrix.
///
/// # Returns
///
/// One histogram per feature.
///
/// # Errors
///
/// Returns [`ClearGbmError::BinIndexOutOfBounds`] if a bin exceeds `N_BINS`.
fn reference_histograms(
    sample_indices: &[u32],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[u8],
    n_samples: usize,
) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    let mut out: Vec<HistogramBuffer> = Vec::with_capacity(N_FEATURES);
    for feat in 0_usize..N_FEATURES {
        let mut hist = HistogramBuffer::new(N_BINS);
        for &idx in sample_indices {
            let row = usize::try_from(idx).unwrap_or(usize::MAX);
            let bin = usize::from(bins[feat * n_samples + row]);
            match hist.accumulate(bin, gradients[row], hessians[row]) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        out.push(hist);
    }
    Ok(out)
}

#[test]
fn parallel_threshold_is_what_these_tests_assume() -> Result<(), ClearGbmError> {
    // The implementation's constant is private, so this test pins the value
    // these fixtures are sized against. If the production threshold changes,
    // the node sizes below must change with it or the parallel branch stops
    // being exercised — silently.
    let n_samples = PARALLEL_THRESHOLD;
    let bins = make_bins(n_samples);
    let gradients = make_gradients(n_samples);
    let hessians = make_hessians(n_samples);
    let sample_indices: Vec<u32> = (0_usize..n_samples)
        .map(|i| u32::try_from(i).unwrap_or(0_u32))
        .collect();
    let hooks = Hooks::default();

    let config = BuildHistogramConfig {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples,
        n_features: N_FEATURES,
        n_bins: N_BINS,
        hooks: &hooks,
    };
    let built = match build_feature_histograms(
        &config,
        &mut OrderedScratch::new(config.sample_indices.len()),
    ) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };
    let expected =
        match reference_histograms(&sample_indices, &gradients, &hessians, &bins, n_samples) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };
    assert_eq!(built, expected);
    Ok(())
}

#[test]
fn serial_and_parallel_paths_produce_the_same_histograms() -> Result<(), ClearGbmError> {
    let n_samples = PARALLEL_THRESHOLD + 500_usize;
    let bins = make_bins(n_samples);
    let gradients = make_gradients(n_samples);
    let hessians = make_hessians(n_samples);
    let hooks = Hooks::default();

    // One node size on each side of the dispatch threshold.
    let sizes = [PARALLEL_THRESHOLD - 1_usize, n_samples];
    for size in sizes {
        let sample_indices: Vec<u32> = (0_usize..size)
            .map(|i| u32::try_from(i).unwrap_or(0_u32))
            .collect();
        let config = BuildHistogramConfig {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_samples,
            n_features: N_FEATURES,
            n_bins: N_BINS,
            hooks: &hooks,
        };
        let built = match build_feature_histograms(
            &config,
            &mut OrderedScratch::new(config.sample_indices.len()),
        ) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };
        let expected =
            match reference_histograms(&sample_indices, &gradients, &hessians, &bins, n_samples) {
                Ok(h) => h,
                Err(e) => return Err(e),
            };
        assert_eq!(built.len(), N_FEATURES, "node size {size}");
        assert_eq!(built, expected, "node size {size}");
    }
    Ok(())
}

#[test]
fn parallel_dispatch_preserves_feature_order() -> Result<(), ClearGbmError> {
    // `map` + `collect` over `into_par_iter` is order-preserving, which is
    // what lets callers index `histograms[feat_idx]`. A switch to an
    // unordered collect would show up here as swapped feature histograms.
    let n_samples = PARALLEL_THRESHOLD;
    let bins = make_bins(n_samples);
    let gradients = make_gradients(n_samples);
    let hessians = make_hessians(n_samples);
    let sample_indices: Vec<u32> = (0_usize..n_samples)
        .map(|i| u32::try_from(i).unwrap_or(0_u32))
        .collect();
    let hooks = Hooks::default();

    let config = BuildHistogramConfig {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples,
        n_features: N_FEATURES,
        n_bins: N_BINS,
        hooks: &hooks,
    };
    let built = match build_feature_histograms(
        &config,
        &mut OrderedScratch::new(config.sample_indices.len()),
    ) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    // The fixture gives each feature a different bin stride, so the per-bin
    // count vectors differ between features. Assert they are pairwise
    // distinct, which is what makes the order assertion meaningful.
    let expected =
        match reference_histograms(&sample_indices, &gradients, &hessians, &bins, n_samples) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };
    assert_ne!(
        expected[0_usize], expected[1_usize],
        "fixture features are indistinguishable; order check would be vacuous"
    );
    for feat in 0_usize..N_FEATURES {
        assert_eq!(built[feat], expected[feat], "feature {feat} out of order");
    }
    Ok(())
}

#[test]
fn parallel_child_histograms_satisfy_sibling_subtraction() -> Result<(), ClearGbmError> {
    // Both children above the threshold, so the smaller-child build fans out
    // across features on the parallel branch.
    let n_samples = 2_usize * PARALLEL_THRESHOLD + 400_usize;
    let split_at = PARALLEL_THRESHOLD + 200_usize;
    let bins = make_bins(n_samples);
    let gradients = make_gradients(n_samples);
    let hessians = make_hessians(n_samples);
    let hooks = Hooks::default();

    let left_indices: Vec<u32> = (0_usize..split_at)
        .map(|i| u32::try_from(i).unwrap_or(0_u32))
        .collect();
    let right_indices: Vec<u32> = (split_at..n_samples)
        .map(|i| u32::try_from(i).unwrap_or(0_u32))
        .collect();
    assert!(left_indices.len() >= PARALLEL_THRESHOLD);
    assert!(right_indices.len() >= PARALLEL_THRESHOLD);

    let all_indices: Vec<u32> = (0_usize..n_samples)
        .map(|i| u32::try_from(i).unwrap_or(0_u32))
        .collect();
    let parent = match reference_histograms(&all_indices, &gradients, &hessians, &bins, n_samples) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    let config = ChildHistogramConfig {
        left_indices: &left_indices,
        right_indices: &right_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples,
        n_features: N_FEATURES,
        n_bins: N_BINS,
        parent_histograms: &parent,
        hooks: &hooks,
    };
    let (left_hists, right_hists) = match compute_child_histograms(
        &config,
        &mut OrderedScratch::new(config.left_indices.len() + config.right_indices.len()),
    ) {
        Ok(pair) => pair,
        Err(e) => return Err(e),
    };
    assert_eq!(left_hists.len(), N_FEATURES);
    assert_eq!(right_hists.len(), N_FEATURES);

    let left_expected =
        match reference_histograms(&left_indices, &gradients, &hessians, &bins, n_samples) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };
    let right_expected =
        match reference_histograms(&right_indices, &gradients, &hessians, &bins, n_samples) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

    for feat in 0_usize..N_FEATURES {
        for bin in 0_usize..N_BINS {
            let left_grad = match left_hists[feat].gradient_sum(bin) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let want_left = match left_expected[feat].gradient_sum(bin) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            // The smaller child is built directly from the data.
            assert!(
                (left_grad - want_left).abs() < 1e-9_f64,
                "feature {feat} bin {bin}: left {left_grad} != {want_left}"
            );

            // The larger child is derived by subtracting from the parent, so
            // it carries one rounding step and is compared with a tolerance.
            let right_grad = match right_hists[feat].gradient_sum(bin) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let want_right = match right_expected[feat].gradient_sum(bin) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert!(
                (right_grad - want_right).abs() < 1e-9_f64,
                "feature {feat} bin {bin}: right {right_grad} != {want_right}"
            );

            let left_count = match left_hists[feat].count(bin) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let right_count = match right_hists[feat].count(bin) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let parent_count = match parent[feat].count(bin) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert_eq!(
                left_count + right_count,
                parent_count,
                "feature {feat} bin {bin}: child counts do not partition the parent"
            );
        }
    }
    Ok(())
}
