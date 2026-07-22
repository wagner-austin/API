//! Example-based unit tests for histogram module.

use crate::error::ClearGbmError;
use crate::histogram::{
    build_histogram, build_histogram_ordered_trusted, reorder_grad_hess_into, subtract_histogram,
};
use crate::narrow::score_narrow;
use crate::types::HistogramBuffer;

#[test]
fn test_build_histogram_simple() -> Result<(), ClearGbmError> {
    // Inner function to test both Ok and Err paths separately
    fn test_ok_path() -> Result<(), ClearGbmError> {
        let sample_indices = vec![0_u32, 1_u32, 2_u32];
        let gradients = vec![0.1_f32, 0.2_f32, 0.3_f32];
        let hessians = vec![1.0_f32, 1.0_f32, 1.0_f32];
        let bins = vec![0_u8, 1_u8, 0_u8];
        let n_bins = 3_usize;

        let hist = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Use direct slice access to avoid unreachable error branches
        // Bin 0: samples 0 and 2, gradients 0.1 + 0.3 = 0.4
        assert!((hist.gradient_sums()[0_usize] - 0.4_f64).abs() < 1e-6_f64);
        assert_eq!(hist.counts()[0_usize], 2_usize);

        // Bin 1: sample 1, gradient 0.2
        assert!((hist.gradient_sums()[1_usize] - 0.2_f64).abs() < 1e-6_f64);
        assert_eq!(hist.counts()[1_usize], 1_usize);

        // Bin 2: empty
        assert!(hist.gradient_sums()[2_usize].abs() < 1e-10_f64);
        assert_eq!(hist.counts()[2_usize], 0_usize);
        Ok(())
    }
    fn test_err_path() -> Result<(), ClearGbmError> {
        let sample_indices: Vec<u32> = vec![];
        let gradients = vec![0.1_f32];
        let hessians = vec![1.0_f32];
        let bins = vec![0_u8];
        let n_bins = 3_usize;

        match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
    // Run Ok path
    match test_ok_path() {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    // Run Err path and verify error
    assert!(test_err_path().is_err());
    Ok(())
}

#[test]
fn test_build_histogram_subset_of_samples() -> Result<(), ClearGbmError> {
    // Inner function to test both Ok and Err paths
    fn test_ok_path() -> Result<(), ClearGbmError> {
        let sample_indices = vec![0_u32, 2_u32]; // Only samples 0 and 2
        let gradients = vec![0.1_f32, 0.2_f32, 0.3_f32];
        let hessians = vec![1.0_f32, 1.0_f32, 1.0_f32];
        let bins = vec![0_u8, 1_u8, 0_u8];
        let n_bins = 2_usize;

        let hist = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Use direct slice access to avoid unreachable error branches
        // Only samples 0 and 2 are included, both in bin 0
        assert!((hist.gradient_sums()[0_usize] - 0.4_f64).abs() < 1e-6_f64);
        assert_eq!(hist.counts()[0_usize], 2_usize);

        // Bin 1 should be empty (sample 1 not included)
        assert_eq!(hist.counts()[1_usize], 0_usize);
        Ok(())
    }
    fn test_err_path() -> Result<(), ClearGbmError> {
        let sample_indices: Vec<u32> = vec![];
        let gradients = vec![0.1_f32];
        let hessians = vec![1.0_f32];
        let bins = vec![0_u8];
        let n_bins = 2_usize;

        match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
    // Run Ok path
    match test_ok_path() {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    // Run Err path and verify error
    assert!(test_err_path().is_err());
    Ok(())
}

#[test]
fn test_build_histogram_empty_indices_fails() -> Result<(), ClearGbmError> {
    let sample_indices: Vec<u32> = vec![];
    let gradients = vec![0.1_f32];
    let hessians = vec![1.0_f32];
    let bins = vec![0_u8];

    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, 2_usize);

    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::EmptyInput { .. })));
    Ok(())
}

#[test]
fn test_build_histogram_index_out_of_bounds() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_u32, 5_u32]; // 5 is out of bounds
    let gradients = vec![0.1_f32, 0.2_f32, 0.3_f32];
    let hessians = vec![1.0_f32, 1.0_f32, 1.0_f32];
    let bins = vec![0_u8, 1_u8, 0_u8];

    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, 3_usize);

    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::SampleIndexOutOfBounds { index: 5_usize, .. })
    ));
    Ok(())
}

#[test]
fn test_build_histogram_hessians_length_mismatch() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_u32, 1_u32];
    let gradients = vec![0.1_f32, 0.2_f32, 0.3_f32];
    let hessians = vec![1.0_f32, 1.0_f32]; // Wrong length
    let bins = vec![0_u8, 1_u8, 0_u8];

    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, 3_usize);

    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_build_histogram_bins_length_mismatch() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_u32, 1_u32];
    let gradients = vec![0.1_f32, 0.2_f32, 0.3_f32];
    let hessians = vec![1.0_f32, 1.0_f32, 1.0_f32];
    let bins = vec![0_u8, 1_u8]; // Wrong length

    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, 3_usize);

    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_build_histogram_bin_out_of_bounds() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![0.1_f32, 0.2_f32, 0.3_f32];
    let hessians = vec![1.0_f32, 1.0_f32, 1.0_f32];
    let bins = vec![0_u8, 5_u8, 0_u8]; // bin 5 is out of bounds for n_bins=3

    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, 3_usize);

    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::BinIndexOutOfBounds {
            bin: 5_usize,
            n_bins: 3_usize
        })
    ));
    Ok(())
}

#[test]
fn test_build_histogram_large() -> Result<(), ClearGbmError> {
    let n = 10000_usize;
    let sample_indices: Vec<u32> = (0_u32..).take(n).collect();
    let gradients: Vec<f32> = (0_u32..10000_u32)
        .map(|i| score_narrow(f64::from(i) * 0.001_f64))
        .collect();
    let hessians: Vec<f32> = vec![1.0_f32; n];
    let bins: Vec<u8> = (0_usize..n)
        .map(|i| {
            let b = i % 64_usize;
            match u8::try_from(b) {
                Ok(v) => v,
                Err(_) => 0_u8,
            }
        })
        .collect();

    let hist = match build_histogram(&sample_indices, &gradients, &hessians, &bins, 64_usize) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    // Verify total count using direct slice access (no error branches)
    let total_count: usize = hist.counts().iter().sum();
    assert_eq!(total_count, n);
    Ok(())
}

#[test]
fn test_subtract_histogram_simple() -> Result<(), ClearGbmError> {
    let mut parent = HistogramBuffer::new(3_usize);
    match parent.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match parent.accumulate(0_usize, 0.3_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match parent.accumulate(1_usize, 0.2_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut child = HistogramBuffer::new(3_usize);
    match child.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let sibling = match subtract_histogram(&parent, &child) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    // Bin 0: parent has 2 samples (0.5+0.3=0.8), child has 1 (0.5)
    // Sibling should have 1 sample with gradient 0.3
    // Use direct slice access to avoid unreachable error branches
    assert_eq!(sibling.counts()[0_usize], 1_usize);
    assert!((sibling.gradient_sums()[0_usize] - 0.3_f64).abs() < 1e-10_f64);

    // Bin 1: parent has 1 sample, child has 0
    assert_eq!(sibling.counts()[1_usize], 1_usize);
    Ok(())
}

#[test]
fn test_subtract_histogram_shape_mismatch() -> Result<(), ClearGbmError> {
    let parent = HistogramBuffer::new(3_usize);
    let child = HistogramBuffer::new(5_usize);

    let result = subtract_histogram(&parent, &child);

    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_subtract_histogram_empty() -> Result<(), ClearGbmError> {
    let parent = HistogramBuffer::new(3_usize);
    let child = HistogramBuffer::new(3_usize);

    let sibling = match subtract_histogram(&parent, &child) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    // Use direct slice access to avoid unreachable error branches
    for i in 0_usize..3_usize {
        assert_eq!(sibling.counts()[i], 0_usize);
        assert!(sibling.gradient_sums()[i].abs() < f64::EPSILON);
    }
    Ok(())
}

// ============================================================================
// Ordered (fast-path) histogram tests — equivalence with the classic path
// ============================================================================
//
// Every ordered-path test constructs its inputs so
// `ordered_gradients[i] = gradients[sample_indices[i]]` and
// `ordered_hessians[i] = hessians[sample_indices[i]]` — the invariant the
// tree builder maintains via `reorder_grad_hess_into`. The tests verify the
// ordered fast path produces bit-identical output to the classic path.

#[test]
fn test_build_histogram_ordered_matches_classic_simple() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![0.1_f32, 0.2_f32, 0.3_f32];
    let hessians = vec![1.0_f32, 1.0_f32, 1.0_f32];
    let bins = vec![0_u8, 1_u8, 0_u8];
    let n_bins = 3_usize;

    // Reference: classic path.
    let classic = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    // Fast path: reorder, then dispatch.
    let mut ordered_g = vec![0.0_f32; sample_indices.len()];
    let mut ordered_h = vec![0.0_f32; sample_indices.len()];
    reorder_grad_hess_into(
        &sample_indices,
        &gradients,
        &hessians,
        &mut ordered_g,
        &mut ordered_h,
    );
    let fast = build_histogram_ordered_trusted(
        &sample_indices,
        &ordered_g,
        &ordered_h,
        &bins,
        n_bins,
    );

    assert_eq!(classic.n_bins(), fast.n_bins());
    assert_eq!(classic.counts(), fast.counts());
    for bin in 0_usize..n_bins {
        let dg = (classic.gradient_sums()[bin] - fast.gradient_sums()[bin]).abs();
        let dh = (classic.hessian_sums()[bin] - fast.hessian_sums()[bin]).abs();
        assert!(dg < 1e-12_f64, "gradient mismatch at bin {bin}: {dg}");
        assert!(dh < 1e-12_f64, "hessian mismatch at bin {bin}: {dh}");
    }
    Ok(())
}

#[test]
fn test_build_histogram_ordered_matches_classic_subset() -> Result<(), ClearGbmError> {
    // Non-identity permutation — sample_indices skips sample 1.
    let sample_indices = vec![0_u32, 2_u32];
    let gradients = vec![0.1_f32, 0.2_f32, 0.3_f32];
    let hessians = vec![1.0_f32, 1.0_f32, 1.0_f32];
    let bins = vec![0_u8, 1_u8, 0_u8];
    let n_bins = 2_usize;

    let classic = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    let mut ordered_g = vec![0.0_f32; sample_indices.len()];
    let mut ordered_h = vec![0.0_f32; sample_indices.len()];
    reorder_grad_hess_into(
        &sample_indices,
        &gradients,
        &hessians,
        &mut ordered_g,
        &mut ordered_h,
    );
    let fast = build_histogram_ordered_trusted(
        &sample_indices,
        &ordered_g,
        &ordered_h,
        &bins,
        n_bins,
    );

    assert_eq!(classic.counts(), fast.counts());
    for bin in 0_usize..n_bins {
        assert!((classic.gradient_sums()[bin] - fast.gradient_sums()[bin]).abs() < 1e-12_f64);
        assert!((classic.hessian_sums()[bin] - fast.hessian_sums()[bin]).abs() < 1e-12_f64);
    }
    Ok(())
}

#[test]
fn test_build_histogram_ordered_matches_classic_large_unrolled() -> Result<(), ClearGbmError> {
    // Exercise the 8-wide unroll: 10000 samples = 1250 full chunks + 0 remainder.
    let n = 10000_usize;
    let sample_indices: Vec<u32> = (0_u32..).take(n).collect();
    let gradients: Vec<f32> = (0_u32..10000_u32)
        .map(|i| score_narrow(f64::from(i) * 0.001_f64))
        .collect();
    let hessians: Vec<f32> = (0_u32..10000_u32)
        .map(|i| score_narrow(f64::from(i) * 0.002_f64 + 0.1_f64))
        .collect();
    let bins: Vec<u8> = (0_usize..n)
        .map(|i| {
            let b = i % 64_usize;
            match u8::try_from(b) {
                Ok(v) => v,
                Err(_) => 0_u8,
            }
        })
        .collect();
    let n_bins = 64_usize;

    let classic = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    let mut ordered_g = vec![0.0_f32; sample_indices.len()];
    let mut ordered_h = vec![0.0_f32; sample_indices.len()];
    reorder_grad_hess_into(
        &sample_indices,
        &gradients,
        &hessians,
        &mut ordered_g,
        &mut ordered_h,
    );
    let fast = build_histogram_ordered_trusted(
        &sample_indices,
        &ordered_g,
        &ordered_h,
        &bins,
        n_bins,
    );

    assert_eq!(classic.counts(), fast.counts());
    for bin in 0_usize..n_bins {
        // 156 samples per bin × sum-of-arithmetic-progression → floating-point noise
        // in the ~1e-13 range. Tolerance chosen to catch algorithmic mismatch but
        // absorb reorder-order rounding drift.
        let dg = (classic.gradient_sums()[bin] - fast.gradient_sums()[bin]).abs();
        let dh = (classic.hessian_sums()[bin] - fast.hessian_sums()[bin]).abs();
        assert!(dg < 1e-9_f64, "gradient drift at bin {bin}: {dg}");
        assert!(dh < 1e-9_f64, "hessian drift at bin {bin}: {dh}");
    }
    Ok(())
}

#[test]
fn test_build_histogram_ordered_matches_classic_tail_remainder() -> Result<(), ClearGbmError> {
    // Force the scalar tail: 11 samples = 1 chunk of 8 + 3 remainder.
    let n = 11_usize;
    let sample_indices: Vec<u32> = (0_u32..).take(n).collect();
    let gradients: Vec<f32> = (0_u32..11_u32).map(|i| score_narrow(f64::from(i))).collect();
    let hessians: Vec<f32> = (10_u32..21_u32).map(|i| score_narrow(f64::from(i))).collect();
    let bins: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2];
    let n_bins = 4_usize;

    let classic = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    let mut ordered_g = vec![0.0_f32; sample_indices.len()];
    let mut ordered_h = vec![0.0_f32; sample_indices.len()];
    reorder_grad_hess_into(
        &sample_indices,
        &gradients,
        &hessians,
        &mut ordered_g,
        &mut ordered_h,
    );
    let fast = build_histogram_ordered_trusted(
        &sample_indices,
        &ordered_g,
        &ordered_h,
        &bins,
        n_bins,
    );

    assert_eq!(classic.counts(), fast.counts());
    for bin in 0_usize..n_bins {
        assert!((classic.gradient_sums()[bin] - fast.gradient_sums()[bin]).abs() < 1e-12_f64);
        assert!((classic.hessian_sums()[bin] - fast.hessian_sums()[bin]).abs() < 1e-12_f64);
    }
    Ok(())
}

#[test]
fn test_build_histogram_ordered_permuted_indices() -> Result<(), ClearGbmError> {
    // Non-monotonic sample_indices — the tree builder emits these on non-root nodes.
    let sample_indices = vec![5_u32, 2_u32, 8_u32, 1_u32, 9_u32, 0_u32, 4_u32];
    let gradients: Vec<f32> = (0_u32..10_u32)
        .map(|i| score_narrow(f64::from(i) * 0.5_f64))
        .collect();
    let hessians: Vec<f32> = vec![1.0_f32; 10_usize];
    let bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
    let n_bins = 3_usize;

    let classic = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    let mut ordered_g = vec![0.0_f32; sample_indices.len()];
    let mut ordered_h = vec![0.0_f32; sample_indices.len()];
    reorder_grad_hess_into(
        &sample_indices,
        &gradients,
        &hessians,
        &mut ordered_g,
        &mut ordered_h,
    );
    let fast = build_histogram_ordered_trusted(
        &sample_indices,
        &ordered_g,
        &ordered_h,
        &bins,
        n_bins,
    );

    assert_eq!(classic.counts(), fast.counts());
    for bin in 0_usize..n_bins {
        assert!((classic.gradient_sums()[bin] - fast.gradient_sums()[bin]).abs() < 1e-12_f64);
        assert!((classic.hessian_sums()[bin] - fast.hessian_sums()[bin]).abs() < 1e-12_f64);
    }
    Ok(())
}

#[test]
fn test_reorder_grad_hess_into_populates_correctly() {
    let sample_indices = vec![3_u32, 0_u32, 5_u32, 2_u32];
    let gradients = vec![10.0_f32, 11.0_f32, 12.0_f32, 13.0_f32, 14.0_f32, 15.0_f32];
    let hessians = vec![20.0_f32, 21.0_f32, 22.0_f32, 23.0_f32, 24.0_f32, 25.0_f32];
    let mut ordered_g = vec![0.0_f32; 4_usize];
    let mut ordered_h = vec![0.0_f32; 4_usize];

    reorder_grad_hess_into(
        &sample_indices,
        &gradients,
        &hessians,
        &mut ordered_g,
        &mut ordered_h,
    );

    // Ordered arrays reflect gradients/hessians AT sample_indices[i].
    assert_eq!(ordered_g, vec![13.0_f32, 10.0_f32, 15.0_f32, 12.0_f32]);
    assert_eq!(ordered_h, vec![23.0_f32, 20.0_f32, 25.0_f32, 22.0_f32]);
}

#[test]
#[should_panic]
fn test_reorder_grad_hess_into_panics_on_output_length_mismatch() {
    let sample_indices = vec![0_u32, 1_u32];
    let gradients = vec![1.0_f32, 2.0_f32];
    let hessians = vec![1.0_f32, 2.0_f32];
    // Wrong size — should panic per the debug assertion.
    let mut ordered_g = vec![0.0_f32; 3_usize];
    let mut ordered_h = vec![0.0_f32; 3_usize];
    reorder_grad_hess_into(
        &sample_indices,
        &gradients,
        &hessians,
        &mut ordered_g,
        &mut ordered_h,
    );
}
