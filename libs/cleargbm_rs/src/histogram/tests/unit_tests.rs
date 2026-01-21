//! Example-based unit tests for histogram module.

use crate::error::ClearGbmError;
use crate::histogram::{build_histogram, subtract_histogram};
use crate::types::HistogramBuffer;

#[test]
fn test_build_histogram_simple() -> Result<(), ClearGbmError> {
    // Inner function to test both Ok and Err paths separately
    fn test_ok_path() -> Result<(), ClearGbmError> {
        let sample_indices = vec![0_usize, 1_usize, 2_usize];
        let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![0_usize, 1_usize, 0_usize];
        let n_bins = 3_usize;

        let hist = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Use direct slice access to avoid unreachable error branches
        // Bin 0: samples 0 and 2, gradients 0.1 + 0.3 = 0.4
        assert!((hist.gradient_sums()[0_usize] - 0.4_f64).abs() < 1e-10_f64);
        assert_eq!(hist.counts()[0_usize], 2_usize);

        // Bin 1: sample 1, gradient 0.2
        assert!((hist.gradient_sums()[1_usize] - 0.2_f64).abs() < 1e-10_f64);
        assert_eq!(hist.counts()[1_usize], 1_usize);

        // Bin 2: empty
        assert!(hist.gradient_sums()[2_usize].abs() < 1e-10_f64);
        assert_eq!(hist.counts()[2_usize], 0_usize);
        Ok(())
    }
    fn test_err_path() -> Result<(), ClearGbmError> {
        let sample_indices: Vec<usize> = vec![];
        let gradients = vec![0.1_f64];
        let hessians = vec![1.0_f64];
        let bins = vec![0_usize];
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
        let sample_indices = vec![0_usize, 2_usize]; // Only samples 0 and 2
        let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![0_usize, 1_usize, 0_usize];
        let n_bins = 2_usize;

        let hist = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Use direct slice access to avoid unreachable error branches
        // Only samples 0 and 2 are included, both in bin 0
        assert!((hist.gradient_sums()[0_usize] - 0.4_f64).abs() < 1e-10_f64);
        assert_eq!(hist.counts()[0_usize], 2_usize);

        // Bin 1 should be empty (sample 1 not included)
        assert_eq!(hist.counts()[1_usize], 0_usize);
        Ok(())
    }
    fn test_err_path() -> Result<(), ClearGbmError> {
        let sample_indices: Vec<usize> = vec![];
        let gradients = vec![0.1_f64];
        let hessians = vec![1.0_f64];
        let bins = vec![0_usize];
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
    let sample_indices: Vec<usize> = vec![];
    let gradients = vec![0.1_f64];
    let hessians = vec![1.0_f64];
    let bins = vec![0_usize];

    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, 2_usize);

    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::EmptyInput { .. })));
    Ok(())
}

#[test]
fn test_build_histogram_index_out_of_bounds() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_usize, 5_usize]; // 5 is out of bounds
    let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![0_usize, 1_usize, 0_usize];

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
    let sample_indices = vec![0_usize, 1_usize];
    let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
    let hessians = vec![1.0_f64, 1.0_f64]; // Wrong length
    let bins = vec![0_usize, 1_usize, 0_usize];

    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, 3_usize);

    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_build_histogram_bins_length_mismatch() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_usize, 1_usize];
    let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![0_usize, 1_usize]; // Wrong length

    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, 3_usize);

    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_build_histogram_bin_out_of_bounds() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_usize, 1_usize, 2_usize];
    let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![0_usize, 5_usize, 0_usize]; // bin 5 is out of bounds for n_bins=3

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
    let sample_indices: Vec<usize> = (0_usize..n).collect();
    let gradients: Vec<f64> = (0_u32..10000_u32)
        .map(|i| f64::from(i) * 0.001_f64)
        .collect();
    let hessians: Vec<f64> = vec![1.0_f64; n];
    let bins: Vec<usize> = (0_usize..n).map(|i| i % 64_usize).collect();

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
