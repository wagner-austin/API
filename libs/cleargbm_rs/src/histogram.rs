//! Histogram building for gradient boosting.
//!
//! Implements O(n) histogram construction with NaN handling.
//! This is the primary hot path and performance-critical code.

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

/// Builds a histogram from sample gradients and hessians.
///
/// This is the core O(n) operation that accumulates gradient statistics
/// into bins for split finding.
///
/// # Args
///
/// * `sample_indices` - Indices of samples at this node.
/// * `gradients` - Gradient values for all samples.
/// * `hessians` - Hessian values for all samples.
/// * `bins` - Pre-computed bin assignments for this feature.
/// * `n_bins` - Number of bins (including NaN bin).
///
/// # Returns
///
/// A `HistogramBuffer` with accumulated statistics.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` - If `sample_indices` is empty.
/// * `ClearGbmError::SampleIndexOutOfBounds` - If any index is out of bounds.
/// * `ClearGbmError::ShapeMismatch` - If array lengths don't match.
/// * `ClearGbmError::BinIndexOutOfBounds` - If any bin index is out of bounds.
pub fn build_histogram(
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[usize],
    n_bins: usize,
) -> Result<HistogramBuffer, ClearGbmError> {
    if sample_indices.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "sample_indices cannot be empty".to_string(),
        });
    }

    let n_samples = gradients.len();

    // Validate array lengths match
    if hessians.len() != n_samples {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("hessians length {n_samples}"),
            got: format!("hessians length {}", hessians.len()),
        });
    }
    if bins.len() != n_samples {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("bins length {n_samples}"),
            got: format!("bins length {}", bins.len()),
        });
    }

    let mut histogram = HistogramBuffer::new(n_bins);

    // Core hot loop - this is where Rust shines
    for &idx in sample_indices {
        if idx >= n_samples {
            return Err(ClearGbmError::SampleIndexOutOfBounds {
                index: idx,
                n_samples,
            });
        }

        let bin = bins[idx];
        let grad = gradients[idx];
        let hess = hessians[idx];

        match histogram.accumulate(bin, grad, hess) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    Ok(histogram)
}

/// Computes sibling histogram by subtraction (2x speedup).
///
/// Given parent histogram and one child histogram, computes the
/// other child by subtraction: sibling = parent - child.
///
/// This matches the Python `subtract_histogram` function.
///
/// # Args
///
/// * `parent` - Parent node histogram.
/// * `child` - One child node histogram.
///
/// # Returns
///
/// The sibling histogram.
///
/// # Errors
///
/// * `ClearGbmError::ShapeMismatch` - If histograms have different `n_bins`.
pub fn subtract_histogram(
    parent: &HistogramBuffer,
    child: &HistogramBuffer,
) -> Result<HistogramBuffer, ClearGbmError> {
    let mut sibling = HistogramBuffer::new(parent.n_bins());
    match sibling.subtract_into(parent, child) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(sibling)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ClearGbmError;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;

    // =========================================================================
    // Property-based tests with proptest
    // =========================================================================

    #[test]
    fn prop_histogram_sums_equal_input_sums() -> Result<(), ClearGbmError> {
        let config = proptest::test_runner::Config::with_cases(100);
        let mut runner = proptest::test_runner::TestRunner::new(config);
        runner
            .run(
                &(1_usize..50_usize, 2_usize..10_usize),
                |(n_samples, n_bins)| {
                    // Generate gradients using accumulator pattern (no casting)
                    let gradients: Vec<f64> = {
                        let mut v = Vec::with_capacity(n_samples);
                        let mut acc = 0.0_f64;
                        for _ in 0_usize..n_samples {
                            v.push(acc);
                            acc += 0.1_f64;
                        }
                        v
                    };
                    let hessians: Vec<f64> = (0_usize..n_samples).map(|_| 1.0_f64).collect();
                    let sample_indices: Vec<usize> = (0_usize..n_samples).collect();
                    let bins: Vec<usize> = (0_usize..n_samples).map(|i| i % n_bins).collect();

                    let hist = match build_histogram(
                        &sample_indices,
                        &gradients,
                        &hessians,
                        &bins,
                        n_bins,
                    ) {
                        Ok(h) => h,
                        Err(e) => {
                            return Err(proptest::test_runner::TestCaseError::fail(format!(
                                "build_histogram failed: {}",
                                e
                            )));
                        }
                    };

                    // Sum of all bin gradient sums should equal sum of input gradients
                    let total_grad: f64 = hist.gradient_sums().iter().sum();
                    let expected_grad: f64 = gradients.iter().sum();
                    prop_assert!(
                        (total_grad - expected_grad).abs() < 1e-9_f64,
                        "Gradient sum mismatch: {} vs {}",
                        total_grad,
                        expected_grad
                    );

                    // Sum of all bin counts should equal n_samples
                    let total_count: usize = hist.counts().iter().sum();
                    prop_assert_eq!(total_count, n_samples, "Count mismatch");
                    Ok(())
                },
            )
            .map_err(|e| ClearGbmError::InvalidParameter {
                name: "proptest".to_string(),
                reason: format!("{}", e),
            })
    }

    #[test]
    fn prop_subtract_histogram_identity() -> Result<(), ClearGbmError> {
        let config = proptest::test_runner::Config::with_cases(100);
        let mut runner = proptest::test_runner::TestRunner::new(config);
        runner
            .run(&(2_usize..10_usize), |n_bins| {
                // parent - parent = zero histogram
                let mut parent = HistogramBuffer::new(n_bins);
                for bin in 0_usize..n_bins {
                    let _ = parent.accumulate(bin, 1.0_f64, 1.0_f64);
                }

                let sibling = match subtract_histogram(&parent, &parent) {
                    Ok(s) => s,
                    Err(e) => {
                        return Err(proptest::test_runner::TestCaseError::fail(format!(
                            "subtract_histogram failed: {}",
                            e
                        )));
                    }
                };

                // All values should be zero
                for bin in 0_usize..n_bins {
                    let g = match sibling.gradient_sum(bin) {
                        Ok(v) => v,
                        Err(e) => {
                            return Err(proptest::test_runner::TestCaseError::fail(format!(
                                "gradient_sum failed: {}",
                                e
                            )));
                        }
                    };
                    prop_assert!(g.abs() < 1e-10_f64, "Expected 0 gradient, got {}", g);
                    let c = match sibling.count(bin) {
                        Ok(v) => v,
                        Err(e) => {
                            return Err(proptest::test_runner::TestCaseError::fail(format!(
                                "count failed: {}",
                                e
                            )));
                        }
                    };
                    prop_assert_eq!(c, 0_usize, "Expected 0 count");
                }
                Ok(())
            })
            .map_err(|e| ClearGbmError::InvalidParameter {
                name: "proptest".to_string(),
                reason: format!("{}", e),
            })
    }

    #[test]
    fn prop_subtract_histogram_correctness() -> Result<(), ClearGbmError> {
        let config = proptest::test_runner::Config::with_cases(100);
        let mut runner = proptest::test_runner::TestRunner::new(config);
        runner
            .run(
                &(2_usize..8_usize, 0.1_f64..0.9_f64),
                |(n_bins, child_fraction)| {
                    // Build parent histogram
                    let mut parent = HistogramBuffer::new(n_bins);
                    for bin in 0_usize..n_bins {
                        let _ = parent.accumulate(bin, 10.0_f64, 5.0_f64);
                    }

                    // Build child histogram (fraction of parent)
                    let mut child = HistogramBuffer::new(n_bins);
                    for bin in 0_usize..n_bins {
                        let _ = child.accumulate(
                            bin,
                            10.0_f64 * child_fraction,
                            5.0_f64 * child_fraction,
                        );
                    }

                    let sibling = match subtract_histogram(&parent, &child) {
                        Ok(s) => s,
                        Err(e) => {
                            return Err(proptest::test_runner::TestCaseError::fail(format!(
                                "subtract_histogram failed: {}",
                                e
                            )));
                        }
                    };

                    // sibling should be parent - child
                    for bin in 0_usize..n_bins {
                        let s = match sibling.gradient_sum(bin) {
                            Ok(v) => v,
                            Err(e) => {
                                return Err(proptest::test_runner::TestCaseError::fail(format!(
                                    "sibling.gradient_sum failed: {}",
                                    e
                                )));
                            }
                        };
                        let p = match parent.gradient_sum(bin) {
                            Ok(v) => v,
                            Err(e) => {
                                return Err(proptest::test_runner::TestCaseError::fail(format!(
                                    "parent.gradient_sum failed: {}",
                                    e
                                )));
                            }
                        };
                        let c = match child.gradient_sum(bin) {
                            Ok(v) => v,
                            Err(e) => {
                                return Err(proptest::test_runner::TestCaseError::fail(format!(
                                    "child.gradient_sum failed: {}",
                                    e
                                )));
                            }
                        };
                        let expected = p - c;
                        prop_assert!(
                            (s - expected).abs() < 1e-9_f64,
                            "Gradient mismatch at bin {}: {} vs {}",
                            bin,
                            s,
                            expected
                        );
                    }
                    Ok(())
                },
            )
            .map_err(|e| ClearGbmError::InvalidParameter {
                name: "proptest".to_string(),
                reason: format!("{}", e),
            })
    }

    // =========================================================================
    // Example-based tests
    // =========================================================================

    #[test]
    fn test_build_histogram_simple() -> Result<(), ClearGbmError> {
        let sample_indices = vec![0_usize, 1_usize, 2_usize];
        let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![0_usize, 1_usize, 0_usize]; // samples 0,2 in bin 0, sample 1 in bin 1
        let n_bins = 3_usize;

        let hist = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Bin 0: samples 0 and 2, gradients 0.1 + 0.3 = 0.4
        let grad_0 = match hist.gradient_sum(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((grad_0 - 0.4_f64).abs() < 1e-10_f64);
        let count_0 = match hist.count(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_0, 2_usize);

        // Bin 1: sample 1, gradient 0.2
        let grad_1 = match hist.gradient_sum(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((grad_1 - 0.2_f64).abs() < 1e-10_f64);
        let count_1 = match hist.count(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_1, 1_usize);

        // Bin 2: empty
        let grad_2 = match hist.gradient_sum(2_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!(grad_2.abs() < 1e-10_f64);
        let count_2 = match hist.count(2_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_2, 0_usize);
        Ok(())
    }

    #[test]
    fn test_build_histogram_subset_of_samples() -> Result<(), ClearGbmError> {
        let sample_indices = vec![0_usize, 2_usize]; // Only samples 0 and 2
        let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![0_usize, 1_usize, 0_usize];
        let n_bins = 2_usize;

        let hist = match build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Only samples 0 and 2 are included, both in bin 0
        let grad_0 = match hist.gradient_sum(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((grad_0 - 0.4_f64).abs() < 1e-10_f64);
        let count_0 = match hist.count(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_0, 2_usize);

        // Bin 1 should be empty (sample 1 not included)
        let count_1 = match hist.count(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_1, 0_usize);
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

        // Verify total count
        let counts_result: Result<Vec<_>, _> = (0_usize..64_usize).map(|i| hist.count(i)).collect();
        let counts = match counts_result {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let total_count: usize = counts.iter().sum();
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
        let count_0 = match sibling.count(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_0, 1_usize);
        let grad = match sibling.gradient_sum(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((grad - 0.3_f64).abs() < 1e-10_f64);

        // Bin 1: parent has 1 sample, child has 0
        let count_1 = match sibling.count(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_1, 1_usize);
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

        for i in 0_usize..3_usize {
            let count = match sibling.count(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert_eq!(count, 0_usize);
            let grad = match sibling.gradient_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert!(grad.abs() < f64::EPSILON);
        }
        Ok(())
    }
}
