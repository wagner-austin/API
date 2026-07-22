//! Property-based tests using proptest for tree module.

use crate::error::ClearGbmError;
use crate::tree::builder::{compute_leaf_value, should_stop, split_samples, EPSILON};
use proptest::prop_assert;
use proptest::prop_assert_eq;

#[test]
fn prop_compute_leaf_value_zero_hessian_returns_zero() -> Result<(), ClearGbmError> {
    let config = proptest::test_runner::Config::with_cases(100);
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(
            &(
                -1000.0_f64..1000.0_f64,
                0.0_f64..10.0_f64,
                0.0_f64..10.0_f64,
            ),
            |(gradient, reg_alpha, reg_lambda)| {
                // When hessian + lambda is near zero, should return 0
                let hessian = 0.0_f64;
                if reg_lambda < EPSILON {
                    let value = compute_leaf_value(gradient, hessian, reg_alpha, reg_lambda);
                    prop_assert!(value.abs() < EPSILON, "Expected 0, got {}", value);
                }
                Ok(())
            },
        )
        .map_err(|e| ClearGbmError::InvalidParameter {
            name: "proptest".to_string(),
            reason: format!("{}", e),
        })
}

#[test]
fn prop_compute_leaf_value_l1_soft_threshold() -> Result<(), ClearGbmError> {
    let config = proptest::test_runner::Config::with_cases(100);
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(
            &(
                -100.0_f64..100.0_f64,
                1.0_f64..100.0_f64,
                0.0_f64..50.0_f64,
                0.0_f64..10.0_f64,
            ),
            |(gradient, hessian, reg_alpha, reg_lambda)| {
                let value = compute_leaf_value(gradient, hessian, reg_alpha, reg_lambda);

                // L1 soft threshold: if |G| <= alpha, value should be 0
                if gradient.abs() <= reg_alpha {
                    prop_assert!(
                        value.abs() < EPSILON,
                        "Expected 0 when |G| <= alpha, got {}",
                        value
                    );
                }

                // Value should be finite
                prop_assert!(value.is_finite(), "Value should be finite, got {}", value);
                Ok(())
            },
        )
        .map_err(|e| ClearGbmError::InvalidParameter {
            name: "proptest".to_string(),
            reason: format!("{}", e),
        })
}

#[test]
fn prop_compute_leaf_value_sign_correct() -> Result<(), ClearGbmError> {
    let config = proptest::test_runner::Config::with_cases(100);
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(
            &(-100.0_f64..100.0_f64, 1.0_f64..100.0_f64),
            |(gradient, hessian)| {
                // Without regularization: -G/H
                let value = compute_leaf_value(gradient, hessian, 0.0_f64, 0.0_f64);

                // Sign should be opposite of gradient (when hessian > 0)
                if gradient.abs() > EPSILON {
                    let expected_sign = if gradient > 0.0_f64 {
                        -1.0_f64
                    } else {
                        1.0_f64
                    };
                    let actual_sign = if value > 0.0_f64 { 1.0_f64 } else { -1.0_f64 };
                    prop_assert_eq!(
                        expected_sign,
                        actual_sign,
                        "Sign mismatch: G={}, value={}",
                        gradient,
                        value
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

#[test]
fn prop_should_stop_respects_constraints() -> Result<(), ClearGbmError> {
    let config = proptest::test_runner::Config::with_cases(100);
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(
            &(
                0_usize..20_usize,
                1_usize..1000_usize,
                0_usize..100_usize,
                0_usize..15_usize,
                0_usize..50_usize,
                2_usize..50_usize,
                1_usize..25_usize,
            ),
            |(
                depth,
                n_samples,
                n_leaves,
                max_depth,
                max_leaves,
                min_samples_split,
                min_samples_leaf,
            )| {
                let result = should_stop(
                    depth,
                    n_samples,
                    n_leaves,
                    max_depth,
                    max_leaves,
                    min_samples_split,
                    min_samples_leaf,
                );

                // If max_depth > 0 and depth >= max_depth, must stop
                if max_depth > 0_usize && depth >= max_depth {
                    prop_assert!(result, "Should stop when depth >= max_depth");
                }

                // If max_leaves > 0 and n_leaves + 1 >= max_leaves, must stop
                if max_leaves > 0_usize && n_leaves + 1_usize >= max_leaves {
                    prop_assert!(result, "Should stop when approaching max_leaves");
                }

                // If n_samples < min_samples_split, must stop
                if n_samples < min_samples_split {
                    prop_assert!(result, "Should stop when n_samples < min_samples_split");
                }

                // If n_samples < 2 * min_samples_leaf, must stop
                if n_samples < 2_usize * min_samples_leaf {
                    prop_assert!(result, "Should stop when n_samples < 2 * min_samples_leaf");
                }
                Ok(())
            },
        )
        .map_err(|e| ClearGbmError::InvalidParameter {
            name: "proptest".to_string(),
            reason: format!("{}", e),
        })
}

#[test]
fn prop_split_samples_preserves_count() -> Result<(), ClearGbmError> {
    let config = proptest::test_runner::Config::with_cases(100);
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(
            &(2_usize..20_usize, 0_usize..5_usize, proptest::bool::ANY),
            |(n_samples, split_bin, nan_goes_left)| {
                let n_regular_bins = 6_usize;
                let sample_indices: Vec<usize> = (0_usize..n_samples).collect();

                // Create column-major flat bin storage (1 feature).
                // n_regular_bins = 6, so each `i % n_regular_bins` fits in u8.
                let bins: Vec<u8> = (0_usize..n_samples)
                    .map(|i| {
                        let bin_usize = i % n_regular_bins;
                        u8::try_from(bin_usize).unwrap_or(0_u8)
                    })
                    .collect();

                let (left, right) = split_samples(
                    &sample_indices,
                    &bins,
                    n_samples,
                    0_usize,
                    split_bin,
                    nan_goes_left,
                    n_regular_bins,
                );

                // Total samples should be preserved
                prop_assert_eq!(
                    left.len() + right.len(),
                    n_samples,
                    "Sample count not preserved: left={}, right={}, total={}",
                    left.len(),
                    right.len(),
                    n_samples
                );

                // No duplicates
                let mut all: Vec<usize> = left.iter().chain(right.iter()).copied().collect();
                all.sort();
                all.dedup();
                prop_assert_eq!(all.len(), n_samples, "Duplicate samples found");
                Ok(())
            },
        )
        .map_err(|e| ClearGbmError::InvalidParameter {
            name: "proptest".to_string(),
            reason: format!("{}", e),
        })
}
