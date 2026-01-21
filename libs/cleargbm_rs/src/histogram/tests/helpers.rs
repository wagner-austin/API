//! Helper functions for histogram testing.

use crate::error::ClearGbmError;
use crate::histogram::subtract_histogram;
use crate::types::HistogramBuffer;

// =========================================================================
// Independent assertion check functions (testable for both pass and fail)
// =========================================================================

/// Checks if gradient sum matches expected value within tolerance.
pub fn check_gradient_sum_matches(
    actual: f64,
    expected: f64,
    tolerance: f64,
) -> Result<(), proptest::test_runner::TestCaseError> {
    if (actual - expected).abs() >= tolerance {
        return Err(proptest::test_runner::TestCaseError::fail(format!(
            "Gradient sum mismatch: {} vs {}",
            actual, expected
        )));
    }
    Ok(())
}

/// Checks if count matches expected value.
pub fn check_count_matches(
    actual: usize,
    expected: usize,
) -> Result<(), proptest::test_runner::TestCaseError> {
    if actual != expected {
        return Err(proptest::test_runner::TestCaseError::fail("Count mismatch"));
    }
    Ok(())
}

/// Checks if gradient is approximately zero.
pub fn check_gradient_is_zero(
    value: f64,
    tolerance: f64,
) -> Result<(), proptest::test_runner::TestCaseError> {
    if value.abs() >= tolerance {
        return Err(proptest::test_runner::TestCaseError::fail(format!(
            "Expected 0 gradient, got {}",
            value
        )));
    }
    Ok(())
}

/// Checks if count is zero.
pub fn check_count_is_zero(value: usize) -> Result<(), proptest::test_runner::TestCaseError> {
    if value != 0_usize {
        return Err(proptest::test_runner::TestCaseError::fail(
            "Expected 0 count",
        ));
    }
    Ok(())
}

/// Checks if subtraction result matches expected.
pub fn check_subtraction_result(
    sibling: f64,
    parent: f64,
    child: f64,
    bin: usize,
    tolerance: f64,
) -> Result<(), proptest::test_runner::TestCaseError> {
    let expected = parent - child;
    if (sibling - expected).abs() >= tolerance {
        return Err(proptest::test_runner::TestCaseError::fail(format!(
            "Gradient mismatch at bin {}: {} vs {}",
            bin, sibling, expected
        )));
    }
    Ok(())
}

// =========================================================================
// Error propagation wrapper functions
// =========================================================================

/// Wrapper to test gradient sum check propagation with pre-computed values.
pub fn test_gradient_sum_propagation(
    actual: f64,
    expected: f64,
) -> Result<(), proptest::test_runner::TestCaseError> {
    match check_gradient_sum_matches(actual, expected, 1e-9_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

/// Wrapper to test count check propagation with pre-computed values.
pub fn test_count_propagation(
    actual: usize,
    expected: usize,
) -> Result<(), proptest::test_runner::TestCaseError> {
    match check_count_matches(actual, expected) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

/// Wrapper to test gradient zero check propagation.
pub fn test_gradient_zero_propagation(
    value: f64,
) -> Result<(), proptest::test_runner::TestCaseError> {
    match check_gradient_is_zero(value, 1e-10_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

/// Wrapper to test count zero check propagation.
pub fn test_count_zero_propagation(
    value: usize,
) -> Result<(), proptest::test_runner::TestCaseError> {
    match check_count_is_zero(value) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

/// Wrapper to test subtraction check propagation.
pub fn test_subtraction_propagation(
    sibling: f64,
    parent: f64,
    child: f64,
    bin: usize,
) -> Result<(), proptest::test_runner::TestCaseError> {
    match check_subtraction_result(sibling, parent, child, bin, 1e-9_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

// =========================================================================
// Conversion and utility helpers
// =========================================================================

/// Helper for converting ClearGbmError to proptest TestCaseError.
pub fn to_test_error<T>(
    result: Result<T, ClearGbmError>,
    context: &str,
) -> Result<T, proptest::test_runner::TestCaseError> {
    match result {
        Ok(v) => Ok(v),
        Err(e) => Err(proptest::test_runner::TestCaseError::fail(format!(
            "{}: {}",
            context, e
        ))),
    }
}

/// Helper for subtract_histogram that converts errors.
pub fn subtract_for_test(
    parent: &HistogramBuffer,
    child: &HistogramBuffer,
) -> Result<HistogramBuffer, proptest::test_runner::TestCaseError> {
    match subtract_histogram(parent, child) {
        Ok(h) => Ok(h),
        Err(e) => Err(proptest::test_runner::TestCaseError::fail(format!(
            "subtract_histogram failed: {}",
            e
        ))),
    }
}

/// Helper to get gradient sum for a bin, converting errors.
pub fn get_gradient_for_test(
    hist: &HistogramBuffer,
    bin: usize,
) -> Result<f64, proptest::test_runner::TestCaseError> {
    match hist.gradient_sum(bin) {
        Ok(v) => Ok(v),
        Err(e) => Err(proptest::test_runner::TestCaseError::fail(format!(
            "gradient_sum failed: {}",
            e
        ))),
    }
}

/// Helper to get count for a bin, converting errors.
pub fn get_count_for_test(
    hist: &HistogramBuffer,
    bin: usize,
) -> Result<usize, proptest::test_runner::TestCaseError> {
    match hist.count(bin) {
        Ok(v) => Ok(v),
        Err(e) => Err(proptest::test_runner::TestCaseError::fail(format!(
            "count failed: {}",
            e
        ))),
    }
}

// =========================================================================
// Validation functions for proptest
// =========================================================================

/// Validates histogram gradient and count sums against expected values.
/// This function is independently testable for both Ok and Err paths.
pub fn validate_histogram_sums(
    hist: &HistogramBuffer,
    expected_grad: f64,
    expected_count: usize,
) -> Result<(), proptest::test_runner::TestCaseError> {
    let total_grad: f64 = hist.gradient_sums().iter().sum();
    match check_gradient_sum_matches(total_grad, expected_grad, 1e-9_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let total_count: usize = hist.counts().iter().sum();
    match check_count_matches(total_count, expected_count) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

/// Validates that all histogram values are zero.
/// This function is independently testable for both Ok and Err paths.
pub fn validate_all_zeros(
    hist: &HistogramBuffer,
) -> Result<(), proptest::test_runner::TestCaseError> {
    for bin in 0_usize..hist.n_bins() {
        let g = hist.gradient_sums()[bin];
        match check_gradient_is_zero(g, 1e-10_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }

        let c = hist.counts()[bin];
        match check_count_is_zero(c) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Validates that sibling = parent - child for all bins.
/// This function is independently testable for both Ok and Err paths.
pub fn validate_subtraction_correctness(
    sibling: &HistogramBuffer,
    parent: &HistogramBuffer,
    child: &HistogramBuffer,
) -> Result<(), proptest::test_runner::TestCaseError> {
    for bin in 0_usize..parent.n_bins() {
        let s = sibling.gradient_sums()[bin];
        let p = parent.gradient_sums()[bin];
        let c = child.gradient_sums()[bin];

        match check_subtraction_result(s, p, c, bin, 1e-9_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
