//! Tests for helper functions to achieve branch coverage.

use super::helpers::{
    check_count_is_zero, check_count_matches, check_gradient_is_zero, check_gradient_sum_matches,
    check_subtraction_result, get_count_for_test, get_gradient_for_test, subtract_for_test,
    test_count_propagation, test_count_zero_propagation, test_gradient_sum_propagation,
    test_gradient_zero_propagation, test_subtraction_propagation, to_test_error,
    validate_all_zeros, validate_histogram_sums, validate_subtraction_correctness,
};
use crate::error::ClearGbmError;
use crate::histogram::subtract_histogram;
use crate::types::HistogramBuffer;

// =========================================================================
// Tests for assertion check functions (cover both pass and fail branches)
// =========================================================================

#[test]
fn test_check_gradient_sum_matches_pass() -> Result<(), ClearGbmError> {
    let result = check_gradient_sum_matches(1.0_f64, 1.0_f64, 1e-9_f64);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_check_gradient_sum_matches_fail() -> Result<(), ClearGbmError> {
    let result = check_gradient_sum_matches(1.0_f64, 100.0_f64, 1e-9_f64);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_check_count_matches_pass() -> Result<(), ClearGbmError> {
    let result = check_count_matches(5_usize, 5_usize);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_check_count_matches_fail() -> Result<(), ClearGbmError> {
    let result = check_count_matches(5_usize, 10_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_check_gradient_is_zero_pass() -> Result<(), ClearGbmError> {
    let result = check_gradient_is_zero(0.0_f64, 1e-10_f64);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_check_gradient_is_zero_fail() -> Result<(), ClearGbmError> {
    let result = check_gradient_is_zero(1.0_f64, 1e-10_f64);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_check_count_is_zero_pass() -> Result<(), ClearGbmError> {
    let result = check_count_is_zero(0_usize);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_check_count_is_zero_fail() -> Result<(), ClearGbmError> {
    let result = check_count_is_zero(5_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_check_subtraction_result_pass() -> Result<(), ClearGbmError> {
    // parent=10, child=3, sibling should be 7
    let result = check_subtraction_result(7.0_f64, 10.0_f64, 3.0_f64, 0_usize, 1e-9_f64);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_check_subtraction_result_fail() -> Result<(), ClearGbmError> {
    // parent=10, child=3, but sibling is wrong (should be 7, not 100)
    let result = check_subtraction_result(100.0_f64, 10.0_f64, 3.0_f64, 0_usize, 1e-9_f64);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests for error propagation wrappers
// =========================================================================

#[test]
fn test_gradient_sum_propagation_ok() -> Result<(), ClearGbmError> {
    let result = test_gradient_sum_propagation(1.0_f64, 1.0_f64);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_gradient_sum_propagation_err() -> Result<(), ClearGbmError> {
    let result = test_gradient_sum_propagation(1.0_f64, 999.0_f64);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_count_propagation_ok() -> Result<(), ClearGbmError> {
    let result = test_count_propagation(10_usize, 10_usize);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_count_propagation_err() -> Result<(), ClearGbmError> {
    let result = test_count_propagation(10_usize, 999_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_gradient_zero_propagation_ok() -> Result<(), ClearGbmError> {
    let result = test_gradient_zero_propagation(0.0_f64);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_gradient_zero_propagation_err() -> Result<(), ClearGbmError> {
    let result = test_gradient_zero_propagation(999.0_f64);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_count_zero_propagation_ok() -> Result<(), ClearGbmError> {
    let result = test_count_zero_propagation(0_usize);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_count_zero_propagation_err() -> Result<(), ClearGbmError> {
    let result = test_count_zero_propagation(999_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_subtraction_propagation_ok() -> Result<(), ClearGbmError> {
    // parent=10, child=3, sibling=7 (correct)
    let result = test_subtraction_propagation(7.0_f64, 10.0_f64, 3.0_f64, 0_usize);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_subtraction_propagation_err() -> Result<(), ClearGbmError> {
    // parent=10, child=3, sibling=999 (wrong)
    let result = test_subtraction_propagation(999.0_f64, 10.0_f64, 3.0_f64, 0_usize);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests for conversion helpers
// =========================================================================

#[test]
fn test_to_test_error_ok() -> Result<(), ClearGbmError> {
    fn inner() -> Result<i32, proptest::test_runner::TestCaseError> {
        let ok_result: Result<i32, ClearGbmError> = Ok(42_i32);
        to_test_error(ok_result, "test")
    }
    let result = inner();
    assert!(result.is_ok());
    assert_eq!(result.ok(), Some(42_i32));
    Ok(())
}

#[test]
fn test_to_test_error_err() -> Result<(), ClearGbmError> {
    fn inner() -> Result<i32, proptest::test_runner::TestCaseError> {
        let err_result: Result<i32, ClearGbmError> = Err(ClearGbmError::EmptyInput {
            context: "test".to_string(),
        });
        to_test_error(err_result, "context")
    }
    let result = inner();
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_subtract_for_test_ok() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = subtract_for_test(&hist, &hist);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_subtract_for_test_err() -> Result<(), ClearGbmError> {
    let a = HistogramBuffer::new(3_usize);
    let b = HistogramBuffer::new(5_usize);
    let result = subtract_for_test(&a, &b);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_get_gradient_for_test_ok() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = get_gradient_for_test(&hist, 0_usize);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_get_gradient_for_test_err() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = get_gradient_for_test(&hist, 100_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_get_count_for_test_ok() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = get_count_for_test(&hist, 0_usize);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_get_count_for_test_err() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = get_count_for_test(&hist, 100_usize);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests for validation functions
// =========================================================================

#[test]
fn test_validate_histogram_sums_ok() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(0_usize, 1.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(1_usize, 2.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    // total_grad = 3.0, count = 2
    let result = validate_histogram_sums(&hist, 3.0_f64, 2_usize);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_validate_histogram_sums_gradient_mismatch() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(0_usize, 1.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    // total_grad = 1.0, but we expect 999.0
    let result = validate_histogram_sums(&hist, 999.0_f64, 1_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_validate_histogram_sums_count_mismatch() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(0_usize, 1.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    // total_grad = 1.0, count = 1, but we expect count = 999
    let result = validate_histogram_sums(&hist, 1.0_f64, 999_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_validate_all_zeros_ok() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = validate_all_zeros(&hist);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_validate_all_zeros_gradient_nonzero() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(0_usize, 1.0_f64, 0.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    let result = validate_all_zeros(&hist);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_validate_all_zeros_count_nonzero() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    // accumulate adds to count even with zero gradient
    match hist.accumulate(0_usize, 0.0_f64, 0.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    let result = validate_all_zeros(&hist);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_validate_subtraction_correctness_ok() -> Result<(), ClearGbmError> {
    let mut parent = HistogramBuffer::new(3_usize);
    match parent.accumulate(0_usize, 10.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut child = HistogramBuffer::new(3_usize);
    match child.accumulate(0_usize, 3.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let sibling = match subtract_histogram(&parent, &child) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    let result = validate_subtraction_correctness(&sibling, &parent, &child);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_validate_subtraction_correctness_wrong_sibling() -> Result<(), ClearGbmError> {
    let mut parent = HistogramBuffer::new(3_usize);
    match parent.accumulate(0_usize, 10.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut child = HistogramBuffer::new(3_usize);
    match child.accumulate(0_usize, 3.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    // Create a wrong sibling (should be 7.0, but we use 999.0)
    let mut wrong_sibling = HistogramBuffer::new(3_usize);
    match wrong_sibling.accumulate(0_usize, 999.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let result = validate_subtraction_correctness(&wrong_sibling, &parent, &child);
    assert!(result.is_err());
    Ok(())
}
