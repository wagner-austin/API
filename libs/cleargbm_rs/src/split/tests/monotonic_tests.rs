//! Tests for MonotonicConstraint type.

use crate::error::ClearGbmError;
use crate::split::MonotonicConstraint;

#[test]
fn test_monotonic_constraint_from_int_none() -> Result<(), ClearGbmError> {
    let constraint = match MonotonicConstraint::from_int(0_i32) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert!(constraint.is_none());
    assert_eq!(constraint, MonotonicConstraint::None);
    Ok(())
}

#[test]
fn test_monotonic_constraint_from_int_increasing() -> Result<(), ClearGbmError> {
    let constraint = match MonotonicConstraint::from_int(1_i32) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert!(!constraint.is_none());
    assert_eq!(constraint, MonotonicConstraint::Increasing);
    Ok(())
}

#[test]
fn test_monotonic_constraint_from_int_decreasing() -> Result<(), ClearGbmError> {
    let constraint = match MonotonicConstraint::from_int(-1_i32) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert!(!constraint.is_none());
    assert_eq!(constraint, MonotonicConstraint::Decreasing);
    Ok(())
}

#[test]
fn test_monotonic_constraint_from_int_invalid() -> Result<(), ClearGbmError> {
    let result = MonotonicConstraint::from_int(2_i32);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "monotonic_constraint"
    ));
    Ok(())
}
