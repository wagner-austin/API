//! Tests for array helper conversion functions.

use crate::error::ClearGbmError;
use crate::pyo3_module::array_helpers::{i64_to_usize, try_convert_int};

// --- i64_to_usize ---

#[test]
fn test_i64_to_usize_zero() -> Result<(), ClearGbmError> {
    let result = match i64_to_usize(0_i64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 0_usize);
    Ok(())
}

#[test]
fn test_i64_to_usize_positive() -> Result<(), ClearGbmError> {
    let result = match i64_to_usize(42_i64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 42_usize);
    Ok(())
}

#[test]
fn test_i64_to_usize_negative_fails() -> Result<(), ClearGbmError> {
    let result = i64_to_usize(-1_i64, "test_context");
    assert!(result.is_err());
    if let Err(ClearGbmError::IntegerConversion { context }) = &result {
        assert!(context.contains("test_context"));
        assert!(context.contains("-1"));
    }
    Ok(())
}

#[test]
fn test_i64_to_usize_large_negative_fails() -> Result<(), ClearGbmError> {
    let result = i64_to_usize(i64::MIN, "large_negative");
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_i64_to_usize_max_positive() -> Result<(), ClearGbmError> {
    // i64::MAX should succeed on 64-bit platforms
    let result = i64_to_usize(i64::MAX, "max");
    assert!(result.is_ok());
    Ok(())
}

// --- usize → u64 (via try_convert_int) ---

#[test]
fn test_usize_to_u64_zero() -> Result<(), ClearGbmError> {
    let result = match try_convert_int::<usize, u64>(0_usize, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 0_u64);
    Ok(())
}

#[test]
fn test_usize_to_u64_positive() -> Result<(), ClearGbmError> {
    let result = match try_convert_int::<usize, u64>(123_usize, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 123_u64);
    Ok(())
}

#[test]
fn test_usize_to_u64_max() -> Result<(), ClearGbmError> {
    let result = match try_convert_int::<usize, u64>(usize::MAX, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    // On 64-bit platforms, usize::MAX == u64::MAX
    assert!(result > 0_u64);
    Ok(())
}

// --- u64 → usize (via try_convert_int) ---

#[test]
fn test_u64_to_usize_zero() -> Result<(), ClearGbmError> {
    let result = match try_convert_int::<u64, usize>(0_u64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 0_usize);
    Ok(())
}

#[test]
fn test_try_convert_int_overflow_u64_to_u32() -> Result<(), ClearGbmError> {
    let result = try_convert_int::<u64, u32>(u64::MAX, "overflow_test");
    assert!(result.is_err());
    if let Err(ClearGbmError::IntegerConversion { context }) = &result {
        assert!(context.contains("overflow_test"));
        assert!(context.contains("u32"));
    }
    Ok(())
}
