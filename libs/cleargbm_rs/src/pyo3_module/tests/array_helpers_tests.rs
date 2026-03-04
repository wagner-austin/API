//! Tests for array helper conversion functions.

use crate::error::ClearGbmError;
use crate::pyo3_module::array_helpers::{
    convert_int_slice, i64_slice_to_usize_vec, i64_to_i32, i64_to_usize, try_convert_int,
    u64_slice_to_usize_vec, usize_slice_to_u64_vec,
};

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
fn test_u64_to_usize_positive() -> Result<(), ClearGbmError> {
    let result = match try_convert_int::<u64, usize>(99_u64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 99_usize);
    Ok(())
}

// --- i64_slice_to_usize_vec ---

#[test]
fn test_i64_slice_to_usize_vec_empty() -> Result<(), ClearGbmError> {
    let input: &[i64] = &[];
    let result = match i64_slice_to_usize_vec(input, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_i64_slice_to_usize_vec_valid() -> Result<(), ClearGbmError> {
    let input = [0_i64, 1_i64, 5_i64, 100_i64];
    let result = match i64_slice_to_usize_vec(&input, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, vec![0_usize, 1_usize, 5_usize, 100_usize]);
    Ok(())
}

#[test]
fn test_i64_slice_to_usize_vec_negative_fails() -> Result<(), ClearGbmError> {
    let input = [0_i64, 1_i64, -3_i64, 4_i64];
    let result = i64_slice_to_usize_vec(&input, "indices");
    assert!(result.is_err());
    if let Err(ClearGbmError::IntegerConversion { context }) = &result {
        assert!(context.contains("indices"));
        assert!(context.contains("-3"));
    }
    Ok(())
}

// --- usize_slice_to_u64_vec ---

#[test]
fn test_usize_slice_to_u64_vec_empty() -> Result<(), ClearGbmError> {
    let input: &[usize] = &[];
    let result = match usize_slice_to_u64_vec(input, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_usize_slice_to_u64_vec_valid() -> Result<(), ClearGbmError> {
    let input = [1_usize, 2_usize, 3_usize];
    let result = match usize_slice_to_u64_vec(&input, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, vec![1_u64, 2_u64, 3_u64]);
    Ok(())
}

// --- u64_slice_to_usize_vec ---

#[test]
fn test_u64_slice_to_usize_vec_empty() -> Result<(), ClearGbmError> {
    let input: &[u64] = &[];
    let result = match u64_slice_to_usize_vec(input, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_u64_slice_to_usize_vec_valid() -> Result<(), ClearGbmError> {
    let input = [10_u64, 20_u64, 30_u64];
    let result = match u64_slice_to_usize_vec(&input, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, vec![10_usize, 20_usize, 30_usize]);
    Ok(())
}

// --- i64_to_i32 ---

#[test]
fn test_i64_to_i32_zero() -> Result<(), ClearGbmError> {
    let result = match i64_to_i32(0_i64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 0_i32);
    Ok(())
}

#[test]
fn test_i64_to_i32_positive() -> Result<(), ClearGbmError> {
    let result = match i64_to_i32(1_i64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 1_i32);
    Ok(())
}

#[test]
fn test_i64_to_i32_negative() -> Result<(), ClearGbmError> {
    let result = match i64_to_i32(-1_i64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, -1_i32);
    Ok(())
}

#[test]
fn test_i64_to_i32_overflow_fails() -> Result<(), ClearGbmError> {
    let result = i64_to_i32(i64::MAX, "overflow");
    assert!(result.is_err());
    if let Err(ClearGbmError::IntegerConversion { context }) = &result {
        assert!(context.contains("overflow"));
    }
    Ok(())
}

#[test]
fn test_i64_to_i32_underflow_fails() -> Result<(), ClearGbmError> {
    let result = i64_to_i32(i64::MIN, "underflow");
    assert!(result.is_err());
    Ok(())
}

// --- try_convert_int generic Err path ---

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

#[test]
fn test_try_convert_int_ok_u32_to_u64() -> Result<(), ClearGbmError> {
    let result = match try_convert_int::<u32, u64>(42_u32, "ok_test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 42_u64);
    Ok(())
}

// --- convert_int_slice generic Err path ---

#[test]
fn test_convert_int_slice_overflow_u64_to_u32() -> Result<(), ClearGbmError> {
    let data = [1_u64, u64::MAX, 3_u64];
    let result = convert_int_slice::<u64, u32>(&data, "slice_overflow");
    assert!(result.is_err());
    if let Err(ClearGbmError::IntegerConversion { context }) = &result {
        assert!(context.contains("slice_overflow"));
    }
    Ok(())
}

#[test]
fn test_convert_int_slice_ok_u32_to_u64() -> Result<(), ClearGbmError> {
    let data = [1_u32, 2_u32, 3_u32];
    let result = match convert_int_slice::<u32, u64>(&data, "ok_test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, vec![1_u64, 2_u64, 3_u64]);
    Ok(())
}
