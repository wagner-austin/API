//! Helpers for converting between numpy arrays and Rust types.
//!
//! All integer conversions use `try_from` (no `as` casts). Failures map to
//! [`ClearGbmError::IntegerConversion`] with descriptive context.
//!
//! The core conversion logic lives in [`try_convert_int`] and [`convert_int_slice`],
//! which are generic over input and output types. Specific functions like
//! [`i64_to_usize`] are thin wrappers for call-site clarity.

use crate::error::ClearGbmError;

/// Converts an integer of type `F` to type `T` using [`TryFrom`].
///
/// # Args
///
/// * `value` - The input value to convert.
/// * `context` - Description of what is being converted (for error messages).
///
/// # Returns
///
/// The value as type `T`.
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if the conversion fails
/// (e.g., value is negative or exceeds the target type's range).
pub(crate) fn try_convert_int<F, T>(value: F, context: &str) -> Result<T, ClearGbmError>
where
    T: TryFrom<F>,
    F: core::fmt::Display + Copy,
{
    match T::try_from(value) {
        Ok(v) => Ok(v),
        Err(_) => Err(ClearGbmError::IntegerConversion {
            context: format!(
                "{context}: cannot convert {value} to {}",
                core::any::type_name::<T>()
            ),
        }),
    }
}

/// Converts a slice of integers from type `F` to `Vec<T>` using [`TryFrom`].
///
/// # Args
///
/// * `slice` - The input slice to convert.
/// * `context` - Description of what is being converted (for error messages).
///
/// # Returns
///
/// A `Vec<T>` with all values converted.
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if any value fails conversion.
pub(crate) fn convert_int_slice<F, T>(slice: &[F], context: &str) -> Result<Vec<T>, ClearGbmError>
where
    T: TryFrom<F>,
    F: core::fmt::Display + Copy,
{
    let mut result = Vec::with_capacity(slice.len());
    for &val in slice {
        let converted = match try_convert_int(val, context) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        result.push(converted);
    }
    Ok(result)
}

/// Converts a Python `i64` to Rust `usize`.
///
/// # Args
///
/// * `value` - The `i64` value from Python.
/// * `context` - Description of what is being converted (for error messages).
///
/// # Returns
///
/// The value as `usize`.
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if `value` is negative
/// or exceeds `usize::MAX`.
pub(crate) fn i64_to_usize(value: i64, context: &str) -> Result<usize, ClearGbmError> {
    try_convert_int(value, context)
}

/// Converts a slice of `i64` values to `Vec<usize>`.
///
/// # Args
///
/// * `slice` - The input `i64` slice.
/// * `context` - Description of what is being converted (for error messages).
///
/// # Returns
///
/// A `Vec<usize>` with all values converted.
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if any value is negative
/// or exceeds `usize::MAX`.
pub(crate) fn i64_slice_to_usize_vec(
    slice: &[i64],
    context: &str,
) -> Result<Vec<usize>, ClearGbmError> {
    convert_int_slice(slice, context)
}

/// Converts a slice of `usize` values to `Vec<u64>`.
///
/// # Args
///
/// * `data` - The input `usize` slice.
/// * `context` - Description of what is being converted (for error messages).
///
/// # Returns
///
/// A `Vec<u64>` with all values converted.
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if any value exceeds `u64::MAX`.
pub(crate) fn usize_slice_to_u64_vec(
    data: &[usize],
    context: &str,
) -> Result<Vec<u64>, ClearGbmError> {
    convert_int_slice(data, context)
}

/// Converts a slice of `u64` values to `Vec<usize>`.
///
/// # Args
///
/// * `data` - The input `u64` slice.
/// * `context` - Description of what is being converted (for error messages).
///
/// # Returns
///
/// A `Vec<usize>` with all values converted.
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if any value exceeds `usize::MAX`.
pub(crate) fn u64_slice_to_usize_vec(
    data: &[u64],
    context: &str,
) -> Result<Vec<usize>, ClearGbmError> {
    convert_int_slice(data, context)
}

/// Converts an `i64` to `i32` for monotonic constraint conversion.
///
/// # Args
///
/// * `value` - The `i64` value from Python.
/// * `context` - Description of what is being converted (for error messages).
///
/// # Returns
///
/// The value as `i32`.
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if `value` exceeds `i32` range.
pub(crate) fn i64_to_i32(value: i64, context: &str) -> Result<i32, ClearGbmError> {
    try_convert_int(value, context)
}
