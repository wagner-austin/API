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
