//! Failing formatter utilities for testing fmt::Result error paths.
//!
//! This module provides utilities to test code paths where `fmt::Write`
//! operations fail, which is needed for complete coverage of `expecting()`
//! methods in serde `Visitor` implementations.

use core::fmt::{self, Write};

/// A writer that always fails on write operations.
///
/// Used to test error handling in `fmt::Display` and `fmt::Debug`
/// implementations, as well as serde `Visitor::expecting()` methods.
pub struct FailingWriter;

impl Write for FailingWriter {
    fn write_str(&mut self, _s: &str) -> fmt::Result {
        Err(fmt::Error)
    }
}

/// A writer that fails after a specified number of bytes.
///
/// Useful for testing partial write scenarios.
pub struct LimitedWriter {
    /// Remaining bytes that can be written before failure.
    remaining: usize,
}

impl LimitedWriter {
    /// Creates a new limited writer that fails after `limit` bytes.
    #[must_use]
    pub const fn new(limit: usize) -> Self {
        Self { remaining: limit }
    }

    /// Returns the remaining capacity.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.remaining
    }
}

impl Write for LimitedWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let len = s.len();
        if len > self.remaining {
            Err(fmt::Error)
        } else {
            self.remaining -= len;
            Ok(())
        }
    }
}

/// Wrapper to test a serde Visitor's expecting() method with a custom writer.
///
/// This allows testing the fmt::Result error path in expecting() implementations.
pub struct ExpectingWrapper<'a, V> {
    visitor: &'a V,
}

impl<'a, V> ExpectingWrapper<'a, V> {
    /// Creates a new wrapper around a visitor reference.
    #[must_use]
    pub const fn new(visitor: &'a V) -> Self {
        Self { visitor }
    }
}

impl<'a, 'de, V> fmt::Display for ExpectingWrapper<'a, V>
where
    V: serde::de::Visitor<'de>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.visitor.expecting(f) {
            Ok(()) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

/// Tests that a visitor's expecting() method handles write failures correctly.
///
/// Returns `Err(fmt::Error)` if the expecting() method propagates the write error,
/// or `Ok(())` if somehow it doesn't (which would indicate a bug).
///
/// # Errors
///
/// Returns `fmt::Error` when the visitor correctly propagates write failures.
pub fn test_expecting_write_failure<'de, V>(visitor: &V) -> fmt::Result
where
    V: serde::de::Visitor<'de>,
{
    let wrapper = ExpectingWrapper::new(visitor);
    let mut writer = FailingWriter;
    match write!(writer, "{}", wrapper) {
        Ok(()) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Tests that a visitor's expecting() method handles partial write failures.
///
/// The writer will fail after `limit` bytes have been written.
///
/// # Errors
///
/// Returns `fmt::Error` when the write exceeds the specified limit.
pub fn test_expecting_limited_write<'de, V>(visitor: &V, limit: usize) -> fmt::Result
where
    V: serde::de::Visitor<'de>,
{
    let wrapper = ExpectingWrapper::new(visitor);
    let mut writer = LimitedWriter::new(limit);
    match write!(writer, "{}", wrapper) {
        Ok(()) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Tests that a visitor's expecting() succeeds with sufficient buffer.
///
/// # Errors
///
/// Returns `fmt::Error` if the buffer is insufficient for the expecting message.
pub fn test_expecting_write_success<'de, V>(visitor: &V, buffer_size: usize) -> fmt::Result
where
    V: serde::de::Visitor<'de>,
{
    let wrapper = ExpectingWrapper::new(visitor);
    let mut writer = LimitedWriter::new(buffer_size);
    match write!(writer, "{}", wrapper) {
        Ok(()) => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ClearGbmError;

    struct TestVisitor;

    impl<'de> serde::de::Visitor<'de> for TestVisitor {
        type Value = ();

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match formatter.write_str("test value") {
                Ok(()) => Ok(()),
                Err(e) => Err(e),
            }
        }
    }

    #[test]
    fn test_failing_writer_always_fails() -> Result<(), ClearGbmError> {
        let mut writer = FailingWriter;
        let result = writer.write_str("anything");
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_failing_writer_empty_string_fails() -> Result<(), ClearGbmError> {
        let mut writer = FailingWriter;
        let result = writer.write_str("");
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_limited_writer_succeeds_within_limit() -> Result<(), ClearGbmError> {
        let mut writer = LimitedWriter::new(10);
        let result = writer.write_str("hello");
        assert!(result.is_ok());
        assert_eq!(writer.remaining(), 5_usize);
        Ok(())
    }

    #[test]
    fn test_limited_writer_fails_beyond_limit() -> Result<(), ClearGbmError> {
        let mut writer = LimitedWriter::new(3);
        let result = writer.write_str("hello");
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_limited_writer_exact_limit_succeeds() -> Result<(), ClearGbmError> {
        let mut writer = LimitedWriter::new(5);
        let result = writer.write_str("hello");
        assert!(result.is_ok());
        assert_eq!(writer.remaining(), 0_usize);
        Ok(())
    }

    #[test]
    fn test_limited_writer_zero_limit_empty_string() -> Result<(), ClearGbmError> {
        let mut writer = LimitedWriter::new(0);
        let result = writer.write_str("");
        assert!(result.is_ok());
        assert_eq!(writer.remaining(), 0_usize);
        Ok(())
    }

    #[test]
    fn test_limited_writer_zero_limit_nonempty_string() -> Result<(), ClearGbmError> {
        let mut writer = LimitedWriter::new(0);
        let result = writer.write_str("x");
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_expecting_wrapper_displays_correctly() -> Result<(), ClearGbmError> {
        let visitor = TestVisitor;
        let wrapper = ExpectingWrapper::new(&visitor);
        let result = format!("{}", wrapper);
        assert_eq!(result, "test value");
        Ok(())
    }

    #[test]
    fn test_expecting_write_failure_propagates_error() -> Result<(), ClearGbmError> {
        let visitor = TestVisitor;
        let result = test_expecting_write_failure(&visitor);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_expecting_limited_write_zero_fails() -> Result<(), ClearGbmError> {
        let visitor = TestVisitor;
        let result = test_expecting_limited_write(&visitor, 0_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_expecting_limited_write_partial_fails() -> Result<(), ClearGbmError> {
        let visitor = TestVisitor;
        // "test value" is 10 chars, so limit of 5 should fail
        let result = test_expecting_limited_write(&visitor, 5_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_expecting_limited_write_sufficient_succeeds() -> Result<(), ClearGbmError> {
        let visitor = TestVisitor;
        // "test value" is 10 chars, so limit of 20 should succeed
        let result = test_expecting_limited_write(&visitor, 20_usize);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_expecting_write_success_sufficient_buffer() -> Result<(), ClearGbmError> {
        let visitor = TestVisitor;
        let result = test_expecting_write_success(&visitor, 100_usize);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_expecting_write_success_exact_buffer() -> Result<(), ClearGbmError> {
        let visitor = TestVisitor;
        // "test value" is 10 chars
        let result = test_expecting_write_success(&visitor, 10_usize);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_expecting_write_success_insufficient_buffer() -> Result<(), ClearGbmError> {
        let visitor = TestVisitor;
        let result = test_expecting_write_success(&visitor, 5_usize);
        assert!(result.is_err());
        Ok(())
    }

    /// A visitor that can be configured to succeed or fail in expecting().
    struct ConfigurableVisitor {
        should_fail: bool,
    }

    impl ConfigurableVisitor {
        const fn new(should_fail: bool) -> Self {
            Self { should_fail }
        }
    }

    impl<'de> serde::de::Visitor<'de> for ConfigurableVisitor {
        type Value = ();

        fn expecting(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.should_fail {
                Err(fmt::Error)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn test_expecting_write_failure_configurable_both_branches() -> Result<(), ClearGbmError> {
        // Ok branch: visitor writes nothing, FailingWriter doesn't fail
        let visitor_ok = ConfigurableVisitor::new(false);
        let result_ok = test_expecting_write_failure(&visitor_ok);
        assert!(result_ok.is_ok());

        // Err branch: visitor returns error
        let visitor_err = ConfigurableVisitor::new(true);
        let result_err = test_expecting_write_failure(&visitor_err);
        assert!(result_err.is_err());
        Ok(())
    }

    #[test]
    fn test_expecting_wrapper_visitor_both_branches() -> Result<(), ClearGbmError> {
        // Test the Ok branch
        let visitor_ok = ConfigurableVisitor::new(false);
        let wrapper_ok = ExpectingWrapper::new(&visitor_ok);
        let mut buf = String::new();
        let result_ok = core::fmt::write(&mut buf, format_args!("{}", wrapper_ok));
        assert!(result_ok.is_ok());

        // Test the Err branch with the same visitor type
        let visitor_err = ConfigurableVisitor::new(true);
        let wrapper_err = ExpectingWrapper::new(&visitor_err);
        let mut buf2 = String::new();
        let result_err = core::fmt::write(&mut buf2, format_args!("{}", wrapper_err));
        assert!(result_err.is_err());
        Ok(())
    }
}
