//! Test utilities for serde serialization/deserialization error paths.
//!
//! This module provides custom serializers and deserializers that can be
//! configured to fail at specific points, enabling comprehensive testing
//! of error handling in serde implementations.

/// Bin-layout helpers for tests (column-major fixtures to row-major).
pub mod binning;

/// Failing deserializer for testing error paths.
pub mod deserializer;

/// Failing formatter for testing fmt::Result error paths.
pub mod formatter;

/// Failing serializer for testing error paths.
pub mod serializer;

#[cfg(test)]
mod tests;

// Re-export main types for convenient access.
pub use deserializer::{
    DeError, DuplicateFieldDeserializer, DuplicateFieldMapAccess, ErrorOnKeyDeserializer,
    ErrorOnKeyMapAccess, ErrorOnValueDeserializer, ErrorOnValueMapAccess, FailingDeserializer,
    FieldNameDeserializer, IntegerDeserializer, IntegerKeyDeserializer, IntegerKeyMapAccess,
    StringDeserializer, StructDuplicateFieldDeserializer, StructDuplicateFieldMapAccess,
    WrongTypeMode, WrongValueDeserializer, WrongValueMapAccess,
};
pub use formatter::{
    test_expecting_limited_write, test_expecting_write_failure, test_expecting_write_success,
    ExpectingWrapper, FailingWriter, LimitedWriter,
};
pub use serializer::{FailingSerializer, FailingSerializerStruct, SerError};
