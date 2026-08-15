//! Decode the cargo-llvm-cov JSON export into typed structures.
//!
//! Parsing walks a dynamic [`serde_json::Value`] and validates every field
//! explicitly, rather than deriving `Deserialize`. Derived implementations
//! generate code that cannot be audited or covered directly; walking the value
//! keeps each failure mode a named error with its own test.

use serde_json::Value;

use crate::coverage_check::types::{CoverageCheckError, CoverageExport, FileCoverage, Segment};

/// cargo-llvm-cov emits each segment as exactly six positional values.
pub const SEGMENT_FIELD_COUNT: usize = 6;

/// Names the JSON type of a value, for use in error messages.
fn type_name(value: &Value) -> String {
    let name = match *value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    };
    name.to_owned()
}

/// Reads a value as a JSON object.
///
/// # Errors
///
/// Returns [`CoverageCheckError::FieldWrongType`] when the value is not an object.
pub fn require_object<'a>(
    value: &'a Value,
    field: &str,
) -> Result<&'a serde_json::Map<String, Value>, CoverageCheckError> {
    match value.as_object() {
        Some(map) => Ok(map),
        None => Err(CoverageCheckError::FieldWrongType {
            field: field.to_owned(),
            expected: "an object".to_owned(),
            got: type_name(value),
        }),
    }
}

/// Reads a value as a JSON array.
///
/// # Errors
///
/// Returns [`CoverageCheckError::FieldWrongType`] when the value is not an array.
pub fn require_array<'a>(
    value: &'a Value,
    field: &str,
) -> Result<&'a Vec<Value>, CoverageCheckError> {
    match value.as_array() {
        Some(items) => Ok(items),
        None => Err(CoverageCheckError::FieldWrongType {
            field: field.to_owned(),
            expected: "an array".to_owned(),
            got: type_name(value),
        }),
    }
}

/// Reads a named member from a JSON object.
///
/// # Errors
///
/// Returns [`CoverageCheckError::FieldMissing`] when the key is absent.
pub fn require_member<'a>(
    map: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a Value, CoverageCheckError> {
    match map.get(field) {
        Some(value) => Ok(value),
        None => Err(CoverageCheckError::FieldMissing {
            field: field.to_owned(),
        }),
    }
}

/// Reads a value as a string.
///
/// # Errors
///
/// Returns [`CoverageCheckError::FieldWrongType`] when the value is not a string.
pub fn require_string(value: &Value, field: &str) -> Result<String, CoverageCheckError> {
    match value.as_str() {
        Some(text) => Ok(text.to_owned()),
        None => Err(CoverageCheckError::FieldWrongType {
            field: field.to_owned(),
            expected: "a string".to_owned(),
            got: type_name(value),
        }),
    }
}

/// Reads a value as a boolean.
///
/// # Errors
///
/// Returns [`CoverageCheckError::FieldWrongType`] when the value is not a boolean.
pub fn require_bool(value: &Value, field: &str) -> Result<bool, CoverageCheckError> {
    match value.as_bool() {
        Some(flag) => Ok(flag),
        None => Err(CoverageCheckError::FieldWrongType {
            field: field.to_owned(),
            expected: "a boolean".to_owned(),
            got: type_name(value),
        }),
    }
}

/// Reads a value as a signed integer.
///
/// # Errors
///
/// Returns [`CoverageCheckError::FieldWrongType`] when the value is not an
/// integer.
fn require_i64(value: &Value, field: &str) -> Result<i64, CoverageCheckError> {
    match value.as_i64() {
        Some(number) => Ok(number),
        None => Err(CoverageCheckError::FieldWrongType {
            field: field.to_owned(),
            expected: "an integer".to_owned(),
            got: type_name(value),
        }),
    }
}

/// Reads a value as a non-negative integer that fits `usize`.
///
/// # Errors
///
/// Returns [`CoverageCheckError::FieldWrongType`] when the value is not an
/// integer, or [`CoverageCheckError::NumberOutOfRange`] when it is negative or
/// exceeds `usize`.
pub fn require_usize(value: &Value, field: &str) -> Result<usize, CoverageCheckError> {
    let raw = propagate!(require_i64(value, field));
    match usize::try_from(raw) {
        Ok(converted) => Ok(converted),
        Err(_) => Err(CoverageCheckError::NumberOutOfRange {
            field: field.to_owned(),
            value: raw,
            target: "usize".to_owned(),
        }),
    }
}

/// Reads a value as a non-negative execution count.
///
/// # Errors
///
/// Returns [`CoverageCheckError::FieldWrongType`] when the value is not an
/// integer, or [`CoverageCheckError::NumberOutOfRange`] when it is negative.
pub fn require_u64(value: &Value, field: &str) -> Result<u64, CoverageCheckError> {
    let raw = propagate!(require_i64(value, field));
    match u64::try_from(raw) {
        Ok(converted) => Ok(converted),
        Err(_) => Err(CoverageCheckError::NumberOutOfRange {
            field: field.to_owned(),
            value: raw,
            target: "u64".to_owned(),
        }),
    }
}

/// Decodes one positional segment array into a named structure.
///
/// # Errors
///
/// Returns [`CoverageCheckError::SegmentArityInvalid`] when the array does not
/// hold exactly [`SEGMENT_FIELD_COUNT`] elements, or a field error when any
/// element has the wrong type.
pub fn decode_segment(value: &Value) -> Result<Segment, CoverageCheckError> {
    let items = propagate!(require_array(value, "segment"));
    if items.len() != SEGMENT_FIELD_COUNT {
        return Err(CoverageCheckError::SegmentArityInvalid {
            expected: SEGMENT_FIELD_COUNT,
            got: items.len(),
        });
    }
    let line = propagate!(require_usize(&items[0], "segment.line"));
    let column = propagate!(require_usize(&items[1], "segment.column"));
    let count = propagate!(require_u64(&items[2], "segment.count"));
    let has_count = propagate!(require_bool(&items[3], "segment.has_count"));
    let is_region_entry = propagate!(require_bool(&items[4], "segment.is_region_entry"));
    let is_gap_region = propagate!(require_bool(&items[5], "segment.is_gap_region"));
    Ok(Segment {
        line,
        column,
        count,
        has_count,
        is_region_entry,
        is_gap_region,
    })
}

/// Decodes the coverage record for one source file.
///
/// # Errors
///
/// Returns a field error when the record or any of its fields is malformed.
pub fn decode_file_coverage(value: &Value) -> Result<FileCoverage, CoverageCheckError> {
    let map = propagate!(require_object(value, "file"));
    let filename = propagate!(require_string(
        propagate!(require_member(map, "filename")),
        "filename"
    ));
    let raw_segments = propagate!(require_array(
        propagate!(require_member(map, "segments")),
        "segments"
    ));
    let mut segments = Vec::with_capacity(raw_segments.len());
    for raw in raw_segments {
        segments.push(propagate!(decode_segment(raw)));
    }
    Ok(FileCoverage { filename, segments })
}

/// Decodes a full cargo-llvm-cov export document.
///
/// # Errors
///
/// Returns [`CoverageCheckError::ExportNotJson`] when `raw` is not valid JSON,
/// [`CoverageCheckError::ExportDataEmpty`] when the document carries no data
/// entries, or a field error when any part of the document is malformed.
pub fn decode_coverage_export(raw: &str, path: &str) -> Result<CoverageExport, CoverageCheckError> {
    let document: Value = match serde_json::from_str(raw) {
        Ok(parsed) => parsed,
        Err(err) => {
            return Err(CoverageCheckError::ExportNotJson {
                path: path.to_owned(),
                reason: err.to_string(),
            })
        }
    };
    let root = propagate!(require_object(&document, "root"));
    let entries = propagate!(require_array(
        propagate!(require_member(root, "data")),
        "data"
    ));
    let first = match entries.first() {
        Some(entry) => entry,
        None => return Err(CoverageCheckError::ExportDataEmpty),
    };
    let entry = propagate!(require_object(first, "data[0]"));
    let raw_files = propagate!(require_array(
        propagate!(require_member(entry, "files")),
        "files"
    ));
    let mut files = Vec::with_capacity(raw_files.len());
    for raw_file in raw_files {
        files.push(propagate!(decode_file_coverage(raw_file)));
    }
    Ok(CoverageExport { files })
}
