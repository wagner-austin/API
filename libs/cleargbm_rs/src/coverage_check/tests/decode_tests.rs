//! Tests for decoding the cargo-llvm-cov export.

use serde_json::{json, Value};

use crate::coverage_check::decode::{
    decode_coverage_export, decode_file_coverage, decode_segment, require_array, require_bool,
    require_member, require_object, require_string, require_u64, require_usize,
    SEGMENT_FIELD_COUNT,
};
use crate::coverage_check::types::CoverageCheckError;

/// Builds a well-formed segment array.
fn segment_json(line: usize, count: u64, has_count: bool) -> Value {
    json!([line, 1_i32, count, has_count, true, false])
}

/// Builds a well-formed export document around one file record.
fn export_json(filename: &str, segments: &[Value]) -> Value {
    json!({"data": [{"files": [{"filename": filename, "segments": segments}]}]})
}

// ── require_* validators ───────────────────────────────────────────

#[test]
fn test_require_object_accepts_object() -> Result<(), CoverageCheckError> {
    let value = json!({"a": 1_i32});
    let map = match require_object(&value, "root") {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(map.len(), 1_usize);
    Ok(())
}

#[test]
fn test_require_object_rejects_array() -> Result<(), CoverageCheckError> {
    let value = json!([]);
    match require_object(&value, "root") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "root".to_owned(),
                    expected: "an object".to_owned(),
                    got: "array".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_array_rejects_object() -> Result<(), CoverageCheckError> {
    let value = json!({});
    match require_array(&value, "data") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "data".to_owned(),
                    expected: "an array".to_owned(),
                    got: "object".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_array_accepts_array() -> Result<(), CoverageCheckError> {
    let value = json!([1_i32, 2_i32]);
    let items = match require_array(&value, "data") {
        Ok(i) => i,
        Err(e) => return Err(e),
    };
    assert_eq!(items.len(), 2_usize);
    Ok(())
}

#[test]
fn test_require_member_reports_missing_field() -> Result<(), CoverageCheckError> {
    let value = json!({"present": 1_i32});
    let map = match require_object(&value, "root") {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    match require_member(map, "absent") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldMissing {
                    field: "absent".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_member_returns_present_field() -> Result<(), CoverageCheckError> {
    let value = json!({"present": 7_i32});
    let map = match require_object(&value, "root") {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let found = match require_member(map, "present") {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    assert_eq!(found, &json!(7_i32));
    Ok(())
}

#[test]
fn test_require_string_accepts_and_rejects() -> Result<(), CoverageCheckError> {
    let text = json!("hello");
    let parsed = match require_string(&text, "filename") {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(parsed, "hello");

    let number = json!(3_i32);
    match require_string(&number, "filename") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "filename".to_owned(),
                    expected: "a string".to_owned(),
                    got: "number".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_bool_accepts_and_rejects() -> Result<(), CoverageCheckError> {
    let flag = json!(true);
    let parsed = match require_bool(&flag, "has_count") {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(parsed);

    let text = json!("true");
    match require_bool(&text, "has_count") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "has_count".to_owned(),
                    expected: "a boolean".to_owned(),
                    got: "string".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_string_names_boolean_type() -> Result<(), CoverageCheckError> {
    let value = json!(true);
    match require_string(&value, "filename") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "filename".to_owned(),
                    expected: "a string".to_owned(),
                    got: "boolean".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_usize_rejects_non_integer() -> Result<(), CoverageCheckError> {
    let value = json!(null);
    match require_usize(&value, "segment.line") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "segment.line".to_owned(),
                    expected: "an integer".to_owned(),
                    got: "null".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_usize_rejects_negative() -> Result<(), CoverageCheckError> {
    let value = json!(-1_i32);
    match require_usize(&value, "segment.line") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::NumberOutOfRange {
                    field: "segment.line".to_owned(),
                    value: -1_i64,
                    target: "usize".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_usize_accepts_zero() -> Result<(), CoverageCheckError> {
    let value = json!(0_i32);
    let parsed = match require_usize(&value, "segment.line") {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(parsed, 0_usize);
    Ok(())
}

#[test]
fn test_require_u64_rejects_non_integer_and_negative() -> Result<(), CoverageCheckError> {
    let text = json!("5");
    match require_u64(&text, "segment.count") {
        Ok(_) => return Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => assert_eq!(
            e,
            CoverageCheckError::FieldWrongType {
                field: "segment.count".to_owned(),
                expected: "an integer".to_owned(),
                got: "string".to_owned(),
            }
        ),
    }

    let negative = json!(-4_i32);
    match require_u64(&negative, "segment.count") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::NumberOutOfRange {
                    field: "segment.count".to_owned(),
                    value: -4_i64,
                    target: "u64".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_require_u64_accepts_positive() -> Result<(), CoverageCheckError> {
    let value = json!(9_i32);
    let parsed = match require_u64(&value, "segment.count") {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(parsed, 9_u64);
    Ok(())
}

// ── decode_segment ─────────────────────────────────────────────────

#[test]
fn test_decode_segment_reads_every_position() -> Result<(), CoverageCheckError> {
    let value = json!([31_i32, 5_i32, 11_i32, true, true, false]);
    let segment = match decode_segment(&value) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };
    assert_eq!(segment.line, 31_usize);
    assert_eq!(segment.column, 5_usize);
    assert_eq!(segment.count, 11_u64);
    assert!(segment.has_count);
    assert!(segment.is_region_entry);
    assert!(!segment.is_gap_region);
    Ok(())
}

#[test]
fn test_decode_segment_rejects_wrong_arity() -> Result<(), CoverageCheckError> {
    let value = json!([1_i32, 2_i32, 3_i32]);
    match decode_segment(&value) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::SegmentArityInvalid {
                    expected: SEGMENT_FIELD_COUNT,
                    got: 3_usize,
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_decode_segment_rejects_non_array() -> Result<(), CoverageCheckError> {
    let value = json!({"line": 1_i32});
    match decode_segment(&value) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "segment".to_owned(),
                    expected: "an array".to_owned(),
                    got: "object".to_owned(),
                }
            );
            Ok(())
        }
    }
}

// ── decode_file_coverage ───────────────────────────────────────────

#[test]
fn test_decode_file_coverage_reads_segments() -> Result<(), CoverageCheckError> {
    let value = json!({
        "filename": "src/a.rs",
        "segments": [segment_json(1_usize, 1_u64, true), segment_json(2_usize, 0_u64, true)],
    });
    let file = match decode_file_coverage(&value) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    assert_eq!(file.filename, "src/a.rs");
    assert_eq!(file.segments.len(), 2_usize);
    Ok(())
}

#[test]
fn test_decode_file_coverage_requires_filename() -> Result<(), CoverageCheckError> {
    let value = json!({"segments": []});
    match decode_file_coverage(&value) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldMissing {
                    field: "filename".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_decode_file_coverage_requires_segments() -> Result<(), CoverageCheckError> {
    let value = json!({"filename": "src/a.rs"});
    match decode_file_coverage(&value) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldMissing {
                    field: "segments".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_decode_file_coverage_rejects_non_object() -> Result<(), CoverageCheckError> {
    let value = json!("nope");
    match decode_file_coverage(&value) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "file".to_owned(),
                    expected: "an object".to_owned(),
                    got: "string".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_decode_file_coverage_propagates_segment_error() -> Result<(), CoverageCheckError> {
    let value = json!({"filename": "src/a.rs", "segments": [[1_i32, 2_i32]]});
    match decode_file_coverage(&value) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::SegmentArityInvalid {
                    expected: SEGMENT_FIELD_COUNT,
                    got: 2_usize,
                }
            );
            Ok(())
        }
    }
}

// ── decode_coverage_export ─────────────────────────────────────────

#[test]
fn test_decode_coverage_export_reads_files() -> Result<(), CoverageCheckError> {
    let document = export_json("src/a.rs", &[segment_json(1_usize, 1_u64, true)]);
    let export = match decode_coverage_export(&document.to_string(), "coverage.json") {
        Ok(x) => x,
        Err(e) => return Err(e),
    };
    assert_eq!(export.files.len(), 1_usize);
    assert_eq!(export.files[0].filename, "src/a.rs");
    Ok(())
}

#[test]
fn test_decode_coverage_export_rejects_invalid_json() -> Result<(), CoverageCheckError> {
    match decode_coverage_export("{not json", "coverage.json") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            match e {
                CoverageCheckError::ExportNotJson { path, reason } => {
                    assert_eq!(path, "coverage.json");
                    assert!(!reason.is_empty());
                }
                other => return Err(other),
            }
            Ok(())
        }
    }
}

#[test]
fn test_decode_coverage_export_rejects_non_object_root() -> Result<(), CoverageCheckError> {
    match decode_coverage_export("[]", "coverage.json") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "root".to_owned(),
                    expected: "an object".to_owned(),
                    got: "array".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_decode_coverage_export_requires_data() -> Result<(), CoverageCheckError> {
    match decode_coverage_export("{}", "coverage.json") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldMissing {
                    field: "data".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_decode_coverage_export_rejects_empty_data() -> Result<(), CoverageCheckError> {
    match decode_coverage_export(r#"{"data": []}"#, "coverage.json") {
        Ok(_) => Err(CoverageCheckError::FieldMissing {
            field: "unreachable".to_owned(),
        }),
        Err(e) => {
            assert_eq!(e, CoverageCheckError::ExportDataEmpty);
            Ok(())
        }
    }
}

#[test]
fn test_decode_coverage_export_rejects_non_object_entry() -> Result<(), CoverageCheckError> {
    match decode_coverage_export(r#"{"data": ["x"]}"#, "coverage.json") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "data[0]".to_owned(),
                    expected: "an object".to_owned(),
                    got: "string".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_decode_coverage_export_requires_files() -> Result<(), CoverageCheckError> {
    match decode_coverage_export(r#"{"data": [{}]}"#, "coverage.json") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldMissing {
                    field: "files".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_decode_coverage_export_propagates_file_error() -> Result<(), CoverageCheckError> {
    match decode_coverage_export(r#"{"data": [{"files": [1]}]}"#, "coverage.json") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::FieldWrongType {
                    field: "file".to_owned(),
                    expected: "an object".to_owned(),
                    got: "number".to_owned(),
                }
            );
            Ok(())
        }
    }
}
