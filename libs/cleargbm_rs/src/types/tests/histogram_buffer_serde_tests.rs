//! Serde error path tests for HistogramBuffer.

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

#[test]
fn test_histogram_buffer_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":123,"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_missing_counts() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("counts"));
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_missing_n_bins() -> Result<(), ClearGbmError> {
    let json = r#"{"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("n_bins"));
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_missing_gradient_sums() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("gradient_sums"));
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_missing_hessian_sums() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("hessian_sums"));
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_unknown_field() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0],"unknown":true}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_all_fields() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[1.0,2.0,3.0],"hessian_sums":[0.5,1.0,1.5],"counts":[10,20,30]}"#;
    let hist: HistogramBuffer = match serde_json::from_str(json) {
        Ok(h) => h,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(hist.n_bins(), 3_usize);
    let g0 = match hist.gradient_sum(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((g0 - 1.0_f64).abs() < 1e-10_f64);
    let c2 = match hist.count(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(c2, 30_usize);
    Ok(())
}

// =========================================================================
// serde_json duplicate field tests (covers visit_map error paths)
// =========================================================================

#[test]
fn test_histogram_buffer_serde_json_duplicate_n_bins() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"n_bins":5,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("duplicate"));
    Ok(())
}

#[test]
fn test_histogram_buffer_serde_json_duplicate_gradient_sums() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"gradient_sums":[1.0,1.0,1.0],"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("duplicate"));
    Ok(())
}

#[test]
fn test_histogram_buffer_serde_json_duplicate_hessian_sums() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0],"hessian_sums":[1.0,1.0,1.0],"counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("duplicate"));
    Ok(())
}

#[test]
fn test_histogram_buffer_serde_json_duplicate_counts() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0],"counts":[1,1,1]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("duplicate"));
    Ok(())
}

#[test]
fn test_histogram_buffer_serde_json_invalid_n_bins_type() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":"not_a_number","gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_serde_json_invalid_hessian_sums_type() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":"not_an_array","counts":[0,0,0]}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_serde_json_invalid_counts_type() -> Result<(), ClearGbmError> {
    let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0],"counts":"not_an_array"}"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingDeserializer tests for HistogramBuffer
// =========================================================================

#[test]
fn test_histogram_buffer_deserialize_with_integer() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingDeserializer;
    use serde::Deserialize;

    let deser = FailingDeserializer::integer();
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_with_string_value() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingDeserializer;
    use serde::Deserialize;

    let deser = FailingDeserializer::string_value();
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_with_integer_key() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;

    let deser = IntegerKeyDeserializer;
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_wrong_value_n_bins() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("n_bins");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_wrong_value_gradient_sums() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("gradient_sums");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_wrong_value_hessian_sums() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("hessian_sums");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_wrong_value_counts() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("counts");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_serialize_roundtrip() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(4_usize);
    match hist.accumulate(0_usize, 1.5_f64, 2.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(1_usize, 3.0_f64, 4.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(2_usize, 0.0_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(3_usize, -1.0_f64, 0.5_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let json_str = match serde_json::to_string(&hist) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };

    let parsed: HistogramBuffer = match serde_json::from_str(&json_str) {
        Ok(h) => h,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };

    assert_eq!(parsed.n_bins(), hist.n_bins());
    for i in 0_usize..4_usize {
        let orig_g = match hist.gradient_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let parsed_g = match parsed.gradient_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((orig_g - parsed_g).abs() < 1e-10_f64);

        let orig_h = match hist.hessian_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let parsed_h = match parsed.hessian_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((orig_h - parsed_h).abs() < 1e-10_f64);

        let orig_c = match hist.count(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let parsed_c = match parsed.count(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(orig_c, parsed_c);
    }
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_error_on_key() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnKeyDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnKeyDeserializer;
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_error_on_value_n_bins() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("n_bins");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_error_on_value_gradient_sums() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("gradient_sums");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_error_on_value_hessian_sums() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("hessian_sums");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_error_on_value_counts() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("counts");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingSerializer tests for HistogramBuffer
// =========================================================================

#[test]
fn test_histogram_buffer_deserialize_duplicate_field() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;

    let deser = DuplicateFieldDeserializer::new("n_bins");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_struct_duplicate_n_bins() -> Result<(), ClearGbmError> {
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;

    let deser = StructDuplicateFieldDeserializer::new("n_bins");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_struct_duplicate_gradient_sums() -> Result<(), ClearGbmError> {
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;

    let deser = StructDuplicateFieldDeserializer::new("gradient_sums");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_struct_duplicate_hessian_sums() -> Result<(), ClearGbmError> {
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;

    let deser = StructDuplicateFieldDeserializer::new("hessian_sums");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_struct_duplicate_counts() -> Result<(), ClearGbmError> {
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;

    let deser = StructDuplicateFieldDeserializer::new("counts");
    let result = HistogramBuffer::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let hist = HistogramBuffer::new(2_usize);
    let mut ser = FailingSerializer::fail_on_struct();
    let result = hist.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let hist = HistogramBuffer::new(2_usize);
    let mut ser = FailingSerializer::fail_on_end();
    let result = hist.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let hist = HistogramBuffer::new(2_usize);
    // HistogramBuffer has 4 fields: n_bins, gradient_sums, hessian_sums, counts
    for fail_at in 0_usize..4_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = hist.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

// =========================================================================
// Expecting error path tests using FailingWriter
// =========================================================================

#[test]
fn test_histogram_buffer_field_visitor_expecting_write_failure() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_write_failure;
    use crate::types::serde_impl::histogram_buffer::HistogramBufferFieldVisitor;

    let visitor = HistogramBufferFieldVisitor;
    let result = test_expecting_write_failure(&visitor);
    match result {
        Ok(()) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "expected fmt error to propagate".to_string(),
        }),
        Err(_) => Ok(()),
    }
}

#[test]
fn test_histogram_buffer_field_visitor_expecting_write_success() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_write_success;
    use crate::types::serde_impl::histogram_buffer::HistogramBufferFieldVisitor;

    let visitor = HistogramBufferFieldVisitor;
    // "field identifier" is 16 chars, so 50 should be plenty
    let result = test_expecting_write_success(&visitor, 50_usize);
    match result {
        Ok(()) => Ok(()),
        Err(_) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "expected success with sufficient buffer".to_string(),
        }),
    }
}

#[test]
fn test_histogram_buffer_field_visitor_expecting_limited_write() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_limited_write;
    use crate::types::serde_impl::histogram_buffer::HistogramBufferFieldVisitor;

    let visitor = HistogramBufferFieldVisitor;
    // "field identifier" is 16 chars, so limit of 5 should fail
    let result = test_expecting_limited_write(&visitor, 5_usize);
    match result {
        Ok(()) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "expected failure with insufficient buffer".to_string(),
        }),
        Err(_) => Ok(()),
    }
}

#[test]
fn test_histogram_buffer_visitor_expecting_write_failure() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_write_failure;
    use crate::types::serde_impl::histogram_buffer::HistogramBufferVisitor;

    let visitor = HistogramBufferVisitor;
    let result = test_expecting_write_failure(&visitor);
    match result {
        Ok(()) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "expected fmt error to propagate".to_string(),
        }),
        Err(_) => Ok(()),
    }
}

#[test]
fn test_histogram_buffer_visitor_expecting_write_success() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_write_success;
    use crate::types::serde_impl::histogram_buffer::HistogramBufferVisitor;

    let visitor = HistogramBufferVisitor;
    // "struct HistogramBuffer" is 22 chars, so 50 should be plenty
    let result = test_expecting_write_success(&visitor, 50_usize);
    match result {
        Ok(()) => Ok(()),
        Err(_) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "expected success with sufficient buffer".to_string(),
        }),
    }
}

#[test]
fn test_histogram_buffer_visitor_expecting_limited_write() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_limited_write;
    use crate::types::serde_impl::histogram_buffer::HistogramBufferVisitor;

    let visitor = HistogramBufferVisitor;
    // "struct HistogramBuffer" is 22 chars, so limit of 10 should fail
    let result = test_expecting_limited_write(&visitor, 10_usize);
    match result {
        Ok(()) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "expected failure with insufficient buffer".to_string(),
        }),
        Err(_) => Ok(()),
    }
}

#[test]
fn test_histogram_buffer_field_visitor_expecting_exact_limit() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_limited_write;
    use crate::types::serde_impl::histogram_buffer::HistogramBufferFieldVisitor;

    let visitor = HistogramBufferFieldVisitor;
    // "field identifier" is exactly 16 chars
    let result = test_expecting_limited_write(&visitor, 16_usize);
    match result {
        Ok(()) => Ok(()),
        Err(_) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "expected success with exact buffer size".to_string(),
        }),
    }
}

#[test]
fn test_histogram_buffer_visitor_expecting_exact_limit() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_limited_write;
    use crate::types::serde_impl::histogram_buffer::HistogramBufferVisitor;

    let visitor = HistogramBufferVisitor;
    // "struct HistogramBuffer" is exactly 22 chars
    let result = test_expecting_limited_write(&visitor, 22_usize);
    match result {
        Ok(()) => Ok(()),
        Err(_) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "expected success with exact buffer size".to_string(),
        }),
    }
}
