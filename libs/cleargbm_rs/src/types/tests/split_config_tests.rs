//! Tests for SplitConfig type.

use crate::error::ClearGbmError;
use crate::types::SplitConfig;

#[test]
fn test_split_config_new_valid() -> Result<(), ClearGbmError> {
    let c = match SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(c.min_samples_split(), 2_usize);
    assert_eq!(c.min_samples_leaf(), 1_usize);
    assert_eq!(c.max_bins(), 64_usize);
    assert!((c.reg_lambda() - 1.0_f64).abs() < f64::EPSILON);
    assert!(c.min_gain().abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn test_split_config_min_samples_split_too_small() -> Result<(), ClearGbmError> {
    let result = SplitConfig::new(1_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_samples_split"
    ));
    Ok(())
}

#[test]
fn test_split_config_min_samples_leaf_zero() -> Result<(), ClearGbmError> {
    let result = SplitConfig::new(2_usize, 0_usize, 64_usize, 1.0_f64, 0.0_f64);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_samples_leaf"
    ));
    Ok(())
}

#[test]
fn test_split_config_max_bins_too_small() -> Result<(), ClearGbmError> {
    let result = SplitConfig::new(2_usize, 1_usize, 1_usize, 1.0_f64, 0.0_f64);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "max_bins"
    ));
    Ok(())
}

#[test]
fn test_split_config_negative_reg_lambda() -> Result<(), ClearGbmError> {
    let result = SplitConfig::new(2_usize, 1_usize, 64_usize, -1.0_f64, 0.0_f64);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "reg_lambda"
    ));
    Ok(())
}

#[test]
fn test_split_config_negative_min_gain() -> Result<(), ClearGbmError> {
    let result = SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, -0.1_f64);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_gain"
    ));
    Ok(())
}

#[test]
fn test_split_config_clone() -> Result<(), ClearGbmError> {
    let c = match SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cloned = c.clone();
    assert_eq!(c, cloned);
    Ok(())
}

#[test]
fn test_split_config_serialize_deserialize() -> Result<(), ClearGbmError> {
    let c = match SplitConfig::new(10_usize, 5_usize, 128_usize, 0.5_f64, 0.01_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json_str = match serde_json::to_string(&c) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let parsed: SplitConfig = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(parsed, c);
    Ok(())
}
