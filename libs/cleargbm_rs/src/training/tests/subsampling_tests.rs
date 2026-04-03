//! Tests for row subsampling and f64_to_usize_checked.

use crate::error::ClearGbmError;
use crate::training::rng::SimpleRng;
use crate::training::subsampling::{f64_to_usize_checked, get_sample_indices};

// --- f64_to_usize_checked tests ---

#[test]
fn test_f64_to_usize_zero() -> Result<(), ClearGbmError> {
    let result = match f64_to_usize_checked(0.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 0_usize);
    Ok(())
}

#[test]
fn test_f64_to_usize_positive_integers() -> Result<(), ClearGbmError> {
    let result = match f64_to_usize_checked(1.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 1_usize);
    let result = match f64_to_usize_checked(42.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 42_usize);
    let result = match f64_to_usize_checked(255.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 255_usize);
    Ok(())
}

#[test]
fn test_f64_to_usize_powers_of_two() -> Result<(), ClearGbmError> {
    let result = match f64_to_usize_checked(2.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 2_usize);
    let result = match f64_to_usize_checked(1024.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 1024_usize);
    let result = match f64_to_usize_checked(65536.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 65536_usize);
    Ok(())
}

#[test]
fn test_f64_to_usize_large_value() -> Result<(), ClearGbmError> {
    let max_u32 = f64::from(u32::MAX);
    let result = match f64_to_usize_checked(max_u32, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 4_294_967_295_usize);
    Ok(())
}

#[test]
fn test_f64_to_usize_odd_values() -> Result<(), ClearGbmError> {
    let result = match f64_to_usize_checked(7.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 7_usize);
    let result = match f64_to_usize_checked(123.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 123_usize);
    Ok(())
}

#[test]
fn test_f64_to_usize_negative() -> Result<(), ClearGbmError> {
    let result = f64_to_usize_checked(-1.0_f64, "test");
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for negative".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_f64_to_usize_non_integer() -> Result<(), ClearGbmError> {
    let result = f64_to_usize_checked(1.5_f64, "test");
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for non-integer".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_f64_to_usize_nan() -> Result<(), ClearGbmError> {
    let result = f64_to_usize_checked(f64::NAN, "test");
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for NaN".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_f64_to_usize_infinity() -> Result<(), ClearGbmError> {
    let result = f64_to_usize_checked(f64::INFINITY, "test");
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for infinity".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_f64_to_usize_exceeds_u32_max() -> Result<(), ClearGbmError> {
    let val = f64::from(u32::MAX) + 1.0_f64;
    let result = f64_to_usize_checked(val, "test");
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for exceeding u32::MAX".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

// --- get_sample_indices tests ---

#[test]
fn test_full_subsample() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let indices = match get_sample_indices(10_usize, 1.0_f64, &mut rng) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(indices.len(), 10_usize);
    let expected: Vec<usize> = (0_usize..10_usize).collect();
    assert_eq!(indices, expected);
    Ok(())
}

#[test]
fn test_partial_subsample() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let indices = match get_sample_indices(100_usize, 0.5_f64, &mut rng) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(indices.len(), 50_usize);
    // All indices in valid range
    for &idx in &indices {
        assert!(idx < 100_usize);
    }
    // No duplicates
    let mut sorted = indices.clone();
    sorted.sort();
    for i in 1_usize..sorted.len() {
        assert_ne!(sorted[i], sorted[i - 1_usize]);
    }
    Ok(())
}

#[test]
fn test_subsample_minimum_one() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    // Very small subsample fraction → at least 1 sample
    let indices = match get_sample_indices(100_usize, 0.001_f64, &mut rng) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(indices.len(), 1_usize);
    assert!(indices[0_usize] < 100_usize);
    Ok(())
}

#[test]
fn test_subsample_deterministic() -> Result<(), ClearGbmError> {
    let mut rng1 = SimpleRng::new(42_u64);
    let mut rng2 = SimpleRng::new(42_u64);
    let indices1 = match get_sample_indices(50_usize, 0.6_f64, &mut rng1) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let indices2 = match get_sample_indices(50_usize, 0.6_f64, &mut rng2) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(indices1, indices2);
    Ok(())
}

#[test]
fn test_n_samples_zero_error() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let result = get_sample_indices(0_usize, 1.0_f64, &mut rng);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for n_samples=0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_samples");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_subsample_above_one() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    // subsample > 1.0 treated as full subsample
    let indices = match get_sample_indices(5_usize, 2.0_f64, &mut rng) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(indices.len(), 5_usize);
    let expected: Vec<usize> = (0_usize..5_usize).collect();
    assert_eq!(indices, expected);
    Ok(())
}

#[test]
fn test_subsample_single_sample() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let indices = match get_sample_indices(1_usize, 0.5_f64, &mut rng) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    // max(1, floor(1 * 0.5)) = max(1, 0) = 1
    assert_eq!(indices.len(), 1_usize);
    assert_eq!(indices[0_usize], 0_usize);
    Ok(())
}

#[test]
fn test_n_samples_exceeds_u32_max() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let big_n = 0xFFFF_FFFF_usize + 1_usize;
    let result = get_sample_indices(big_n, 0.5_f64, &mut rng);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for n_samples > u32::MAX".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}
