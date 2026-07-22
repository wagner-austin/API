//! Tests for the SimpleRng pseudo-random number generator.

use crate::error::ClearGbmError;
use crate::training::rng::{u64_to_usize_via_u32, usize_to_u32, SimpleRng};

#[test]
fn test_determinism_same_seed() -> Result<(), ClearGbmError> {
    let mut rng1 = SimpleRng::new(42_u64);
    let mut rng2 = SimpleRng::new(42_u64);
    for _ in 0_usize..100_usize {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
    Ok(())
}

#[test]
fn test_different_seeds_differ() -> Result<(), ClearGbmError> {
    let mut rng1 = SimpleRng::new(42_u64);
    let mut rng2 = SimpleRng::new(99_u64);
    // Very unlikely to produce same sequence
    let mut any_differ = false;
    for _ in 0_usize..10_usize {
        if rng1.next_u64() != rng2.next_u64() {
            any_differ = true;
        }
    }
    assert!(any_differ);
    Ok(())
}

#[test]
fn test_zero_seed_mapped_to_nonzero() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(0_u64);
    // Should produce nonzero output (seed mapped to nonzero constant)
    let val = rng.next_u64();
    assert_ne!(val, 0_u64);
    Ok(())
}

#[test]
fn test_zero_seed_deterministic() -> Result<(), ClearGbmError> {
    let mut rng1 = SimpleRng::new(0_u64);
    let mut rng2 = SimpleRng::new(0_u64);
    for _ in 0_usize..10_usize {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
    Ok(())
}

#[test]
fn test_next_usize_below_in_range() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(123_u64);
    for _ in 0_usize..200_usize {
        let val = match rng.next_usize_below(10_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!(val < 10_usize);
    }
    Ok(())
}

#[test]
fn test_next_usize_below_single() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    // n=1: always returns 0
    for _ in 0_usize..20_usize {
        let val = match rng.next_usize_below(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(val, 0_usize);
    }
    Ok(())
}

#[test]
fn test_next_usize_below_zero_errors() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let result = rng.next_usize_below(0_usize);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for n=0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_shuffle_partial_full() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let mut indices: Vec<u32> = (0_u32..10_u32).collect();
    match rng.shuffle_partial(&mut indices, 10_usize) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    // All original values should still be present (permutation)
    let mut sorted = indices.clone();
    sorted.sort();
    let expected: Vec<u32> = (0_u32..10_u32).collect();
    assert_eq!(sorted, expected);
    Ok(())
}

#[test]
fn test_shuffle_partial_subset() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let mut indices: Vec<u32> = (0_u32..20_u32).collect();
    match rng.shuffle_partial(&mut indices, 5_usize) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    // First 5 elements are a random sample from 0..20
    for &val in &indices[0_usize..5_usize] {
        assert!(val < 20_u32);
    }
    // All 20 values still present
    let mut sorted = indices.clone();
    sorted.sort();
    let expected: Vec<u32> = (0_u32..20_u32).collect();
    assert_eq!(sorted, expected);
    Ok(())
}

#[test]
fn test_shuffle_partial_zero_take() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let mut indices: Vec<u32> = vec![1_u32, 2_u32, 3_u32];
    match rng.shuffle_partial(&mut indices, 0_usize) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    // No elements shuffled — original order preserved
    assert_eq!(indices, vec![1_u32, 2_u32, 3_u32]);
    Ok(())
}

#[test]
fn test_shuffle_partial_exceeds_length() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let mut indices: Vec<u32> = vec![1_u32, 2_u32];
    let result = rng.shuffle_partial(&mut indices, 5_usize);
    match result {
        Ok(()) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for n_take > len".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_take");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_clone_and_partial_eq() -> Result<(), ClearGbmError> {
    let rng1 = SimpleRng::new(42_u64);
    let rng2 = rng1.clone();
    assert_eq!(rng1, rng2);
    Ok(())
}

#[test]
fn test_debug_format() -> Result<(), ClearGbmError> {
    let rng = SimpleRng::new(42_u64);
    let debug_str = format!("{rng:?}");
    assert!(debug_str.contains("SimpleRng"));
    Ok(())
}

// --- usize_to_u32 tests ---

#[test]
fn test_usize_to_u32_valid() -> Result<(), ClearGbmError> {
    let result = match usize_to_u32(42_usize, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 42_u32);
    Ok(())
}

#[test]
fn test_usize_to_u32_max() -> Result<(), ClearGbmError> {
    let val = 0xFFFF_FFFF_usize;
    let result = match usize_to_u32(val, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, u32::MAX);
    Ok(())
}

#[test]
fn test_usize_to_u32_overflow() -> Result<(), ClearGbmError> {
    let val = 0xFFFF_FFFF_usize + 1_usize;
    match usize_to_u32(val, "test") {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for overflow".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

// --- u64_to_usize_via_u32 tests ---

#[test]
fn test_u64_to_usize_via_u32_valid() -> Result<(), ClearGbmError> {
    let result = match u64_to_usize_via_u32(42_u64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 42_usize);
    Ok(())
}

#[test]
fn test_u64_to_usize_via_u32_max() -> Result<(), ClearGbmError> {
    let result = match u64_to_usize_via_u32(u64::from(u32::MAX), "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 0xFFFF_FFFF_usize);
    Ok(())
}

#[test]
fn test_u64_to_usize_via_u32_overflow() -> Result<(), ClearGbmError> {
    let val = u64::from(u32::MAX) + 1_u64;
    match u64_to_usize_via_u32(val, "test") {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for overflow".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_u64_to_usize_via_u32_zero() -> Result<(), ClearGbmError> {
    let result = match u64_to_usize_via_u32(0_u64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 0_usize);
    Ok(())
}

// --- next_usize_below with large n ---

#[test]
fn test_next_usize_below_exceeds_u32_max() -> Result<(), ClearGbmError> {
    let mut rng = SimpleRng::new(42_u64);
    let big_n = 0xFFFF_FFFF_usize + 1_usize;
    match rng.next_usize_below(big_n) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for n > u32::MAX".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}
