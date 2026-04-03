//! Tests for vectorized sigmoid.

use crate::error::ClearGbmError;
use crate::losses::sigmoid_array;
use crate::predict::sigmoid;

#[test]
fn test_sigmoid_array_empty() -> Result<(), ClearGbmError> {
    let result = sigmoid_array(&[]);
    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_sigmoid_array_single() -> Result<(), ClearGbmError> {
    let result = sigmoid_array(&[0.0_f64]);
    assert_eq!(result.len(), 1_usize);
    assert!((result[0_usize] - 0.5_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_sigmoid_array_matches_scalar() -> Result<(), ClearGbmError> {
    // Vectorized result must exactly match scalar sigmoid for every element
    let inputs = [
        -100.0_f64, -10.0_f64, -1.0_f64, 0.0_f64, 1.0_f64, 10.0_f64, 100.0_f64,
    ];
    let array_result = sigmoid_array(&inputs);
    assert_eq!(array_result.len(), inputs.len());
    for (idx, &x) in inputs.iter().enumerate() {
        let scalar = sigmoid(x);
        assert!(
            (array_result[idx] - scalar).abs() < 1e-15_f64,
            "mismatch at index {idx}: array={}, scalar={scalar}",
            array_result[idx]
        );
    }
    Ok(())
}

#[test]
fn test_sigmoid_array_all_probabilities() -> Result<(), ClearGbmError> {
    // All outputs should be in [0, 1]
    // At extreme values (±500), f64 precision rounds to exactly 0.0 or 1.0
    let inputs = [-500.0_f64, -50.0_f64, 0.0_f64, 50.0_f64, 500.0_f64];
    let result = sigmoid_array(&inputs);
    for (idx, &p) in result.iter().enumerate() {
        assert!(
            (0.0_f64..=1.0_f64).contains(&p),
            "sigmoid_array[{idx}] = {p} not in [0, 1]"
        );
    }
    Ok(())
}

#[test]
fn test_sigmoid_array_moderate_inputs_strictly_interior() -> Result<(), ClearGbmError> {
    // Moderate inputs produce strictly interior probabilities (0, 1)
    let inputs = [-10.0_f64, -1.0_f64, 0.0_f64, 1.0_f64, 10.0_f64];
    let result = sigmoid_array(&inputs);
    for (idx, &p) in result.iter().enumerate() {
        assert!(
            p > 0.0_f64 && p < 1.0_f64,
            "sigmoid_array[{idx}] = {p} not in (0, 1)"
        );
    }
    Ok(())
}

#[test]
fn test_sigmoid_array_monotonic() -> Result<(), ClearGbmError> {
    // Sigmoid is monotonically increasing
    let inputs = [-10.0_f64, -5.0_f64, 0.0_f64, 5.0_f64, 10.0_f64];
    let result = sigmoid_array(&inputs);
    for i in 1_usize..result.len() {
        assert!(
            result[i] > result[i - 1_usize],
            "not monotonic at index {i}: {} <= {}",
            result[i],
            result[i - 1_usize]
        );
    }
    Ok(())
}

#[test]
fn test_sigmoid_array_symmetry() -> Result<(), ClearGbmError> {
    // sigmoid(x) + sigmoid(-x) = 1
    let inputs = [1.0_f64, 2.0_f64, 5.0_f64, 10.0_f64];
    let neg_inputs: Vec<f64> = inputs.iter().map(|x| -x).collect();
    let pos_result = sigmoid_array(&inputs);
    let neg_result = sigmoid_array(&neg_inputs);
    for i in 0_usize..inputs.len() {
        let sum = pos_result[i] + neg_result[i];
        assert!(
            (sum - 1.0_f64).abs() < 1e-14_f64,
            "symmetry failed at index {i}: {sum}"
        );
    }
    Ok(())
}
