//! Unit tests for the softmax cross-entropy pieces.

use crate::error::ClearGbmError;
use crate::losses::multiclass::{fill_row_grad_hess, softmax_row_into, MulticlassGradBuffers};
use crate::losses::{multiclass_initial_predictions, multiclass_log_loss};

#[test]
fn test_softmax_row_gathers_class_major_and_normalizes() {
    // Two rows, three classes, class-major: class k's block is contiguous.
    // Row 1 scores are [0, ln 2, ln 4]: softmax = [1/7, 2/7, 4/7].
    let scores = vec![
        0.0_f64,
        0.0_f64, // class 0
        0.0_f64,
        2.0_f64.ln(), // class 1
        0.0_f64,
        4.0_f64.ln(), // class 2
    ];
    let mut out = vec![0.0_f64; 3];
    softmax_row_into(&scores, 2_usize, 1_usize, &mut out);
    assert!((out[0] - 1.0_f64 / 7.0_f64).abs() < 1e-12_f64, "{out:?}");
    assert!((out[1] - 2.0_f64 / 7.0_f64).abs() < 1e-12_f64, "{out:?}");
    assert!((out[2] - 4.0_f64 / 7.0_f64).abs() < 1e-12_f64, "{out:?}");
    let sum: f64 = out.iter().sum();
    assert!((sum - 1.0_f64).abs() < 1e-12_f64);
}

#[test]
fn test_softmax_is_shift_stable_at_large_scores() {
    // Max subtraction keeps huge scores finite.
    let scores = vec![1000.0_f64, 1001.0_f64, 999.0_f64];
    let mut out = vec![0.0_f64; 3];
    softmax_row_into(&scores, 1_usize, 0_usize, &mut out);
    assert!(out.iter().all(|p| p.is_finite() && *p > 0.0_f64), "{out:?}");
    let sum: f64 = out.iter().sum();
    assert!((sum - 1.0_f64).abs() < 1e-12_f64);
}

#[test]
fn test_initial_predictions_are_log_priors() -> Result<(), ClearGbmError> {
    // Labels [0, 0, 1, 2]: priors [1/2, 1/4, 1/4].
    let y = [0_u32, 0_u32, 1_u32, 2_u32];
    let bases = propagate!(multiclass_initial_predictions(&y, 3_usize, None));
    assert!((bases[0] - 0.5_f64.ln()).abs() < 1e-12_f64);
    assert!((bases[1] - 0.25_f64.ln()).abs() < 1e-12_f64);
    assert!((bases[2] - 0.25_f64.ln()).abs() < 1e-12_f64);
    Ok(())
}

#[test]
fn test_initial_predictions_weight_the_priors() -> Result<(), ClearGbmError> {
    // Weight 3 on the single class-1 row: priors [1/4, 3/4].
    let y = [0_u32, 1_u32];
    let w = [1.0_f64, 3.0_f64];
    let bases = propagate!(multiclass_initial_predictions(&y, 2_usize, Some(&w)));
    assert!((bases[0] - 0.25_f64.ln()).abs() < 1e-12_f64);
    assert!((bases[1] - 0.75_f64.ln()).abs() < 1e-12_f64);
    Ok(())
}

#[test]
fn test_initial_predictions_floor_an_empty_class() -> Result<(), ClearGbmError> {
    // Class 2 has no rows: its prior floors at the epsilon rather than
    // producing -inf.
    let y = [0_u32, 1_u32];
    let bases = propagate!(multiclass_initial_predictions(&y, 3_usize, None));
    assert!(bases[2].is_finite());
    assert!(bases[2] < bases[0]);
    Ok(())
}

#[test]
fn test_initial_predictions_reject_empty_labels() -> Result<(), ClearGbmError> {
    match multiclass_initial_predictions(&[], 3_usize, None) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "empty labels must be rejected".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_log_loss_matches_a_hand_computation() -> Result<(), ClearGbmError> {
    // One row, two classes, scores [0, ln 3]: p = [1/4, 3/4]. Label 1:
    // loss = -ln(3/4).
    let y = [1_u32];
    let scores = vec![0.0_f64, 3.0_f64.ln()];
    let loss = propagate!(multiclass_log_loss(&y, &scores, 2_usize, None));
    assert!((loss - -(0.75_f64.ln())).abs() < 1e-12_f64);
    Ok(())
}

#[test]
fn test_log_loss_weights_average_per_row() -> Result<(), ClearGbmError> {
    // Two identical rows with weights [1, 3]: the weighted mean equals the
    // unweighted per-row loss (both rows contribute the same value).
    let y = [0_u32, 0_u32];
    let scores = vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64];
    let unweighted = propagate!(multiclass_log_loss(&y, &scores, 2_usize, None));
    let w = [1.0_f64, 3.0_f64];
    let weighted = propagate!(multiclass_log_loss(&y, &scores, 2_usize, Some(&w)));
    assert!((unweighted - weighted).abs() < 1e-12_f64);
    assert!((unweighted - 2.0_f64.ln()).abs() < 1e-12_f64);
    Ok(())
}

#[test]
fn test_log_loss_rejects_bad_shapes() -> Result<(), ClearGbmError> {
    match multiclass_log_loss(&[], &[], 2_usize, None) {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "empty labels must be rejected".to_string(),
            })
        }
        Err(ClearGbmError::EmptyInput { .. }) => {}
        Err(e) => return Err(e),
    }
    let y = [0_u32];
    match multiclass_log_loss(&y, &[0.0_f64], 2_usize, None) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a short score buffer must be rejected".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_grad_hess_formulas_per_class() {
    // One row, three classes, uniform probabilities p = 1/3. Label 1.
    // grad = p - onehot; hess = (K/(K-1)) * p * (1-p) = 1.5 * (1/3) * (2/3)
    // = 1/3. Weight 2 doubles both.
    let probas = vec![1.0_f64 / 3.0_f64, 1.0_f64 / 3.0_f64, 1.0_f64 / 3.0_f64];
    let mut gradients = vec![0.0_f64; 3];
    let mut hessians = vec![0.0_f64; 3];
    let mut bufs = MulticlassGradBuffers {
        gradients: &mut gradients,
        hessians: &mut hessians,
        n_samples: 1_usize,
        factor: 1.5_f64,
    };
    fill_row_grad_hess(&mut bufs, &probas, 1_u32, 2.0_f64, 0_usize);
    assert!((gradients[0] - 2.0_f64 / 3.0_f64).abs() < 1e-12_f64);
    assert!((gradients[1] - 2.0_f64 * (1.0_f64 / 3.0_f64 - 1.0_f64)).abs() < 1e-12_f64);
    assert!((gradients[2] - 2.0_f64 / 3.0_f64).abs() < 1e-12_f64);
    for &h in &hessians {
        assert!((h - 2.0_f64 / 3.0_f64).abs() < 1e-12_f64, "{hessians:?}");
    }
}
