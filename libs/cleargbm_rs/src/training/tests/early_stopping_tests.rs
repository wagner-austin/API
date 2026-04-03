//! Tests for EarlyStoppingState.

use crate::error::ClearGbmError;
use crate::training::early_stopping::EarlyStoppingState;

#[test]
fn test_initial_state() -> Result<(), ClearGbmError> {
    let es = EarlyStoppingState::new(5_usize);
    assert_eq!(es.best_round(), 0_usize);
    Ok(())
}

#[test]
fn test_improvement_resets_counter() -> Result<(), ClearGbmError> {
    let mut es = EarlyStoppingState::new(3_usize);
    // Decreasing losses should never trigger stop
    assert!(!es.update(1.0_f64, 0_usize));
    assert!(!es.update(0.9_f64, 1_usize));
    assert!(!es.update(0.8_f64, 2_usize));
    assert!(!es.update(0.7_f64, 3_usize));
    assert!(!es.update(0.6_f64, 4_usize));
    assert_eq!(es.best_round(), 4_usize);
    Ok(())
}

#[test]
fn test_plateau_triggers_stop() -> Result<(), ClearGbmError> {
    let mut es = EarlyStoppingState::new(3_usize);
    // Good loss at round 0
    assert!(!es.update(0.5_f64, 0_usize));
    assert_eq!(es.best_round(), 0_usize);
    // Worse for 3 rounds (patience=3)
    assert!(!es.update(0.6_f64, 1_usize));
    assert!(!es.update(0.7_f64, 2_usize));
    // Third consecutive no-improvement triggers stop
    assert!(es.update(0.8_f64, 3_usize));
    assert_eq!(es.best_round(), 0_usize);
    Ok(())
}

#[test]
fn test_improvement_after_plateau() -> Result<(), ClearGbmError> {
    let mut es = EarlyStoppingState::new(3_usize);
    // Good loss
    assert!(!es.update(0.5_f64, 0_usize));
    // Two rounds of no improvement (but not yet patience)
    assert!(!es.update(0.6_f64, 1_usize));
    assert!(!es.update(0.7_f64, 2_usize));
    // Improvement! Resets counter
    assert!(!es.update(0.4_f64, 3_usize));
    assert_eq!(es.best_round(), 3_usize);
    // Two more rounds of no improvement — still under patience
    assert!(!es.update(0.5_f64, 4_usize));
    assert!(!es.update(0.6_f64, 5_usize));
    // Third consecutive — triggers stop
    assert!(es.update(0.7_f64, 6_usize));
    assert_eq!(es.best_round(), 3_usize);
    Ok(())
}

#[test]
fn test_equal_loss_no_improvement() -> Result<(), ClearGbmError> {
    let mut es = EarlyStoppingState::new(2_usize);
    assert!(!es.update(0.5_f64, 0_usize));
    // Same loss is NOT an improvement
    assert!(!es.update(0.5_f64, 1_usize));
    // Second consecutive no-improvement triggers stop
    assert!(es.update(0.5_f64, 2_usize));
    assert_eq!(es.best_round(), 0_usize);
    Ok(())
}

#[test]
fn test_patience_one() -> Result<(), ClearGbmError> {
    let mut es = EarlyStoppingState::new(1_usize);
    assert!(!es.update(0.5_f64, 0_usize));
    // Any non-improvement immediately triggers stop
    assert!(es.update(0.6_f64, 1_usize));
    assert_eq!(es.best_round(), 0_usize);
    Ok(())
}

#[test]
fn test_best_round_updates() -> Result<(), ClearGbmError> {
    let mut es = EarlyStoppingState::new(10_usize);
    assert!(!es.update(1.0_f64, 0_usize));
    assert_eq!(es.best_round(), 0_usize);
    assert!(!es.update(0.8_f64, 1_usize));
    assert_eq!(es.best_round(), 1_usize);
    assert!(!es.update(0.9_f64, 2_usize));
    // Best is still round 1
    assert_eq!(es.best_round(), 1_usize);
    assert!(!es.update(0.7_f64, 3_usize));
    assert_eq!(es.best_round(), 3_usize);
    Ok(())
}

#[test]
fn test_clone_and_eq() -> Result<(), ClearGbmError> {
    let mut es1 = EarlyStoppingState::new(5_usize);
    es1.update(0.5_f64, 0_usize);
    let es2 = es1.clone();
    assert_eq!(es1, es2);
    Ok(())
}

#[test]
fn test_debug_format() -> Result<(), ClearGbmError> {
    let es = EarlyStoppingState::new(3_usize);
    let debug_str = format!("{es:?}");
    assert!(debug_str.contains("EarlyStoppingState"));
    Ok(())
}
