//! Tests for the LambdaMART ranking loss: gain/discount tables, query-group
//! validation, max DCG, the pair scan, and the NDCG metric.

use crate::error::ClearGbmError;
use crate::losses::lambdarank::{
    fill_query_lambdas, inverse_max_dcg_at_k, label_gains, mean_ndcg_at_k, ndcg_at_k,
    position_discount, validate_query_groups, validate_ranking_labels, MAX_QUERY_LENGTH,
};

fn assert_close(a: f64, b: f64, tolerance: f64, what: &str) {
    assert!((a - b).abs() < tolerance, "{what}: {a} vs {b}");
}

// =============================================================================
// Tables
// =============================================================================

#[test]
fn test_label_gains_are_two_to_the_label_minus_one() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    assert_eq!(gains.len(), 32_usize);
    assert_close(gains[0], 0.0_f64, 1e-12_f64, "gain[0]");
    assert_close(gains[1], 1.0_f64, 1e-12_f64, "gain[1]");
    assert_close(gains[2], 3.0_f64, 1e-12_f64, "gain[2]");
    assert_close(gains[5], 31.0_f64, 1e-12_f64, "gain[5]");
    assert_close(gains[31], 2_147_483_647.0_f64, 1e-3_f64, "gain[31]");
    Ok(())
}

#[test]
fn test_position_discount_is_inverse_log2_of_rank_plus_two() -> Result<(), ClearGbmError> {
    assert_close(position_discount(0_usize), 1.0_f64, 1e-12_f64, "rank 0");
    // 1 / log2(3)
    assert_close(
        position_discount(1_usize),
        1.0_f64 / 3.0_f64.log2(),
        1e-12_f64,
        "rank 1",
    );
    // 1 / log2(4) = 0.5
    assert_close(position_discount(2_usize), 0.5_f64, 1e-12_f64, "rank 2");
    Ok(())
}

// =============================================================================
// Validation
// =============================================================================

#[test]
fn test_labels_up_to_31_pass_and_32_is_rejected() -> Result<(), ClearGbmError> {
    assert!(validate_ranking_labels(&[0_u32, 5_u32, 31_u32], "y_train").is_ok());
    match validate_ranking_labels(&[0_u32, 32_u32], "y_train") {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_train");
            assert!(reason.contains("got 32 at index 1"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter, got {other:?}"),
        }),
    }
}

#[test]
fn test_groups_must_be_non_empty() -> Result<(), ClearGbmError> {
    match validate_query_groups(&[], 0_usize, "group") {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "group");
            assert!(reason.contains("non-empty"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter, got {other:?}"),
        }),
    }
}

#[test]
fn test_an_empty_group_is_rejected() -> Result<(), ClearGbmError> {
    match validate_query_groups(&[2_usize, 0_usize], 2_usize, "group") {
        Err(ClearGbmError::InvalidParameter { reason, .. }) => {
            assert!(reason.contains("index 1 is empty"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter, got {other:?}"),
        }),
    }
}

#[test]
fn test_an_oversize_group_is_rejected_naming_the_cap() -> Result<(), ClearGbmError> {
    let oversize = MAX_QUERY_LENGTH + 1_usize;
    match validate_query_groups(&[oversize], oversize, "group") {
        Err(ClearGbmError::InvalidParameter { reason, .. }) => {
            assert!(reason.contains("10000 cap"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter, got {other:?}"),
        }),
    }
}

#[test]
fn test_group_sum_must_match_the_row_count() -> Result<(), ClearGbmError> {
    match validate_query_groups(&[2_usize, 3_usize], 6_usize, "val_group") {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "val_group");
            assert!(reason.contains("sum to 5 but there are 6 rows"), "{reason}");
        }
        other => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected InvalidParameter, got {other:?}"),
            })
        }
    }
    assert!(validate_query_groups(&[2_usize, 3_usize], 5_usize, "group").is_ok());
    Ok(())
}

// =============================================================================
// Max DCG
// =============================================================================

#[test]
fn test_inverse_max_dcg_matches_the_hand_computed_ideal() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    // Labels {2, 1, 0}: ideal ordering puts gain 3 at discount 1.0 and
    // gain 1 at discount 1/log2(3).
    let expected = 3.0_f64 + 1.0_f64 / 3.0_f64.log2();
    let inv = inverse_max_dcg_at_k(&[0_u32, 2_u32, 1_u32], 3_usize, &gains);
    assert_close(1.0_f64 / inv, expected, 1e-12_f64, "max dcg");
    Ok(())
}

#[test]
fn test_inverse_max_dcg_truncates_at_k() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    // At k=1 only the best label counts: maxDCG = 3.0.
    let inv = inverse_max_dcg_at_k(&[0_u32, 2_u32, 1_u32], 1_usize, &gains);
    assert_close(1.0_f64 / inv, 3.0_f64, 1e-12_f64, "max dcg at 1");
    Ok(())
}

#[test]
fn test_all_zero_labels_yield_inverse_zero() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    let inv = inverse_max_dcg_at_k(&[0_u32, 0_u32, 0_u32], 3_usize, &gains);
    assert_close(inv, 0.0_f64, 1e-15_f64, "degenerate query");
    Ok(())
}

// =============================================================================
// The pair scan
// =============================================================================

#[test]
fn test_a_misordered_pair_pushes_the_relevant_document_up() -> Result<(), ClearGbmError> {
    // Two documents, the relevant one scored LOWER: its lambda must be
    // negative (the leaf formula -G/H then pushes its score up) and the
    // irrelevant one's positive, with hessians positive and symmetric.
    let gains = label_gains();
    let scores = [0.0_f64, 1.0_f64];
    let labels = [1_u32, 0_u32];
    let inv = inverse_max_dcg_at_k(&labels, 2_usize, &gains);
    let mut grad = [0.0_f64; 2];
    let mut hess = [0.0_f64; 2];
    fill_query_lambdas(&scores, &labels, inv, 2_usize, &gains, &mut grad, &mut hess);
    assert!(grad[0] < 0.0_f64, "relevant doc gradient: {}", grad[0]);
    assert!(grad[1] > 0.0_f64, "irrelevant doc gradient: {}", grad[1]);
    assert_close(grad[0], -grad[1], 1e-12_f64, "lambda symmetry");
    assert!(hess[0] > 0.0_f64 && hess[1] > 0.0_f64);
    assert_close(hess[0], hess[1], 1e-12_f64, "hessian symmetry");
    Ok(())
}

#[test]
fn test_equal_labels_produce_no_lambdas() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    let scores = [0.3_f64, 0.9_f64, 0.1_f64];
    let labels = [1_u32, 1_u32, 1_u32];
    let inv = inverse_max_dcg_at_k(&labels, 3_usize, &gains);
    let mut grad = [1.0_f64; 3];
    let mut hess = [1.0_f64; 3];
    fill_query_lambdas(&scores, &labels, inv, 3_usize, &gains, &mut grad, &mut hess);
    assert_eq!(grad, [0.0_f64; 3]);
    assert_eq!(hess, [0.0_f64; 3]);
    Ok(())
}

#[test]
fn test_a_single_document_query_is_a_no_op() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    let mut grad = [7.0_f64];
    let mut hess = [7.0_f64];
    fill_query_lambdas(
        &[0.5_f64],
        &[2_u32],
        1.0_f64,
        10_usize,
        &gains,
        &mut grad,
        &mut hess,
    );
    assert_eq!(grad, [0.0_f64]);
    assert_eq!(hess, [0.0_f64]);
    Ok(())
}

#[test]
fn test_matches_the_hand_computed_two_document_pair() -> Result<(), ClearGbmError> {
    // Degenerate equal scores: the norm's score-distance division is
    // skipped (best == worst), leaving the closed form
    //   |dNDCG| = (gain gap) * (discount gap) * 1/maxDCG,
    //   lambda = -|dNDCG| * sigmoid(0) = -|dNDCG| / 2,
    // then the log2(1 + sum)/sum rescale with sum = 2 * |dNDCG| / 2.
    let gains = label_gains();
    let scores = [0.0_f64, 0.0_f64];
    let labels = [0_u32, 1_u32];
    let inv = inverse_max_dcg_at_k(&labels, 2_usize, &gains);
    // maxDCG = 1.0 (gain 1 at discount 1.0), so inv = 1.
    assert_close(inv, 1.0_f64, 1e-12_f64, "inv max dcg");
    let mut grad = [0.0_f64; 2];
    let mut hess = [0.0_f64; 2];
    fill_query_lambdas(&scores, &labels, inv, 2_usize, &gains, &mut grad, &mut hess);
    let delta = (1.0_f64 - 0.0_f64) * (1.0_f64 - 1.0_f64 / 3.0_f64.log2()) * 1.0_f64;
    let raw_lambda = 0.5_f64 * delta;
    let sum = 2.0_f64 * raw_lambda;
    let norm = (1.0_f64 + sum).log2() / sum;
    // Row 0 (label 0) is pushed down, row 1 (label 1) up.
    assert_close(grad[0], raw_lambda * norm, 1e-12_f64, "low-row lambda");
    assert_close(grad[1], -raw_lambda * norm, 1e-12_f64, "high-row lambda");
    let raw_hessian = 0.25_f64 * delta;
    assert_close(hess[0], raw_hessian * norm, 1e-12_f64, "hessian");
    Ok(())
}

#[test]
fn test_truncation_level_bounds_the_outer_loop() -> Result<(), ClearGbmError> {
    // Three documents scored in label order except the last two swapped;
    // at truncation 1 only pairs containing the top-scored document count.
    let gains = label_gains();
    let scores = [3.0_f64, 1.0_f64, 2.0_f64];
    let labels = [2_u32, 1_u32, 0_u32];
    let inv = inverse_max_dcg_at_k(&labels, 3_usize, &gains);
    let mut grad_full = [0.0_f64; 3];
    let mut hess_full = [0.0_f64; 3];
    fill_query_lambdas(
        &scores,
        &labels,
        inv,
        3_usize,
        &gains,
        &mut grad_full,
        &mut hess_full,
    );
    let mut grad_cut = [0.0_f64; 3];
    let mut hess_cut = [0.0_f64; 3];
    fill_query_lambdas(
        &scores,
        &labels,
        inv,
        1_usize,
        &gains,
        &mut grad_cut,
        &mut hess_cut,
    );
    // The full scan also counts the (rank 1, rank 2) pair — the swapped
    // tail — so the truncated gradients differ from the full ones there.
    assert!(
        (grad_full[1] - grad_cut[1]).abs() > 1e-9_f64,
        "truncation had no effect: {} vs {}",
        grad_full[1],
        grad_cut[1]
    );
    Ok(())
}

// =============================================================================
// NDCG
// =============================================================================

#[test]
fn test_perfect_ordering_scores_one() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    let scores = [3.0_f64, 2.0_f64, 1.0_f64];
    let labels = [2_u32, 1_u32, 0_u32];
    assert_close(
        ndcg_at_k(&scores, &labels, 3_usize, &gains),
        1.0_f64,
        1e-12_f64,
        "perfect ndcg",
    );
    Ok(())
}

#[test]
fn test_reversed_ordering_matches_the_hand_computed_ratio() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    let scores = [1.0_f64, 2.0_f64, 3.0_f64];
    let labels = [2_u32, 1_u32, 0_u32];
    // Observed: gain 0 at pos 0, gain 1 at pos 1, gain 3 at pos 2.
    let observed = 1.0_f64 / 3.0_f64.log2() + 3.0_f64 * 0.5_f64;
    let ideal = 3.0_f64 + 1.0_f64 / 3.0_f64.log2();
    assert_close(
        ndcg_at_k(&scores, &labels, 3_usize, &gains),
        observed / ideal,
        1e-12_f64,
        "reversed ndcg",
    );
    Ok(())
}

#[test]
fn test_all_zero_labels_score_one() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    assert_close(
        ndcg_at_k(&[0.5_f64, 0.1_f64], &[0_u32, 0_u32], 2_usize, &gains),
        1.0_f64,
        1e-15_f64,
        "nothing to rank",
    );
    Ok(())
}

#[test]
fn test_mean_ndcg_averages_per_query_values() -> Result<(), ClearGbmError> {
    let gains = label_gains();
    // Query 1 perfectly ordered (1.0), query 2 all-zero labels (1.0 by
    // definition), query 3 reversed two documents.
    let scores = [2.0_f64, 1.0_f64, 0.4_f64, 0.6_f64, 0.0_f64, 1.0_f64];
    let labels = [1_u32, 0_u32, 0_u32, 0_u32, 1_u32, 0_u32];
    let groups = [2_usize, 2_usize, 2_usize];
    // Query 3: observed = 1/log2(3), ideal = 1.
    let q3 = 1.0_f64 / 3.0_f64.log2();
    let expected = (1.0_f64 + 1.0_f64 + q3) / 3.0_f64;
    assert_close(
        mean_ndcg_at_k(&scores, &labels, &groups, 2_usize, &gains),
        expected,
        1e-12_f64,
        "mean ndcg",
    );
    Ok(())
}
