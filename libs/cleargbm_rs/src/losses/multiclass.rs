//! Softmax cross-entropy pieces for `MulticlassSoftmax`.
//!
//! Scores live in one flat CLASS-MAJOR buffer (`scores[k * n_samples + i]`,
//! LightGBM's layout): each class's block is a contiguous gradient/hessian
//! slice the tree builder consumes directly, and the per-row softmax
//! gathers K strided values. The softmax subtracts the row maximum before
//! exponentiating, so scores of any magnitude stay finite.
//!
//! The hessian uses LightGBM's Friedman rescale `K/(K-1) * p * (1-p)`
//! rather than XGBoost's constant-2 form; the quality gate benchmarks
//! against LightGBM, so its formula is the comparison arm.

use crate::error::ClearGbmError;

/// Probability floor for the class-prior log and the evaluation log; keeps
/// `log` finite when a class is empty or a probability saturates.
const PROB_EPSILON: f64 = 1e-15_f64;

/// Computes one row's softmax from class-major scores.
///
/// # Args
///
/// * `scores` - Flat class-major score buffer, length `n_samples * n_classes`.
/// * `n_samples` - Row count (the class stride).
/// * `row` - The row to gather.
/// * `out` - Receives the row's `n_classes` probabilities.
pub(crate) fn softmax_row_into(scores: &[f64], n_samples: usize, row: usize, out: &mut [f64]) {
    let n_classes = out.len();
    let mut max_score = f64::NEG_INFINITY;
    for (k, slot) in out.iter_mut().enumerate().take(n_classes) {
        let s = scores[k * n_samples + row];
        *slot = s;
        if s > max_score {
            max_score = s;
        }
    }
    let mut sum = 0.0_f64;
    for slot in out.iter_mut() {
        *slot = (*slot - max_score).exp();
        sum += *slot;
    }
    for slot in out.iter_mut() {
        *slot /= sum;
    }
}

/// Computes the per-class base scores: log of the (optionally weighted)
/// class prior, floored at [`PROB_EPSILON`].
///
/// LightGBM's `BoostFromScore` form, uncentered: each class's score column
/// starts from `log(prior_k)`. Softmax is shift-invariant, so centering is
/// a representational choice; the uncentered form is the comparison arm.
///
/// # Args
///
/// * `y_train` - Class labels, each already validated `< n_classes`.
/// * `n_classes` - The class count.
/// * `weights` - Optional per-row weights; `None` weighs every row 1.
///
/// # Returns
///
/// One base score per class.
///
/// # Errors
///
/// Returns [`ClearGbmError::EmptyInput`] if `y_train` is empty.
pub fn multiclass_initial_predictions(
    y_train: &[u32],
    n_classes: usize,
    weights: Option<&[f64]>,
) -> Result<Vec<f64>, ClearGbmError> {
    if y_train.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "y_train for multiclass base scores".to_string(),
        });
    }
    let mut class_weight = vec![0.0_f64; n_classes];
    let mut total = 0.0_f64;
    match weights {
        Some(ws) => {
            for (&y, &w) in y_train.iter().zip(ws.iter()) {
                class_weight[crate::narrow::index_widen(y)] += w;
                total += w;
            }
        }
        None => {
            for &y in y_train {
                class_weight[crate::narrow::index_widen(y)] += 1.0_f64;
            }
            total = f64::from(u32::try_from(y_train.len()).unwrap_or(u32::MAX));
        }
    }
    Ok(class_weight
        .iter()
        .map(|&cw| (cw / total).max(PROB_EPSILON).ln())
        .collect())
}

/// Computes the (optionally weighted) mean multiclass log loss.
///
/// `loss = -sum_i w_i * log(max(eps, p_i[y_i])) / sum_i w_i`, with
/// probabilities gathered row by row from the class-major score buffer via
/// the max-subtracted softmax.
///
/// # Args
///
/// * `y` - Class labels, each `< n_classes`.
/// * `scores` - Flat class-major raw scores, length `y.len() * n_classes`.
/// * `n_classes` - The class count.
/// * `weights` - Optional per-row evaluation weights; `None` weighs every
///   row 1.
///
/// # Returns
///
/// The mean loss.
///
/// # Errors
///
/// * [`ClearGbmError::EmptyInput`] if `y` is empty.
/// * [`ClearGbmError::ShapeMismatch`] if `scores` is not `y.len() *
///   n_classes` long.
pub fn multiclass_log_loss(
    y: &[u32],
    scores: &[f64],
    n_classes: usize,
    weights: Option<&[f64]>,
) -> Result<f64, ClearGbmError> {
    let n = y.len();
    if n == 0_usize {
        return Err(ClearGbmError::EmptyInput {
            context: "labels for multiclass log loss".to_string(),
        });
    }
    if scores.len() != n * n_classes {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("{} scores ({n} rows x {n_classes} classes)", n * n_classes),
            got: format!("{} scores", scores.len()),
        });
    }
    let mut probas = vec![0.0_f64; n_classes];
    let mut loss_sum = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for (i, &label) in y.iter().enumerate() {
        softmax_row_into(scores, n, i, &mut probas);
        let p = probas[crate::narrow::index_widen(label)].max(PROB_EPSILON);
        let w = match weights {
            Some(ws) => ws[i],
            None => 1.0_f64,
        };
        loss_sum -= w * p.ln();
        weight_sum += w;
    }
    Ok(loss_sum / weight_sum)
}

/// The class-major gradient/hessian buffers one boosting round fills.
pub(crate) struct MulticlassGradBuffers<'a> {
    /// Flat class-major gradient buffer, length `n_samples * n_classes`.
    pub gradients: &'a mut [f64],
    /// Flat class-major hessian buffer, same length.
    pub hessians: &'a mut [f64],
    /// The class stride (row count).
    pub n_samples: usize,
    /// The precomputed hessian rescale `K / (K - 1)`.
    pub factor: f64,
}

/// Fills one row's per-class gradient/hessian entries from its softmax.
///
/// For class `k` with row probability `p`: gradient is `p - 1` on the
/// label class and `p` elsewhere; the hessian is `factor * p * (1 - p)`
/// with `factor = K / (K - 1)` (the Friedman rescale of the redundant
/// K-output parameterization). The row's weight multiplies both.
///
/// # Args
///
/// * `bufs` - The class-major output buffers and their geometry.
/// * `probas` - The row's `n_classes` softmax probabilities.
/// * `label` - The row's class label.
/// * `weight` - The row's weight (1.0 when unweighted).
/// * `row` - The row index.
pub(crate) fn fill_row_grad_hess(
    bufs: &mut MulticlassGradBuffers<'_>,
    probas: &[f64],
    label: u32,
    weight: f64,
    row: usize,
) {
    let label_idx = crate::narrow::index_widen(label);
    for (k, &p) in probas.iter().enumerate() {
        let idx = k * bufs.n_samples + row;
        let g = if k == label_idx { p - 1.0_f64 } else { p };
        bufs.gradients[idx] = weight * g;
        bufs.hessians[idx] = weight * (bufs.factor * p * (1.0_f64 - p));
    }
}
