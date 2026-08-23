//! Categorical feature binning: one bin per distinct category code.
//!
//! A categorical feature's values are non-negative integer codes carried in
//! `f64` (the matrix dtype). Binning assigns each distinct code its own bin
//! in ascending code order — no ordering assumption ever enters a split,
//! because the categorical split search partitions bins by set membership
//! rather than by threshold. Missing values keep the shared NaN bin.
//!
//! There is no overflow policy: a feature with more distinct codes than
//! `max_bins` is an error naming the feature and both counts, never a
//! silent grouping of rare categories.

use crate::error::ClearGbmError;

/// Upper bound accepted for a category code.
///
/// Codes are integer-valued `f64`; capping at `u32::MAX` keeps every code
/// exactly representable and matches the crate's other u32 ceilings
/// (`n_samples`, `n_features`).
const MAX_CATEGORY_CODE: f64 = 4_294_967_295.0_f64;

/// The category-code table for one categorical feature.
///
/// `codes` is sorted ascending, distinct, integer-valued, and free of
/// `-0.0` (normalized to `0.0` at construction). Bin `i` holds exactly the
/// samples whose value equals `codes[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoryMap {
    /// The distinct category codes, ascending.
    codes: Vec<f64>,
}

impl CategoryMap {
    /// Returns the distinct category codes, ascending.
    #[must_use]
    pub fn codes(&self) -> &[f64] {
        &self.codes
    }

    /// Returns the number of categories (= the feature's bin count).
    #[must_use]
    pub fn n_categories(&self) -> usize {
        self.codes.len()
    }
}

/// Builds the category map and per-row bin assignments for one column.
///
/// Single construction pass: the non-NaN values are sorted with their row
/// positions, distinct codes become bins in ascending order, and each row's
/// bin is assigned during the same walk — no lookup step exists to fail.
///
/// # Args
///
/// * `column` - The feature's values, one per sample. NaN marks missing.
/// * `feature_index` - The feature's position, for error messages.
/// * `max_bins` - The uniform regular-bin budget; the distinct-code count
///   must fit within it.
///
/// # Returns
///
/// The map plus one `Option<usize>` per row: `Some(bin)` for a category
/// value, `None` for missing (the caller writes the shared NaN bin).
///
/// # Errors
///
/// Returns [`ClearGbmError::InvalidParameter`] if any value is not a
/// non-negative integer within `u32::MAX` (naming the feature, row and
/// value), or if the distinct-code count exceeds `max_bins`.
pub(super) fn categorical_column_bins(
    column: &[f64],
    feature_index: usize,
    max_bins: usize,
) -> Result<(CategoryMap, Vec<Option<usize>>), ClearGbmError> {
    // Validate + normalize (-0.0 -> 0.0) the non-missing values, keeping
    // their row positions for the assignment pass.
    let mut indexed: Vec<(f64, usize)> = Vec::with_capacity(column.len());
    for (row, &raw) in column.iter().enumerate() {
        if raw.is_nan() {
            continue;
        }
        if !raw.is_finite() || raw < 0.0_f64 || raw.fract() != 0.0_f64 || raw > MAX_CATEGORY_CODE {
            return Err(ClearGbmError::InvalidParameter {
                name: "categorical_features".to_string(),
                reason: format!(
                    "feature {feature_index} row {row}: categorical values must be \
                     non-negative integer codes (or NaN for missing), got {raw}"
                ),
            });
        }
        indexed.push((raw + 0.0_f64, row));
    }

    // Ascending by code; ties keep row order (irrelevant to the result but
    // deterministic). total_cmp is total on the validated, NaN-free codes.
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut codes: Vec<f64> = Vec::new();
    let mut bins: Vec<Option<usize>> = vec![None; column.len()];
    for &(code, row) in &indexed {
        let is_new = match codes.last() {
            Some(&last) => code > last,
            None => true,
        };
        if is_new {
            codes.push(code);
        }
        bins[row] = Some(codes.len() - 1_usize);
    }

    if codes.len() > max_bins {
        return Err(ClearGbmError::InvalidParameter {
            name: "categorical_features".to_string(),
            reason: format!(
                "feature {feature_index} has {} distinct categories, which exceeds \
                 max_bins ({max_bins}); raise max_bins to cover every category",
                codes.len()
            ),
        });
    }

    Ok((CategoryMap { codes }, bins))
}
