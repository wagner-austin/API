//! Bin-layout helpers for tests.
//!
//! Test fixtures across the crate are written as column-major bin arrays
//! (the layout the crate used until 2026-08-21); production storage is
//! row-major. This transpose lets those fixtures feed the row-major
//! surface without rewriting every literal.

/// Transposes a column-major bin matrix into row-major.
///
/// # Args
///
/// * `cols` - Column-major bins: `cols[feat_idx * n_samples + sample_idx]`.
/// * `n_samples` - Row count.
/// * `n_features` - Column count.
///
/// # Returns
///
/// Row-major bins: sample `i`'s features at `rows[i * n_features..]`.
///
/// # Panics
///
/// Panics if `cols.len() != n_samples * n_features` via safe indexing.
#[must_use]
pub fn transpose_cols_to_rows(cols: &[u8], n_samples: usize, n_features: usize) -> Vec<u8> {
    let mut rows = vec![0_u8; n_samples * n_features];
    for feat_idx in 0_usize..n_features {
        let col_start = feat_idx * n_samples;
        for sample_idx in 0_usize..n_samples {
            rows[sample_idx * n_features + feat_idx] = cols[col_start + sample_idx];
        }
    }
    rows
}
