//! The per-feature category tables a tree build consults.
//!
//! Built once per training run from the binning result and threaded through
//! [`super::builder::BuildTreeInput`]: the split search reads which features
//! are categorical (and how many category bins each uses), and node
//! finalization translates a categorical split's left-routed BINS into the
//! raw category CODES the prediction path compares against — prediction has
//! no binning, so the codes must live on the node.

use crate::error::ClearGbmError;
use crate::split::CategoryBinSet;

/// Per-feature category tables for one training run.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalLayout {
    /// `Some(sorted category codes)` for categorical features, `None` for
    /// numeric ones. Indexed by feature.
    per_feature: Vec<Option<Vec<f64>>>,
}

impl CategoricalLayout {
    /// Creates a layout from per-feature code tables.
    ///
    /// # Args
    ///
    /// * `per_feature` - One entry per feature: the sorted distinct category
    ///   codes for categorical features, `None` for numeric ones.
    ///
    /// # Returns
    ///
    /// The layout.
    #[must_use]
    pub fn new(per_feature: Vec<Option<Vec<f64>>>) -> Self {
        Self { per_feature }
    }

    /// Returns the number of category bins a feature uses, or `None` for a
    /// numeric feature (or a feature index beyond the layout).
    #[must_use]
    pub fn n_categories(&self, feature_index: usize) -> Option<usize> {
        match self.per_feature.get(feature_index) {
            Some(Some(codes)) => Some(codes.len()),
            _ => None,
        }
    }

    /// Translates a categorical split's left-routed bins into raw codes.
    ///
    /// # Args
    ///
    /// * `feature_index` - The split's feature.
    /// * `left_bins` - The bins the split routes left.
    ///
    /// # Returns
    ///
    /// The category codes routed left, ascending (bin order IS code order).
    ///
    /// # Errors
    ///
    /// Returns [`ClearGbmError::TreeConstructionFailed`] if the feature is
    /// not categorical in this layout or a bin index falls outside its code
    /// table — either means the split search and the layout disagree about
    /// the feature, which is a construction bug worth failing loudly on.
    pub fn left_codes(
        &self,
        feature_index: usize,
        left_bins: CategoryBinSet,
    ) -> Result<Vec<f64>, ClearGbmError> {
        let codes = match self.per_feature.get(feature_index) {
            Some(Some(codes)) => codes,
            _ => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!(
                        "categorical split on feature {feature_index}, which the \
                         categorical layout does not mark as categorical"
                    ),
                })
            }
        };
        let mut out = Vec::with_capacity(left_bins.len());
        for bin in left_bins.bins() {
            match codes.get(bin) {
                Some(&code) => out.push(code),
                None => {
                    return Err(ClearGbmError::TreeConstructionFailed {
                        reason: format!(
                            "categorical split on feature {feature_index} routes bin {bin} \
                             left, but the feature has only {} categories",
                            codes.len()
                        ),
                    })
                }
            }
        }
        Ok(out)
    }
}
