//! Histogram accumulation types: the per-bin gradient/hessian/count
//! records and the buffer that holds one feature's bins.

use crate::error::ClearGbmError;

/// Histogram buffer for gradient/hessian accumulation.
///
/// Used during split finding to accumulate statistics per bin.
/// Equivalent to Python `HistogramBuffer` but with explicit sizing.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBuffer {
    /// Per-bin accumulators, stored interleaved so one sample's update
    /// touches one contiguous 24-byte record instead of three parallel
    /// arrays. See [`BinAccumulator`] for why.
    pub(crate) bins: Vec<BinAccumulator>,

    /// Number of bins (fixed at construction).
    pub(crate) n_bins: usize,
}

/// One bin's accumulators, interleaved.
///
/// The histogram hot loop performs a read-modify-write on all three values
/// for one bin per sample. With three parallel arrays that update touches
/// three cache lines and pays three bounds checks; interleaved it touches one
/// 24-byte record (one line, occasionally two when the record straddles a
/// boundary) and pays a single bounds check. This is LightGBM's `hist_t`
/// grad/hess interleaving, extended to carry the sample count this codebase
/// also accumulates. Splitting the count into a parallel `u32` array to
/// shrink the record to 16 bytes was measured 2026-08-22 at +20% fit time —
/// the second memory touch per update costs far more than the straddle it
/// removes — so the count stays fused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BinAccumulator {
    /// Sum of gradients in this bin.
    pub(crate) gradient_sum: f64,

    /// Sum of hessians in this bin.
    pub(crate) hessian_sum: f64,

    /// Count of samples in this bin.
    pub(crate) count: usize,
}

impl BinAccumulator {
    /// The all-zero accumulator a fresh histogram starts from.
    pub(crate) const ZERO: Self = Self {
        gradient_sum: 0.0_f64,
        hessian_sum: 0.0_f64,
        count: 0_usize,
    };
}

impl HistogramBuffer {
    /// Creates a new zeroed histogram buffer.
    ///
    /// # Args
    ///
    /// * `n_bins` - Number of bins (including NaN bin).
    ///
    /// # Returns
    ///
    /// A new zeroed `HistogramBuffer`.
    #[must_use]
    pub fn new(n_bins: usize) -> Self {
        Self {
            bins: vec![BinAccumulator::ZERO; n_bins],
            n_bins,
        }
    }

    /// Returns the number of bins.
    #[must_use]
    pub const fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Accumulates a sample into the appropriate bin.
    ///
    /// # Args
    ///
    /// * `bin` - Bin index for this sample.
    /// * `gradient` - Gradient value.
    /// * `hessian` - Hessian value.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn accumulate(
        &mut self,
        bin: usize,
        gradient: f64,
        hessian: f64,
    ) -> Result<(), ClearGbmError> {
        let n_bins = self.n_bins;
        let Some(acc) = self.bins.get_mut(bin) else {
            return Err(ClearGbmError::BinIndexOutOfBounds { bin, n_bins });
        };
        acc.gradient_sum += gradient;
        acc.hessian_sum += hessian;
        acc.count += 1;
        Ok(())
    }

    /// Returns the gradient sum for a bin.
    ///
    /// # Args
    ///
    /// * `bin` - Bin index.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn gradient_sum(&self, bin: usize) -> Result<f64, ClearGbmError> {
        self.bins
            .get(bin)
            .map(|acc| acc.gradient_sum)
            .ok_or(ClearGbmError::BinIndexOutOfBounds {
                bin,
                n_bins: self.n_bins,
            })
    }

    /// Returns the hessian sum for a bin.
    ///
    /// # Args
    ///
    /// * `bin` - Bin index.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn hessian_sum(&self, bin: usize) -> Result<f64, ClearGbmError> {
        self.bins
            .get(bin)
            .map(|acc| acc.hessian_sum)
            .ok_or(ClearGbmError::BinIndexOutOfBounds {
                bin,
                n_bins: self.n_bins,
            })
    }

    /// Returns the count for a bin.
    ///
    /// # Args
    ///
    /// * `bin` - Bin index.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn count(&self, bin: usize) -> Result<usize, ClearGbmError> {
        self.bins
            .get(bin)
            .map(|acc| acc.count)
            .ok_or(ClearGbmError::BinIndexOutOfBounds {
                bin,
                n_bins: self.n_bins,
            })
    }

    /// Returns all gradient sums, materialized in bin order.
    ///
    /// Materializes from the interleaved storage; intended for
    /// serialization and inspection, not for hot-path use.
    #[must_use]
    pub fn gradient_sums(&self) -> Vec<f64> {
        self.bins.iter().map(|acc| acc.gradient_sum).collect()
    }

    /// Returns all hessian sums, materialized in bin order.
    ///
    /// Materializes from the interleaved storage; intended for
    /// serialization and inspection, not for hot-path use.
    #[must_use]
    pub fn hessian_sums(&self) -> Vec<f64> {
        self.bins.iter().map(|acc| acc.hessian_sum).collect()
    }

    /// Returns all counts, materialized in bin order.
    ///
    /// Materializes from the interleaved storage; intended for
    /// serialization and inspection, not for hot-path use.
    #[must_use]
    pub fn counts(&self) -> Vec<usize> {
        self.bins.iter().map(|acc| acc.count).collect()
    }

    /// Resets all bins to zero (for reuse).
    pub fn reset(&mut self) {
        self.bins.fill(BinAccumulator::ZERO);
    }

    /// Computes sibling histogram by subtraction: self = parent - child.
    ///
    /// This is the "histogram trick" for 2x speedup: instead of building
    /// both child histograms, build one and subtract from parent.
    ///
    /// # Args
    ///
    /// * `parent` - Parent node histogram.
    /// * `child` - One child's histogram.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::ShapeMismatch` if bin counts don't match.
    pub fn subtract_into(&mut self, parent: &Self, child: &Self) -> Result<(), ClearGbmError> {
        if parent.n_bins != self.n_bins || child.n_bins != self.n_bins {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!(
                    "self.n_bins={}, parent.n_bins={}, child.n_bins={}",
                    self.n_bins, parent.n_bins, child.n_bins
                ),
                got: "bin counts must all match".to_string(),
            });
        }

        for i in 0_usize..self.n_bins {
            self.bins[i] = BinAccumulator {
                gradient_sum: parent.bins[i].gradient_sum - child.bins[i].gradient_sum,
                hessian_sum: parent.bins[i].hessian_sum - child.bins[i].hessian_sum,
                count: parent.bins[i].count.saturating_sub(child.bins[i].count),
            };
        }

        Ok(())
    }

    /// Copies contents from another histogram buffer.
    ///
    /// # Args
    ///
    /// * `other` - Source histogram buffer.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::ShapeMismatch` if bin counts don't match.
    pub fn copy_from(&mut self, other: &Self) -> Result<(), ClearGbmError> {
        if other.n_bins != self.n_bins {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("n_bins={}", self.n_bins),
                got: format!("n_bins={}", other.n_bins),
            });
        }

        self.bins.copy_from_slice(&other.bins);

        Ok(())
    }
}
