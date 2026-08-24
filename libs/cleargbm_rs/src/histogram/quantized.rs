//! Packed integer histograms for quantized training.
//!
//! The histogram half of LightGBM's quantized path (`serial_tree_learner.cpp`
//! + `feature_histogram.hpp` @ 3ec5b99b, pinned in the tech-wiki): a node's
//! per-bin gradient/hessian sums travel as ONE packed integer — the signed
//! gradient sum in the high half, the non-negative hessian sum in the low
//! half — so one integer add replaces two float adds and the entries pack
//! several per cache line. The width is chosen per node from
//! `node count * quantization bins`: below 65536 the entry is an `i32`
//! (`(grad << 16) | hess`), otherwise an `i64` (`(grad << 32) | hess`).
//! LightGBM's 8-bit tier exists only as a label; its construction path
//! instantiates 16-bit accumulation for those leaves, and so does this one.
//!
//! Packed arithmetic is exact: the hessian half never carries into the
//! gradient half because per-bin hessian sums are bounded by
//! `node count * bins`, which the width selection keeps below the half's
//! capacity, and the discretizer's clamp makes that bound provable. Each
//! bin also carries an exact sample count — this crate enforces
//! `min_samples_leaf` on true counts, not on hessian-derived estimates.
//!
//! Unlike the float path there is no ordered-stream reorder pass: the
//! single node walk gathers each row's 2-byte pair directly, which touches
//! a strict subset of the bytes the float path's reorder would.

use crate::error::ClearGbmError;

/// One bin's accumulators at 16-bit component width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantAcc16 {
    /// Packed sums: signed gradient in the high 16 bits, hessian in the
    /// low 16.
    pub packed: i32,
    /// Exact sample count.
    pub count: u32,
}

impl QuantAcc16 {
    /// The all-zero accumulator a fresh histogram starts from.
    pub const ZERO: Self = Self {
        packed: 0_i32,
        count: 0_u32,
    };
}

/// One bin's accumulators at 32-bit component width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantAcc32 {
    /// Packed sums: signed gradient in the high 32 bits, hessian in the
    /// low 32.
    pub packed: i64,
    /// Exact sample count.
    pub count: u32,
}

impl QuantAcc32 {
    /// The all-zero accumulator a fresh histogram starts from.
    pub const ZERO: Self = Self {
        packed: 0_i64,
        count: 0_u32,
    };
}

/// The per-node histogram component width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantHistWidth {
    /// 16-bit components in an `i32` entry.
    B16,
    /// 32-bit components in an `i64` entry.
    B32,
}

/// One node's per-feature quantized histograms, at the node's width.
///
/// The outer `Vec` is indexed by feature; each inner `Vec` holds `n_bins`
/// accumulators (regular bins plus the NaN bin), mirroring the float
/// path's `Vec<HistogramBuffer>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizedNodeHistograms {
    /// 16-bit-component histograms.
    B16(Vec<Vec<QuantAcc16>>),
    /// 32-bit-component histograms.
    B32(Vec<Vec<QuantAcc32>>),
}

/// Selects a node's histogram width from its sample count.
///
/// LightGBM's `SetNumBitsInHistogramBin` threshold: the largest packed
/// statistic any bin can hold is `node count * quantization bins`
/// (hessian side; the gradient side is half that, signed). Below 65536
/// both halves fit 16-bit components; otherwise 32-bit. The multiply is
/// in u64, so it cannot overflow for any u32-bounded row count
/// (the saturating arms are the crate's dead-arm idiom).
#[must_use]
pub fn select_hist_width(node_count: usize, n_quant_bins: usize) -> QuantHistWidth {
    let count_u64 = u64::try_from(node_count).unwrap_or(u64::MAX);
    let bins_u64 = u64::try_from(n_quant_bins).unwrap_or(u64::MAX);
    if count_u64 * bins_u64 < 65536_u64 {
        QuantHistWidth::B16
    } else {
        QuantHistWidth::B32
    }
}

/// The inputs one single-pass quantized node build needs.
///
/// The quantized sibling of the float path's `NodeHistogramRequest`:
/// instead of two ordered f64 streams it carries the round's interleaved
/// `int8` stream, gathered per row inside the walk.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedNodeHistogramRequest<'a> {
    /// Indices of samples at this node.
    pub sample_indices: &'a [u32],
    /// The round's interleaved stream: hessian at `2i`, gradient at
    /// `2i + 1`, length `2 * n_samples`.
    pub packed_int8: &'a [i8],
    /// Row-major bin matrix, as in the float path.
    pub bins_rows: &'a [u8],
    /// Number of features (the row stride of `bins_rows`).
    pub n_features: usize,
    /// Number of bins per feature, including the NaN bin.
    pub n_bins: usize,
    /// This node's component width, from [`select_hist_width`].
    pub width: QuantHistWidth,
}

/// Builds every feature's quantized histogram for one node in a single
/// sample walk.
///
/// Bit-exact by construction: integer addition is associative, so the
/// per-bin packed sums are independent of walk order — quantized
/// histograms carry none of the float path's summation-order
/// sensitivity.
///
/// # Args
///
/// * `request` - The node-scoped inputs.
///
/// # Returns
///
/// One populated per-feature histogram vector at the requested width.
///
/// # Panics
///
/// Rust's safe indexing will panic if a row extends past `bins_rows`, a
/// bin byte reaches `n_bins`, or `packed_int8` is shorter than the bin
/// matrix implies — caller bugs, the same trust contract as the float
/// single-pass builder.
#[must_use]
pub fn build_quantized_node_histograms(
    request: QuantizedNodeHistogramRequest<'_>,
) -> QuantizedNodeHistograms {
    let QuantizedNodeHistogramRequest {
        sample_indices,
        packed_int8,
        bins_rows,
        n_features,
        n_bins,
        width,
    } = request;

    match width {
        QuantHistWidth::B16 => {
            let mut flat: Vec<QuantAcc16> = vec![QuantAcc16::ZERO; n_features * n_bins];
            for &idx in sample_indices {
                let row_idx = crate::narrow::index_widen(idx);
                let hess = i32::from(packed_int8[2_usize * row_idx]);
                let grad = i32::from(packed_int8[2_usize * row_idx + 1_usize]);
                let packed_value = (grad << 16_u32) + hess;
                let row_start = row_idx * n_features;
                let row = &bins_rows[row_start..row_start + n_features];
                let mut base = 0_usize;
                for &bin in row {
                    let acc = &mut flat[base + usize::from(bin)];
                    acc.packed += packed_value;
                    acc.count += 1_u32;
                    base += n_bins;
                }
            }
            QuantizedNodeHistograms::B16(carve_flat(&flat, n_features, n_bins))
        }
        QuantHistWidth::B32 => {
            let mut flat: Vec<QuantAcc32> = vec![QuantAcc32::ZERO; n_features * n_bins];
            for &idx in sample_indices {
                let row_idx = crate::narrow::index_widen(idx);
                let hess = i64::from(packed_int8[2_usize * row_idx]);
                let grad = i64::from(packed_int8[2_usize * row_idx + 1_usize]);
                let packed_value = (grad << 32_u32) + hess;
                let row_start = row_idx * n_features;
                let row = &bins_rows[row_start..row_start + n_features];
                let mut base = 0_usize;
                for &bin in row {
                    let acc = &mut flat[base + usize::from(bin)];
                    acc.packed += packed_value;
                    acc.count += 1_u32;
                    base += n_bins;
                }
            }
            QuantizedNodeHistograms::B32(carve_flat(&flat, n_features, n_bins))
        }
    }
}

/// Carves one flat accumulation block into per-feature vectors.
fn carve_flat<T: Copy>(flat: &[T], n_features: usize, n_bins: usize) -> Vec<Vec<T>> {
    let mut out: Vec<Vec<T>> = Vec::with_capacity(n_features);
    for feat_idx in 0_usize..n_features {
        out.push(flat[feat_idx * n_bins..(feat_idx + 1_usize) * n_bins].to_vec());
    }
    out
}

/// Unpacks a 16-bit-component entry into `(gradient, hessian, count)`.
///
/// The arithmetic right shift recovers the signed gradient half; the
/// mask recovers the non-negative hessian half — LightGBM's decode.
#[must_use]
pub fn unpack_acc16(acc: QuantAcc16) -> (i64, i64, usize) {
    let grad = i64::from(acc.packed >> 16_u32);
    let hess = i64::from(acc.packed & 0xFFFF_i32);
    (grad, hess, crate::narrow::index_widen(acc.count))
}

/// Unpacks a 32-bit-component entry into `(gradient, hessian, count)`.
#[must_use]
pub fn unpack_acc32(acc: QuantAcc32) -> (i64, i64, usize) {
    let grad = acc.packed >> 32_u32;
    let hess = acc.packed & 0xFFFF_FFFF_i64;
    (grad, hess, crate::narrow::index_widen(acc.count))
}

/// Packs `(gradient, hessian)` into a 16-bit-component entry.
///
/// The component bounds hold by the width-selection invariant, so the
/// conversion arms are statically dead (the crate's dead-arm idiom).
fn pack_acc16(grad: i64, hess: i64, count: u32) -> QuantAcc16 {
    let grad_i32 = i32::try_from(grad).unwrap_or(i32::MAX);
    let hess_i32 = i32::try_from(hess).unwrap_or(i32::MAX);
    QuantAcc16 {
        packed: (grad_i32 << 16_u32) + hess_i32,
        count,
    }
}

/// Packs `(gradient, hessian)` into a 32-bit-component entry.
fn pack_acc32(grad: i64, hess: i64, count: u32) -> QuantAcc32 {
    QuantAcc32 {
        packed: (grad << 32_u32) + hess,
        count,
    }
}

/// Computes the larger sibling's histograms by packed subtraction.
///
/// `sibling = parent - child`, exact in integers — no float
/// reassociation anywhere. Widths mix the way LightGBM's `Subtract`
/// dispatch allows: a child's count never exceeds its parent's, so a
/// 16-bit parent only ever has 16-bit children, while a 32-bit parent
/// may produce the sibling at either width (`larger_width` is selected
/// from the sibling's own count by the caller).
///
/// # Args
///
/// * `parent` - The parent node's histograms.
/// * `child` - The smaller child's histograms.
/// * `larger_width` - The width selected for the sibling being derived.
///
/// # Returns
///
/// The sibling's histograms at `larger_width`.
///
/// # Errors
///
/// Returns `ClearGbmError::ShapeMismatch` if the width combination
/// violates the count-monotonicity invariant (a 32-bit child under a
/// 16-bit parent, or a 32-bit sibling requested under a 16-bit parent).
pub fn subtract_quantized(
    parent: &QuantizedNodeHistograms,
    child: &QuantizedNodeHistograms,
    larger_width: QuantHistWidth,
) -> Result<QuantizedNodeHistograms, ClearGbmError> {
    match (parent, child, larger_width) {
        (
            QuantizedNodeHistograms::B16(parent_features),
            QuantizedNodeHistograms::B16(child_features),
            QuantHistWidth::B16,
        ) => {
            // Same width throughout: subtract the packed entries
            // directly, no unpack round-trip.
            let out: Vec<Vec<QuantAcc16>> = parent_features
                .iter()
                .zip(child_features.iter())
                .map(|(parent_bins, child_bins)| {
                    parent_bins
                        .iter()
                        .zip(child_bins.iter())
                        .map(|(p, c)| QuantAcc16 {
                            packed: p.packed - c.packed,
                            count: p.count.saturating_sub(c.count),
                        })
                        .collect()
                })
                .collect();
            Ok(QuantizedNodeHistograms::B16(out))
        }
        (
            QuantizedNodeHistograms::B32(parent_features),
            QuantizedNodeHistograms::B32(child_features),
            QuantHistWidth::B32,
        ) => {
            // Same width throughout: subtract the packed entries
            // directly, no unpack round-trip — this is the common case
            // on large nodes, where the churn would cost the most.
            let out: Vec<Vec<QuantAcc32>> = parent_features
                .iter()
                .zip(child_features.iter())
                .map(|(parent_bins, child_bins)| {
                    parent_bins
                        .iter()
                        .zip(child_bins.iter())
                        .map(|(p, c)| QuantAcc32 {
                            packed: p.packed - c.packed,
                            count: p.count.saturating_sub(c.count),
                        })
                        .collect()
                })
                .collect();
            Ok(QuantizedNodeHistograms::B32(out))
        }
        (
            QuantizedNodeHistograms::B32(parent_features),
            QuantizedNodeHistograms::B32(child_features),
            QuantHistWidth::B16,
        ) => {
            let unpacked =
                subtract_unpacked(parent_features, child_features, unpack_acc32, unpack_acc32);
            Ok(pack_at_width(unpacked, QuantHistWidth::B16))
        }
        (
            QuantizedNodeHistograms::B32(parent_features),
            QuantizedNodeHistograms::B16(child_features),
            width,
        ) => {
            let unpacked =
                subtract_unpacked(parent_features, child_features, unpack_acc32, unpack_acc16);
            Ok(pack_at_width(unpacked, width))
        }
        (QuantizedNodeHistograms::B16(_), QuantizedNodeHistograms::B32(_), _)
        | (QuantizedNodeHistograms::B16(_), QuantizedNodeHistograms::B16(_), QuantHistWidth::B32) => {
            Err(ClearGbmError::ShapeMismatch {
                expected: "child histogram width <= parent width, sibling width <= parent width"
                    .to_string(),
                got: "a wider histogram under a narrower parent (count monotonicity violated)"
                    .to_string(),
            })
        }
    }
}

/// Subtracts two feature-histogram sets in unpacked `(g, h, count)` space.
fn subtract_unpacked<P: Copy, C: Copy>(
    parent_features: &[Vec<P>],
    child_features: &[Vec<C>],
    unpack_parent: fn(P) -> (i64, i64, usize),
    unpack_child: fn(C) -> (i64, i64, usize),
) -> Vec<Vec<(i64, i64, u32)>> {
    parent_features
        .iter()
        .zip(child_features.iter())
        .map(|(parent_bins, child_bins)| {
            parent_bins
                .iter()
                .zip(child_bins.iter())
                .map(|(&p, &c)| {
                    let (pg, ph, pc) = unpack_parent(p);
                    let (cg, ch, cc) = unpack_child(c);
                    // Counts are u32-bounded by construction; the
                    // saturating arms mirror the float path's
                    // `saturating_sub` and the crate's dead-arm idiom.
                    let count = u32::try_from(pc.saturating_sub(cc)).unwrap_or(u32::MAX);
                    (pg - cg, ph - ch, count)
                })
                .collect()
        })
        .collect()
}

/// Packs unpacked feature histograms at the requested width.
fn pack_at_width(
    unpacked: Vec<Vec<(i64, i64, u32)>>,
    width: QuantHistWidth,
) -> QuantizedNodeHistograms {
    match width {
        QuantHistWidth::B16 => QuantizedNodeHistograms::B16(
            unpacked
                .into_iter()
                .map(|bins| {
                    bins.into_iter()
                        .map(|(g, h, count)| pack_acc16(g, h, count))
                        .collect()
                })
                .collect(),
        ),
        QuantHistWidth::B32 => QuantizedNodeHistograms::B32(
            unpacked
                .into_iter()
                .map(|bins| {
                    bins.into_iter()
                        .map(|(g, h, count)| pack_acc32(g, h, count))
                        .collect()
                })
                .collect(),
        ),
    }
}

/// Materializes every feature's bins as unpacked `(g, h, count)` triples.
///
/// The split scan consumes this shape so one integer scan serves both
/// widths; the materialization is O(features x bins), far off the
/// per-sample hot path.
///
/// # Args
///
/// * `histograms` - A node's histograms.
///
/// # Returns
///
/// Per-feature bin triples, in feature order.
#[must_use]
pub fn unpacked_features(histograms: &QuantizedNodeHistograms) -> Vec<Vec<(i64, i64, usize)>> {
    match histograms {
        QuantizedNodeHistograms::B16(features) => features
            .iter()
            .map(|bins| bins.iter().map(|&acc| unpack_acc16(acc)).collect())
            .collect(),
        QuantizedNodeHistograms::B32(features) => features
            .iter()
            .map(|bins| bins.iter().map(|&acc| unpack_acc32(acc)).collect())
            .collect(),
    }
}
