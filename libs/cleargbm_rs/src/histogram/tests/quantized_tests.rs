//! Unit tests for the packed integer histograms (`histogram::quantized`).
//!
//! Packed values are hand-computed: at 16-bit width an entry is
//! `(grad << 16) + hess`, at 32-bit `(grad << 32) + hess`, with the
//! hessian always non-negative.

use crate::error::ClearGbmError;
use crate::histogram::quantized::{
    build_quantized_node_histograms, select_hist_width, subtract_quantized, unpack_acc16,
    unpack_acc32, unpacked_features, QuantAcc16, QuantAcc32, QuantHistWidth,
    QuantizedNodeHistogramRequest, QuantizedNodeHistograms,
};

/// Interleaved int8 stream for 3 rows: (h, g) = (2, 0), (2, -1), (4, 2).
fn three_row_stream() -> Vec<i8> {
    vec![2_i8, 0_i8, 2_i8, -1_i8, 4_i8, 2_i8]
}

#[test]
fn test_width_selection_follows_the_65536_threshold() -> Result<(), ClearGbmError> {
    // count * bins below 65536 -> 16-bit entries; at or above -> 32-bit.
    assert_eq!(select_hist_width(3_usize, 4_usize), QuantHistWidth::B16);
    assert_eq!(select_hist_width(16383_usize, 4_usize), QuantHistWidth::B16);
    assert_eq!(select_hist_width(16384_usize, 4_usize), QuantHistWidth::B32);
    assert_eq!(
        select_hist_width(1_000_000_usize, 4_usize),
        QuantHistWidth::B32
    );
    Ok(())
}

#[test]
fn test_b16_build_accumulates_packed_sums_and_counts() -> Result<(), ClearGbmError> {
    // One feature, bins [0, 1, 1] over the three rows. Bin 0 holds row 0
    // (g 0, h 2); bin 1 holds rows 1 + 2 (g -1 + 2 = 1, h 2 + 4 = 6).
    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let stream = three_row_stream();
    let bins_rows = vec![0_u8, 1_u8, 1_u8];
    let out = build_quantized_node_histograms(QuantizedNodeHistogramRequest {
        sample_indices: &sample_indices,
        packed_int8: &stream,
        bins_rows: &bins_rows,
        n_features: 1_usize,
        n_bins: 2_usize,
        width: QuantHistWidth::B16,
    });
    let QuantizedNodeHistograms::B16(features) = out else {
        return Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected 16-bit histograms".to_string(),
        });
    };
    assert_eq!(features.len(), 1_usize);
    assert_eq!(
        unpack_acc16(features[0_usize][0_usize]),
        (0_i64, 2_i64, 1_usize)
    );
    assert_eq!(
        unpack_acc16(features[0_usize][1_usize]),
        (1_i64, 6_i64, 2_usize)
    );
    Ok(())
}

#[test]
fn test_b32_build_matches_the_b16_build_in_unpacked_space() -> Result<(), ClearGbmError> {
    // The same walk at 32-bit width must produce the same unpacked
    // triples — the width changes the storage, never the sums.
    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let stream = three_row_stream();
    let bins_rows = vec![0_u8, 1_u8, 1_u8];
    let request = QuantizedNodeHistogramRequest {
        sample_indices: &sample_indices,
        packed_int8: &stream,
        bins_rows: &bins_rows,
        n_features: 1_usize,
        n_bins: 2_usize,
        width: QuantHistWidth::B16,
    };
    let b16 = build_quantized_node_histograms(request);
    let b32 = build_quantized_node_histograms(QuantizedNodeHistogramRequest {
        width: QuantHistWidth::B32,
        ..request
    });
    assert_eq!(unpacked_features(&b16), unpacked_features(&b32));
    Ok(())
}

#[test]
fn test_multi_feature_build_carves_per_feature() -> Result<(), ClearGbmError> {
    // Two features with different bin columns: feature 0 sends all rows
    // to bin 0; feature 1 splits them 1/2.
    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let stream = three_row_stream();
    let bins_rows = vec![0_u8, 0_u8, 0_u8, 1_u8, 0_u8, 1_u8];
    let out = build_quantized_node_histograms(QuantizedNodeHistogramRequest {
        sample_indices: &sample_indices,
        packed_int8: &stream,
        bins_rows: &bins_rows,
        n_features: 2_usize,
        n_bins: 2_usize,
        width: QuantHistWidth::B16,
    });
    let unpacked = unpacked_features(&out);
    assert_eq!(unpacked[0_usize][0_usize], (1_i64, 8_i64, 3_usize));
    assert_eq!(unpacked[0_usize][1_usize], (0_i64, 0_i64, 0_usize));
    assert_eq!(unpacked[1_usize][0_usize], (0_i64, 2_i64, 1_usize));
    assert_eq!(unpacked[1_usize][1_usize], (1_i64, 6_i64, 2_usize));
    Ok(())
}

#[test]
fn test_b16_minus_b16_subtracts_packed_entries() -> Result<(), ClearGbmError> {
    let parent = QuantizedNodeHistograms::B16(vec![vec![
        QuantAcc16 {
            packed: (3_i32 << 16_u32) + 8_i32,
            count: 5_u32,
        },
        QuantAcc16 {
            packed: (-2_i32 << 16_u32) + 4_i32,
            count: 3_u32,
        },
    ]]);
    let child = QuantizedNodeHistograms::B16(vec![vec![
        QuantAcc16 {
            packed: (1_i32 << 16_u32) + 3_i32,
            count: 2_u32,
        },
        QuantAcc16 {
            packed: (-1_i32 << 16_u32) + 1_i32,
            count: 1_u32,
        },
    ]]);
    let sibling = propagate!(subtract_quantized(&parent, &child, QuantHistWidth::B16));
    let unpacked = unpacked_features(&sibling);
    assert_eq!(unpacked[0_usize][0_usize], (2_i64, 5_i64, 3_usize));
    assert_eq!(unpacked[0_usize][1_usize], (-1_i64, 3_i64, 2_usize));
    Ok(())
}

#[test]
fn test_b32_parent_minus_b16_child_lands_at_either_width() -> Result<(), ClearGbmError> {
    // Parent sums small enough that the sibling fits BOTH widths, so the
    // same subtraction can be checked at each landing. (In production
    // the landing width always comes from the sibling's own count.)
    let parent = QuantizedNodeHistograms::B32(vec![vec![QuantAcc32 {
        packed: (1_000_i64 << 32_u32) + 2_000_i64,
        count: 900_u32,
    }]]);
    let child = QuantizedNodeHistograms::B16(vec![vec![QuantAcc16 {
        packed: (7_i32 << 16_u32) + 9_i32,
        count: 4_u32,
    }]]);
    let at32 = propagate!(subtract_quantized(&parent, &child, QuantHistWidth::B32));
    let at16 = propagate!(subtract_quantized(&parent, &child, QuantHistWidth::B16));
    let expected = (993_i64, 1_991_i64, 896_usize);
    assert_eq!(unpacked_features(&at32)[0_usize][0_usize], expected);
    assert_eq!(unpacked_features(&at16)[0_usize][0_usize], expected);
    let QuantizedNodeHistograms::B16(_) = at16 else {
        return Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected a 16-bit sibling".to_string(),
        });
    };
    Ok(())
}

#[test]
fn test_b32_minus_b32_subtracts_in_unpacked_space() -> Result<(), ClearGbmError> {
    let parent = QuantizedNodeHistograms::B32(vec![vec![QuantAcc32 {
        packed: (-50_000_i64 << 32_u32) + 70_000_i64,
        count: 40_000_u32,
    }]]);
    let child = QuantizedNodeHistograms::B32(vec![vec![QuantAcc32 {
        packed: (-20_000_i64 << 32_u32) + 30_000_i64,
        count: 17_000_u32,
    }]]);
    let sibling = propagate!(subtract_quantized(&parent, &child, QuantHistWidth::B32));
    assert_eq!(
        unpacked_features(&sibling)[0_usize][0_usize],
        (-30_000_i64, 40_000_i64, 23_000_usize)
    );
    Ok(())
}

#[test]
fn test_b32_minus_b32_can_land_at_16_bits() -> Result<(), ClearGbmError> {
    // A 32-bit parent whose sibling count selects 16 bits: the
    // subtraction goes through the unpack path and repacks narrow.
    let parent = QuantizedNodeHistograms::B32(vec![vec![QuantAcc32 {
        packed: (300_i64 << 32_u32) + 500_i64,
        count: 200_u32,
    }]]);
    let child = QuantizedNodeHistograms::B32(vec![vec![QuantAcc32 {
        packed: (100_i64 << 32_u32) + 150_i64,
        count: 80_u32,
    }]]);
    let sibling = propagate!(subtract_quantized(&parent, &child, QuantHistWidth::B16));
    let QuantizedNodeHistograms::B16(_) = sibling else {
        return Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected a 16-bit sibling".to_string(),
        });
    };
    assert_eq!(
        unpacked_features(&sibling)[0_usize][0_usize],
        (200_i64, 350_i64, 120_usize)
    );
    Ok(())
}

#[test]
fn test_wider_child_under_narrow_parent_is_refused() -> Result<(), ClearGbmError> {
    let parent = QuantizedNodeHistograms::B16(vec![vec![QuantAcc16::ZERO]]);
    let child = QuantizedNodeHistograms::B32(vec![vec![QuantAcc32::ZERO]]);
    match subtract_quantized(&parent, &child, QuantHistWidth::B16) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a 32-bit child under a 16-bit parent must be refused".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_wide_sibling_under_narrow_parent_is_refused() -> Result<(), ClearGbmError> {
    let parent = QuantizedNodeHistograms::B16(vec![vec![QuantAcc16::ZERO]]);
    let child = QuantizedNodeHistograms::B16(vec![vec![QuantAcc16::ZERO]]);
    match subtract_quantized(&parent, &child, QuantHistWidth::B32) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a 32-bit sibling under a 16-bit parent must be refused".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_unpack_recovers_negative_gradients_exactly() -> Result<(), ClearGbmError> {
    // The arithmetic shift must recover the sign; the mask must recover
    // the hessian untouched by the negative high half.
    let acc16 = QuantAcc16 {
        packed: (-3_i32 << 16_u32) + 5_i32,
        count: 2_u32,
    };
    assert_eq!(unpack_acc16(acc16), (-3_i64, 5_i64, 2_usize));
    let acc32 = QuantAcc32 {
        packed: (-70_000_i64 << 32_u32) + 90_000_i64,
        count: 7_u32,
    };
    assert_eq!(unpack_acc32(acc32), (-70_000_i64, 90_000_i64, 7_usize));
    Ok(())
}
