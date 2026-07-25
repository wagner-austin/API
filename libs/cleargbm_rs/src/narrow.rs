//! Index width conversion for the perf-critical histogram inputs.
//!
//! `sample_indices` are stored as `u32` to halve cache-line pressure versus
//! `usize` on 64-bit targets (LightGBM's `data_size_t = int32` pattern), so
//! every slice access widens back to `usize`. This module holds that one
//! conversion.
//!
//! It contains no `as` casts. Gradients and hessians are `f64` end to end:
//! narrowing them to `f32` was measured 8% SLOWER on this workload, because
//! both widths already fit in L2 at the node sizes reached here, so there is
//! no bandwidth to save and each element pays a widening conversion before
//! its accumulate. See the wiki page `cleargbm-f32-score-narrowing-reverted`.

/// Compile-time guarantee that `u32` always fits in `usize`.
///
/// Makes the error arm of [`index_widen`] statically dead rather than a
/// runtime fallback: a target with a narrower `usize` fails to build here
/// instead of silently truncating sample indices at run time.
const _: () = assert!(usize::BITS >= u32::BITS);

/// Widens a `u32` sample index to `usize` for slice/vec indexing.
///
/// `sample_indices` are stored as `u32` to halve cache-line pressure vs
/// `usize` on 64-bit targets (LightGBM's `data_size_t = int32` pattern).
///
/// Uses `try_from` rather than an `as` cast, which costs nothing and is
/// strictly safer. Verified by comparing emitted assembly at `-O`: LLVM
/// proves the conversion infallible and compiles this to the identical
/// single `movl` that `idx as usize` produces — rustc aliases the two
/// symbols outright — and the histogram hot loop emits the same instruction
/// sequence either way. An `as` cast, by contrast, would silently truncate
/// on a target with a narrower `usize`; the static assertion above plus
/// `try_from` make that case a build failure instead.
#[inline]
#[must_use]
pub fn index_widen(idx: u32) -> usize {
    // `unwrap_or` rather than a `match`: the error arm is statically dead
    // (see the assertion above), so writing it as a branch here would leave a
    // permanently uncoverable segment in a crate that requires 100% coverage.
    usize::try_from(idx).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {

    #[test]
    fn index_widen_zero() -> Result<(), crate::error::ClearGbmError> {
        assert!(super::index_widen(0_u32) == 0_usize);
        Ok(())
    }

    #[test]
    fn index_widen_max_u32() -> Result<(), crate::error::ClearGbmError> {
        // The module's const assertion guarantees usize >= 32 bits, so u32::MAX
        // always widens exactly. The expected value is spelled out rather than
        // computed with `usize::try_from`, so the assertion does not restate
        // the conversion it is checking.
        assert!(super::index_widen(u32::MAX) == 4_294_967_295_usize);
        Ok(())
    }

    #[test]
    fn index_widen_mid_range() -> Result<(), crate::error::ClearGbmError> {
        assert!(super::index_widen(1_000_000_u32) == 1_000_000_usize);
        Ok(())
    }
}
