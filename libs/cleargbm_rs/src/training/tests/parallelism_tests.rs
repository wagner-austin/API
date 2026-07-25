//! Tests for the worker-thread policy.
//!
//! [`Parallelism`] is the only thing standing between a caller's `n_jobs` and
//! the rayon pool the boosting loop runs inside, so these cover both the
//! scikit-learn conventions it must honour and the values it must reject.

use core::num::NonZeroUsize;

use crate::error::ClearGbmError;
use crate::training::Parallelism;

/// Builds a `NonZeroUsize` for an expected-value assertion.
///
/// # Args
///
/// * `value` - A count the caller knows to be nonzero.
///
/// # Returns
///
/// `value` as a [`NonZeroUsize`], saturating at one if the caller passed zero.
fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap_or(NonZeroUsize::MIN)
}

#[test]
fn one_job_is_single_threaded() -> Result<(), ClearGbmError> {
    let policy = match Parallelism::from_n_jobs(1_i64) {
        Ok(value) => value,
        Err(e) => return Err(e),
    };
    assert_eq!(policy, Parallelism::Single);
    assert_eq!(policy.thread_count().get(), 1_usize);
    Ok(())
}

#[test]
fn negative_one_means_all_cores() -> Result<(), ClearGbmError> {
    let policy = match Parallelism::from_n_jobs(-1_i64) {
        Ok(value) => value,
        Err(e) => return Err(e),
    };
    assert_eq!(policy, Parallelism::AllCores);
    // Resolves against the machine, so assert the contract rather than a
    // specific core count: at least one worker, and no more than the number
    // the platform reports.
    let reported = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    assert_eq!(policy.thread_count(), reported);
    Ok(())
}

#[test]
fn explicit_count_is_honoured() -> Result<(), ClearGbmError> {
    let policy = match Parallelism::from_n_jobs(4_i64) {
        Ok(value) => value,
        Err(e) => return Err(e),
    };
    assert_eq!(policy, Parallelism::Fixed(nonzero(4_usize)));
    assert_eq!(policy.thread_count().get(), 4_usize);
    Ok(())
}

#[test]
fn two_is_the_smallest_fixed_count() -> Result<(), ClearGbmError> {
    // The boundary between the `Single` short-circuit and the `Fixed` path.
    let policy = match Parallelism::from_n_jobs(2_i64) {
        Ok(value) => value,
        Err(e) => return Err(e),
    };
    assert_eq!(policy, Parallelism::Fixed(nonzero(2_usize)));
    assert_eq!(policy.thread_count().get(), 2_usize);
    Ok(())
}

#[test]
fn absurdly_large_counts_saturate_rather_than_fail() -> Result<(), ClearGbmError> {
    // Documented behaviour: a request beyond the target's addressable thread
    // count means "as many as possible", not an error.
    let policy = match Parallelism::from_n_jobs(i64::MAX) {
        Ok(value) => value,
        Err(e) => return Err(e),
    };
    assert!(policy.thread_count().get() >= 2_usize);
    Ok(())
}

#[test]
fn zero_is_rejected() -> Result<(), ClearGbmError> {
    assert!(Parallelism::from_n_jobs(0_i64).is_err());
    Ok(())
}

#[test]
fn negative_other_than_minus_one_is_rejected() -> Result<(), ClearGbmError> {
    assert!(Parallelism::from_n_jobs(-2_i64).is_err());
    Ok(())
}

#[test]
fn rejection_names_the_parameter() -> Result<(), ClearGbmError> {
    match Parallelism::from_n_jobs(0_i64) {
        Ok(_) => Err(ClearGbmError::InvalidParameter {
            name: "test".to_string(),
            reason: "zero must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "n_jobs");
            assert!(reason.contains("got 0"));
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn single_and_fixed_one_are_distinct_values() -> Result<(), ClearGbmError> {
    // Both resolve to one worker, but they are separate variants; a future
    // change that collapsed them would silently lose the caller's intent.
    assert_ne!(Parallelism::Single, Parallelism::Fixed(nonzero(1_usize)));
    assert_eq!(
        Parallelism::Single.thread_count(),
        Parallelism::Fixed(nonzero(1_usize)).thread_count()
    );
    Ok(())
}
