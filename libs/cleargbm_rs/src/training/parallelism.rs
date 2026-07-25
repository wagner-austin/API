//! Thread-count policy for a training run.
//!
//! Kept out of [`super::config::GradientBoostingConfig`] deliberately. The
//! thread count does not change the fitted model, and the config is
//! serialized into the saved model — a model reloaded on a different machine
//! must not carry the thread count it happened to be trained with.

use core::num::NonZeroUsize;

use crate::error::ClearGbmError;

/// How many OS threads a training run may use.
///
/// Built from scikit-learn's `n_jobs` convention via
/// [`Parallelism::from_n_jobs`], which is what callers pass through the Python
/// boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parallelism {
    /// Exactly one worker thread.
    ///
    /// Distinct from [`Parallelism::Fixed`] with one thread only in intent;
    /// both build a single-worker pool, so per-feature histogram dispatch
    /// stays on one core and the run is reproducible under contention.
    Single,

    /// A fixed number of worker threads.
    Fixed(NonZeroUsize),

    /// One worker thread per available core.
    AllCores,
}

impl Parallelism {
    /// Interprets scikit-learn's `n_jobs` convention.
    ///
    /// `1` means single-threaded, `-1` means one thread per core, and any
    /// `n > 1` means exactly `n` threads.
    ///
    /// # Args
    ///
    /// * `n_jobs` - The requested worker count.
    ///
    /// # Returns
    ///
    /// The corresponding policy.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` for `0` and for negative
    /// values other than `-1`. Those are rejected rather than rounded to a
    /// nearby legal value, so a typo surfaces instead of silently changing
    /// how many cores the run takes.
    ///
    /// Counts above the target's addressable thread count are *not* an error:
    /// they saturate, since "more threads than the machine can name" can only
    /// mean "as many as possible".
    pub fn from_n_jobs(n_jobs: i64) -> Result<Self, ClearGbmError> {
        if n_jobs == -1_i64 {
            return Ok(Self::AllCores);
        }
        if n_jobs == 1_i64 {
            return Ok(Self::Single);
        }
        if n_jobs < 1_i64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "n_jobs".to_string(),
                reason: format!("must be -1 (all cores) or >= 1, got {n_jobs}"),
            });
        }
        // `n_jobs >= 2` here, so both conversions below are infallible in
        // practice. They are written as saturating expressions rather than
        // `match` arms for the same reason as [`crate::narrow::index_widen`]:
        // a branch whose error arm cannot be reached would be a permanently
        // uncoverable segment in a crate that requires 100% coverage, and a
        // saturating conversion is the honest reading anyway — a request for
        // more threads than the target can address means "as many as
        // possible", not "fail".
        let count = usize::try_from(n_jobs).unwrap_or(usize::MAX);
        Ok(Self::Fixed(
            NonZeroUsize::new(count).unwrap_or(NonZeroUsize::MIN),
        ))
    }

    /// Resolves the policy to a concrete worker-thread count.
    ///
    /// # Returns
    ///
    /// The number of worker threads to build the pool with. `AllCores`
    /// resolves against the machine's reported parallelism, falling back to a
    /// single thread only when the platform cannot report it.
    #[must_use]
    pub fn thread_count(self) -> NonZeroUsize {
        match self {
            Self::Single => NonZeroUsize::MIN,
            Self::Fixed(count) => count,
            Self::AllCores => std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
        }
    }
}
