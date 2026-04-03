//! Early stopping state machine for gradient boosting training.
//!
//! Tracks validation loss across rounds and signals when training should
//! stop due to lack of improvement.

/// Tracks validation loss and determines when to stop training.
///
/// Matches the Python `_EarlyStoppingState` class exactly.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EarlyStoppingState {
    /// Best validation loss observed so far.
    best_val_loss: f64,
    /// Round index where the best loss was observed.
    best_round: usize,
    /// Number of consecutive rounds without improvement.
    rounds_without_improvement: usize,
    /// Maximum rounds without improvement before stopping.
    patience: usize,
}

impl EarlyStoppingState {
    /// Creates a new early stopping tracker.
    ///
    /// # Args
    ///
    /// * `patience` - Number of rounds without improvement before stopping.
    pub(crate) fn new(patience: usize) -> Self {
        Self {
            best_val_loss: f64::INFINITY,
            best_round: 0_usize,
            rounds_without_improvement: 0_usize,
            patience,
        }
    }

    /// Updates state with a new validation loss.
    ///
    /// Returns `true` if training should stop (patience exhausted).
    ///
    /// # Args
    ///
    /// * `val_loss` - Validation loss for the current round.
    /// * `round` - Current round index (0-based).
    pub(crate) fn update(&mut self, val_loss: f64, round: usize) -> bool {
        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.best_round = round;
            self.rounds_without_improvement = 0_usize;
        } else {
            self.rounds_without_improvement += 1_usize;
        }
        self.rounds_without_improvement >= self.patience
    }

    /// Returns the round index where the best validation loss was observed.
    #[must_use]
    pub(crate) fn best_round(&self) -> usize {
        self.best_round
    }
}
