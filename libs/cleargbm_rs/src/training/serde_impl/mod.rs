//! Manual serde implementations for training types.
//!
//! These implementations avoid the `?` operator per project rules and follow
//! the `SplitResult` / `Tree` pattern established elsewhere in the crate.
//!
//! Deserialization routes through the validating constructors
//! (`GradientBoostingConfig::new`, `GradientBoostingModel::new`) so an
//! inbound JSON payload is checked before it becomes a live value.
//!
//! # Module Structure
//!
//! - this module — the wire enums ([`GrowthStrategy`], [`Objective`]) that
//!   serialize as their single spelling on both boundaries
//! - [`config`] — `GradientBoostingConfig` (de)serialization
//! - [`model`] — `GradientBoostingModel` (de)serialization

mod config;
mod model;

#[cfg(test)]
pub(crate) use config::GradientBoostingConfigFieldVisitor;
#[cfg(test)]
pub(crate) use model::GradientBoostingModelFieldVisitor;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::config::{GrowthStrategy, Objective};

// =============================================================================
// GrowthStrategy Serialization
// =============================================================================

impl Serialize for GrowthStrategy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Visitor for deserializing `GrowthStrategy` from its wire spelling.
///
/// `pub(crate)` so [`crate::training::tests`] can drive its `expecting`
/// formatter directly, matching the convention used by the field visitors
/// in the sibling modules.
pub(crate) struct GrowthStrategyVisitor;

impl<'de> Visitor<'de> for GrowthStrategyVisitor {
    type Value = GrowthStrategy;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("\"depth_wise\" or \"leaf_wise\"")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match GrowthStrategy::from_wire(value) {
            Ok(strategy) => Ok(strategy),
            Err(e) => Err(E::custom(e.to_string())),
        }
    }
}

impl<'de> Deserialize<'de> for GrowthStrategy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(GrowthStrategyVisitor)
    }
}

// =============================================================================
// Objective Serialization
// =============================================================================

impl Serialize for Objective {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Visitor for deserializing `Objective` from its wire spelling.
///
/// `pub(crate)` so [`crate::training::tests`] can drive its `expecting`
/// formatter directly, matching [`GrowthStrategyVisitor`].
pub(crate) struct ObjectiveVisitor;

impl<'de> Visitor<'de> for ObjectiveVisitor {
    type Value = Objective;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("\"binary_log_loss\" or \"squared_error\"")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match Objective::from_wire(value) {
            Ok(objective) => Ok(objective),
            Err(e) => Err(E::custom(e.to_string())),
        }
    }
}

impl<'de> Deserialize<'de> for Objective {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ObjectiveVisitor)
    }
}
