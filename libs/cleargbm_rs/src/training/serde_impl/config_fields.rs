//! Field identifiers for `GradientBoostingConfig` deserialization.
//!
//! Split from `config.rs` so the field ledger and the visitor body each
//! stay within the file-size discipline. The ledger order IS the wire
//! format: appending a field is a serde break, and every break is an
//! artifact retrain round by policy.

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer};

/// Field identifiers for `GradientBoostingConfig` deserialization.
///
/// `pub(crate)` because it is the `Value` type of the `pub(crate)`
/// [`GradientBoostingConfigFieldVisitor`].
pub(crate) enum GradientBoostingConfigField {
    /// The number of boosting iterations.
    NEstimators,
    /// The maximum tree depth.
    MaxDepth,
    /// The learning rate.
    LearningRate,
    /// The minimum samples required to split a node.
    MinSamplesSplit,
    /// The minimum samples per leaf.
    MinSamplesLeaf,
    /// The maximum number of histogram bins per feature.
    MaxBins,
    /// The row-subsampling fraction.
    Subsample,
    /// The random seed.
    RandomState,
    /// The per-feature monotonic constraints (optional).
    MonotonicConstraints,
    /// The L1 regularization term.
    RegAlpha,
    /// The L2 regularization term.
    RegLambda,
    /// The early stopping patience (optional).
    EarlyStoppingRounds,
    /// The tree growth policy.
    GrowthStrategy,
    /// The leaf budget (optional; set exactly under leaf-wise growth).
    NumLeaves,
    /// The training objective.
    Objective,
    /// The positive-class weight (optional; set exactly under binary log
    /// loss).
    ScalePosWeight,
    /// The per-split feature budget (optional).
    MaxFeatures,
    /// The per-tree feature fraction (optional).
    ColsampleBytree,
    /// The categorical feature indices (optional).
    CategoricalFeatures,
    /// The class count (optional).
    NClasses,
    /// The NDCG truncation position (optional).
    LambdarankTruncationLevel,
    /// The GOSS top rate (optional).
    GossTopRate,
    /// The GOSS other rate (optional).
    GossOtherRate,
    /// The quantized-training bin count (optional).
    QuantizedGradientBins,
}

/// Visitor for deserializing `GradientBoostingConfigField` from string.
///
/// `pub(crate)` so [`crate::training::tests`] can drive its `expecting`
/// formatter directly, matching the convention in [`crate::types::serde_impl`].
pub(crate) struct GradientBoostingConfigFieldVisitor;

impl<'de> Visitor<'de> for GradientBoostingConfigFieldVisitor {
    type Value = GradientBoostingConfigField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "n_estimators" => Ok(GradientBoostingConfigField::NEstimators),
            "max_depth" => Ok(GradientBoostingConfigField::MaxDepth),
            "learning_rate" => Ok(GradientBoostingConfigField::LearningRate),
            "min_samples_split" => Ok(GradientBoostingConfigField::MinSamplesSplit),
            "min_samples_leaf" => Ok(GradientBoostingConfigField::MinSamplesLeaf),
            "max_bins" => Ok(GradientBoostingConfigField::MaxBins),
            "subsample" => Ok(GradientBoostingConfigField::Subsample),
            "random_state" => Ok(GradientBoostingConfigField::RandomState),
            "monotonic_constraints" => Ok(GradientBoostingConfigField::MonotonicConstraints),
            "reg_alpha" => Ok(GradientBoostingConfigField::RegAlpha),
            "reg_lambda" => Ok(GradientBoostingConfigField::RegLambda),
            "early_stopping_rounds" => Ok(GradientBoostingConfigField::EarlyStoppingRounds),
            "growth_strategy" => Ok(GradientBoostingConfigField::GrowthStrategy),
            "num_leaves" => Ok(GradientBoostingConfigField::NumLeaves),
            "objective" => Ok(GradientBoostingConfigField::Objective),
            "scale_pos_weight" => Ok(GradientBoostingConfigField::ScalePosWeight),
            "max_features" => Ok(GradientBoostingConfigField::MaxFeatures),
            "colsample_bytree" => Ok(GradientBoostingConfigField::ColsampleBytree),
            "categorical_features" => Ok(GradientBoostingConfigField::CategoricalFeatures),
            "n_classes" => Ok(GradientBoostingConfigField::NClasses),
            "lambdarank_truncation_level" => {
                Ok(GradientBoostingConfigField::LambdarankTruncationLevel)
            }
            "goss_top_rate" => Ok(GradientBoostingConfigField::GossTopRate),
            "goss_other_rate" => Ok(GradientBoostingConfigField::GossOtherRate),
            "quantized_gradient_bins" => Ok(GradientBoostingConfigField::QuantizedGradientBins),
            _ => Err(E::unknown_field(value, GRADIENT_BOOSTING_CONFIG_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for GradientBoostingConfigField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(GradientBoostingConfigFieldVisitor)
    }
}

/// Field names for `GradientBoostingConfig` serialization.
pub(crate) const GRADIENT_BOOSTING_CONFIG_FIELDS: &[&str] = &[
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_samples_split",
    "min_samples_leaf",
    "max_bins",
    "subsample",
    "random_state",
    "monotonic_constraints",
    "reg_alpha",
    "reg_lambda",
    "early_stopping_rounds",
    "growth_strategy",
    "num_leaves",
    "objective",
    "scale_pos_weight",
    "max_features",
    "colsample_bytree",
    "categorical_features",
    "n_classes",
    "lambdarank_truncation_level",
    "goss_top_rate",
    "goss_other_rate",
    "quantized_gradient_bins",
];
