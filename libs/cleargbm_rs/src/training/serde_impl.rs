//! Manual serde implementations for training types.
//!
//! These implementations avoid the `?` operator per project rules and follow the
//! `SplitResult` / `Tree` pattern established elsewhere in the crate.
//!
//! Deserialization routes through the validating constructors
//! (`GradientBoostingConfig::new`, `GradientBoostingModel::new`) so an
//! inbound JSON payload is checked before it becomes a live value.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::split::MonotonicConstraint;
use crate::tree::Tree;

use super::config::{GradientBoostingConfig, GradientBoostingConfigParams, GrowthStrategy};
use super::model::GradientBoostingModel;

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
/// below.
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
// GradientBoostingConfig Serialization
// =============================================================================

impl Serialize for GradientBoostingConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("GradientBoostingConfig", 14) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("n_estimators", &self.n_estimators()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("max_depth", &self.max_depth()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("learning_rate", &self.learning_rate()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("min_samples_split", &self.min_samples_split()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("min_samples_leaf", &self.min_samples_leaf()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("max_bins", &self.max_bins()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("subsample", &self.subsample()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("random_state", &self.random_state()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        let mc_vec: Option<Vec<MonotonicConstraint>> = self
            .monotonic_constraints()
            .map(<[MonotonicConstraint]>::to_vec);
        match state.serialize_field("monotonic_constraints", &mc_vec) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("reg_alpha", &self.reg_alpha()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("reg_lambda", &self.reg_lambda()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("early_stopping_rounds", &self.early_stopping_rounds()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("growth_strategy", &self.growth_strategy()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("num_leaves", &self.num_leaves()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

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
const GRADIENT_BOOSTING_CONFIG_FIELDS: &[&str] = &[
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
];

impl<'de> Deserialize<'de> for GradientBoostingConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct GradientBoostingConfigVisitor;

        impl<'de> Visitor<'de> for GradientBoostingConfigVisitor {
            type Value = GradientBoostingConfig;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct GradientBoostingConfig")
            }

            fn visit_map<V>(self, mut map: V) -> Result<GradientBoostingConfig, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut n_estimators: Option<usize> = None;
                let mut max_depth: Option<usize> = None;
                let mut learning_rate: Option<f64> = None;
                let mut min_samples_split: Option<usize> = None;
                let mut min_samples_leaf: Option<usize> = None;
                let mut max_bins: Option<usize> = None;
                let mut subsample: Option<f64> = None;
                let mut random_state: Option<u64> = None;
                let mut monotonic_constraints: Option<Option<Vec<MonotonicConstraint>>> = None;
                let mut reg_alpha: Option<f64> = None;
                let mut reg_lambda: Option<f64> = None;
                let mut early_stopping_rounds: Option<Option<usize>> = None;
                let mut growth_strategy: Option<GrowthStrategy> = None;
                let mut num_leaves: Option<Option<usize>> = None;

                loop {
                    let key: Option<GradientBoostingConfigField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        GradientBoostingConfigField::NEstimators => {
                            n_estimators = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::MaxDepth => {
                            max_depth = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::LearningRate => {
                            learning_rate = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::MinSamplesSplit => {
                            min_samples_split = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::MinSamplesLeaf => {
                            min_samples_leaf = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::MaxBins => {
                            max_bins = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::Subsample => {
                            subsample = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::RandomState => {
                            random_state = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::MonotonicConstraints => {
                            monotonic_constraints = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::RegAlpha => {
                            reg_alpha = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::RegLambda => {
                            reg_lambda = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::EarlyStoppingRounds => {
                            early_stopping_rounds = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::GrowthStrategy => {
                            growth_strategy = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingConfigField::NumLeaves => {
                            num_leaves = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let n_estimators = match n_estimators {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("n_estimators")),
                };
                let max_depth = match max_depth {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("max_depth")),
                };
                let learning_rate = match learning_rate {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("learning_rate")),
                };
                let min_samples_split = match min_samples_split {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("min_samples_split")),
                };
                let min_samples_leaf = match min_samples_leaf {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("min_samples_leaf")),
                };
                let max_bins = match max_bins {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("max_bins")),
                };
                let subsample = match subsample {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("subsample")),
                };
                let random_state = match random_state {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("random_state")),
                };
                let monotonic_constraints = match monotonic_constraints {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("monotonic_constraints")),
                };
                let reg_alpha = match reg_alpha {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("reg_alpha")),
                };
                let reg_lambda = match reg_lambda {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("reg_lambda")),
                };
                let early_stopping_rounds = match early_stopping_rounds {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("early_stopping_rounds")),
                };
                let growth_strategy = match growth_strategy {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("growth_strategy")),
                };
                let num_leaves = match num_leaves {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("num_leaves")),
                };

                let params = GradientBoostingConfigParams {
                    n_estimators,
                    max_depth,
                    learning_rate,
                    min_samples_split,
                    min_samples_leaf,
                    max_bins,
                    subsample,
                    random_state,
                    monotonic_constraints,
                    reg_alpha,
                    reg_lambda,
                    early_stopping_rounds,
                    growth_strategy,
                    num_leaves,
                };
                match GradientBoostingConfig::new(params) {
                    Ok(cfg) => Ok(cfg),
                    Err(e) => Err(de::Error::custom(e.to_string())),
                }
            }
        }

        deserializer.deserialize_struct(
            "GradientBoostingConfig",
            GRADIENT_BOOSTING_CONFIG_FIELDS,
            GradientBoostingConfigVisitor,
        )
    }
}

// =============================================================================
// GradientBoostingModel Serialization
// =============================================================================

impl Serialize for GradientBoostingModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("GradientBoostingModel", 6) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("trees", &self.trees()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("base_prediction", &self.base_prediction()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("learning_rate", &self.learning_rate()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("feature_names", &self.feature_names()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("n_classes", &self.n_classes()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("config", &self.config()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `GradientBoostingModel` deserialization.
///
/// `pub(crate)` because it is the `Value` type of the `pub(crate)`
/// [`GradientBoostingModelFieldVisitor`].
pub(crate) enum GradientBoostingModelField {
    /// The trained decision trees.
    Trees,
    /// The base log-odds prediction.
    BasePrediction,
    /// The learning rate.
    LearningRate,
    /// The feature names captured at training time.
    FeatureNames,
    /// The number of classes.
    NClasses,
    /// The training configuration.
    Config,
}

/// Visitor for deserializing `GradientBoostingModelField` from string.
///
/// `pub(crate)` so [`crate::training::tests`] can drive its `expecting`
/// formatter directly, matching the convention in [`crate::types::serde_impl`].
pub(crate) struct GradientBoostingModelFieldVisitor;

impl<'de> Visitor<'de> for GradientBoostingModelFieldVisitor {
    type Value = GradientBoostingModelField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "trees" => Ok(GradientBoostingModelField::Trees),
            "base_prediction" => Ok(GradientBoostingModelField::BasePrediction),
            "learning_rate" => Ok(GradientBoostingModelField::LearningRate),
            "feature_names" => Ok(GradientBoostingModelField::FeatureNames),
            "n_classes" => Ok(GradientBoostingModelField::NClasses),
            "config" => Ok(GradientBoostingModelField::Config),
            _ => Err(E::unknown_field(value, GRADIENT_BOOSTING_MODEL_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for GradientBoostingModelField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(GradientBoostingModelFieldVisitor)
    }
}

/// Field names for `GradientBoostingModel` serialization.
const GRADIENT_BOOSTING_MODEL_FIELDS: &[&str] = &[
    "trees",
    "base_prediction",
    "learning_rate",
    "feature_names",
    "n_classes",
    "config",
];

impl<'de> Deserialize<'de> for GradientBoostingModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct GradientBoostingModelVisitor;

        impl<'de> Visitor<'de> for GradientBoostingModelVisitor {
            type Value = GradientBoostingModel;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct GradientBoostingModel")
            }

            fn visit_map<V>(self, mut map: V) -> Result<GradientBoostingModel, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut trees: Option<Vec<Tree>> = None;
                let mut base_prediction: Option<f64> = None;
                let mut learning_rate: Option<f64> = None;
                let mut feature_names: Option<Vec<String>> = None;
                let mut n_classes: Option<usize> = None;
                let mut config: Option<GradientBoostingConfig> = None;

                loop {
                    let key: Option<GradientBoostingModelField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        GradientBoostingModelField::Trees => {
                            trees = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingModelField::BasePrediction => {
                            base_prediction = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingModelField::LearningRate => {
                            learning_rate = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingModelField::FeatureNames => {
                            feature_names = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingModelField::NClasses => {
                            n_classes = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        GradientBoostingModelField::Config => {
                            config = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let trees = match trees {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("trees")),
                };
                let base_prediction = match base_prediction {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("base_prediction")),
                };
                let learning_rate = match learning_rate {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("learning_rate")),
                };
                let feature_names = match feature_names {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("feature_names")),
                };
                let n_classes = match n_classes {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("n_classes")),
                };
                let config = match config {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("config")),
                };

                Ok(GradientBoostingModel::new(
                    trees,
                    base_prediction,
                    learning_rate,
                    feature_names,
                    n_classes,
                    config,
                ))
            }
        }

        deserializer.deserialize_struct(
            "GradientBoostingModel",
            GRADIENT_BOOSTING_MODEL_FIELDS,
            GradientBoostingModelVisitor,
        )
    }
}
