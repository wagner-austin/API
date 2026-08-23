//! Manual serde implementation for [`GradientBoostingModel`].
//!
//! The model persists six fields; everything objective-dependent (how a raw
//! score reads, whether probabilities exist) is answered by the embedded
//! config's `objective` tag rather than duplicated at the model level. The
//! base score has two mutually exclusive spellings — a scalar for the
//! single-score objectives, one score per class for multiclass — and
//! reassembly enforces the pairing against the config's objective.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::tree::Tree;

use super::super::config::GradientBoostingConfig;
use super::super::model::GradientBoostingModel;

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
        match state.serialize_field("class_base_predictions", &self.class_base_predictions()) {
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
    /// The objective's scalar base score (single-score models; null
    /// otherwise).
    BasePrediction,
    /// The per-class base scores (multiclass models; null otherwise).
    ClassBasePredictions,
    /// The learning rate.
    LearningRate,
    /// The feature names captured at training time.
    FeatureNames,
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
            "class_base_predictions" => Ok(GradientBoostingModelField::ClassBasePredictions),
            "learning_rate" => Ok(GradientBoostingModelField::LearningRate),
            "feature_names" => Ok(GradientBoostingModelField::FeatureNames),
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
    "class_base_predictions",
    "learning_rate",
    "feature_names",
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
                let mut base_prediction: Option<Option<f64>> = None;
                let mut class_base_predictions: Option<Option<Vec<f64>>> = None;
                let mut learning_rate: Option<f64> = None;
                let mut feature_names: Option<Vec<String>> = None;
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
                        GradientBoostingModelField::ClassBasePredictions => {
                            class_base_predictions = Some(match map.next_value() {
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
                let class_base_predictions = match class_base_predictions {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("class_base_predictions")),
                };
                let learning_rate = match learning_rate {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("learning_rate")),
                };
                let feature_names = match feature_names {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("feature_names")),
                };
                let config = match config {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("config")),
                };

                match GradientBoostingModel::from_parts(
                    trees,
                    base_prediction,
                    class_base_predictions,
                    learning_rate,
                    feature_names,
                    config,
                ) {
                    Ok(model) => Ok(model),
                    Err(e) => Err(de::Error::custom(e.to_string())),
                }
            }
        }

        deserializer.deserialize_struct(
            "GradientBoostingModel",
            GRADIENT_BOOSTING_MODEL_FIELDS,
            GradientBoostingModelVisitor,
        )
    }
}
