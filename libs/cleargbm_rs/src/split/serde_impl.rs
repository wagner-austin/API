//! Manual serde implementations for split types.
//!
//! These implementations avoid the `?` operator per project rules.

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{MonotonicConstraint, NanDirection, SplitResult, SplitResultConfig};

// =============================================================================
// MonotonicConstraint Serialization
// =============================================================================

impl Serialize for MonotonicConstraint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::None => serializer.serialize_str("None"),
            Self::Increasing => serializer.serialize_str("Increasing"),
            Self::Decreasing => serializer.serialize_str("Decreasing"),
        }
    }
}

/// Visitor for deserializing `MonotonicConstraint` from string.
struct MonotonicConstraintVisitor;

impl<'de> Visitor<'de> for MonotonicConstraintVisitor {
    type Value = MonotonicConstraint;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("\"None\", \"Increasing\", or \"Decreasing\"")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "None" => Ok(MonotonicConstraint::None),
            "Increasing" => Ok(MonotonicConstraint::Increasing),
            "Decreasing" => Ok(MonotonicConstraint::Decreasing),
            _ => Err(E::custom(format!(
                "unknown MonotonicConstraint variant: {value}"
            ))),
        }
    }
}

impl<'de> Deserialize<'de> for MonotonicConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(MonotonicConstraintVisitor)
    }
}

// =============================================================================
// NanDirection Serialization
// =============================================================================

impl Serialize for NanDirection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Left => serializer.serialize_str("Left"),
            Self::Right => serializer.serialize_str("Right"),
        }
    }
}

/// Visitor for deserializing `NanDirection` from string.
struct NanDirectionVisitor;

impl<'de> Visitor<'de> for NanDirectionVisitor {
    type Value = NanDirection;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("\"Left\" or \"Right\"")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "Left" => Ok(NanDirection::Left),
            "Right" => Ok(NanDirection::Right),
            _ => Err(E::custom(format!("unknown NanDirection variant: {value}"))),
        }
    }
}

impl<'de> Deserialize<'de> for NanDirection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(NanDirectionVisitor)
    }
}

// =============================================================================
// SplitResult Serialization
// =============================================================================

impl Serialize for SplitResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("SplitResult", 10) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("feature_index", &self.feature_index()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("split_bin", &self.split_bin()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("gain", &self.gain()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("left_gradient_sum", &self.left_gradient_sum()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("left_hessian_sum", &self.left_hessian_sum()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("left_count", &self.left_count()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("right_gradient_sum", &self.right_gradient_sum()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("right_hessian_sum", &self.right_hessian_sum()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("right_count", &self.right_count()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("nan_direction", &self.nan_direction()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `SplitResult` deserialization.
enum SplitResultField {
    /// The feature index field.
    FeatureIndex,
    /// The split bin field.
    SplitBin,
    /// The gain field.
    Gain,
    /// The left gradient sum field.
    LeftGradientSum,
    /// The left hessian sum field.
    LeftHessianSum,
    /// The left count field.
    LeftCount,
    /// The right gradient sum field.
    RightGradientSum,
    /// The right hessian sum field.
    RightHessianSum,
    /// The right count field.
    RightCount,
    /// The NaN direction field.
    NanDirection,
}

/// Visitor for deserializing `SplitResultField` from string.
struct SplitResultFieldVisitor;

impl<'de> Visitor<'de> for SplitResultFieldVisitor {
    type Value = SplitResultField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "feature_index" => Ok(SplitResultField::FeatureIndex),
            "split_bin" => Ok(SplitResultField::SplitBin),
            "gain" => Ok(SplitResultField::Gain),
            "left_gradient_sum" => Ok(SplitResultField::LeftGradientSum),
            "left_hessian_sum" => Ok(SplitResultField::LeftHessianSum),
            "left_count" => Ok(SplitResultField::LeftCount),
            "right_gradient_sum" => Ok(SplitResultField::RightGradientSum),
            "right_hessian_sum" => Ok(SplitResultField::RightHessianSum),
            "right_count" => Ok(SplitResultField::RightCount),
            "nan_direction" => Ok(SplitResultField::NanDirection),
            _ => Err(E::unknown_field(value, SPLIT_RESULT_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for SplitResultField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(SplitResultFieldVisitor)
    }
}

/// Field names for `SplitResult` serialization.
const SPLIT_RESULT_FIELDS: &[&str] = &[
    "feature_index",
    "split_bin",
    "gain",
    "left_gradient_sum",
    "left_hessian_sum",
    "left_count",
    "right_gradient_sum",
    "right_hessian_sum",
    "right_count",
    "nan_direction",
];

impl<'de> Deserialize<'de> for SplitResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SplitResultVisitor;

        impl<'de> Visitor<'de> for SplitResultVisitor {
            type Value = SplitResult;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct SplitResult")
            }

            fn visit_map<V>(self, mut map: V) -> Result<SplitResult, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                let mut feature_index = None;
                let mut split_bin = None;
                let mut gain = None;
                let mut left_gradient_sum = None;
                let mut left_hessian_sum = None;
                let mut left_count = None;
                let mut right_gradient_sum = None;
                let mut right_hessian_sum = None;
                let mut right_count = None;
                let mut nan_direction = None;

                loop {
                    let key: Option<SplitResultField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        SplitResultField::FeatureIndex => {
                            feature_index = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::SplitBin => {
                            split_bin = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::Gain => {
                            gain = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::LeftGradientSum => {
                            left_gradient_sum = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::LeftHessianSum => {
                            left_hessian_sum = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::LeftCount => {
                            left_count = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::RightGradientSum => {
                            right_gradient_sum = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::RightHessianSum => {
                            right_hessian_sum = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::RightCount => {
                            right_count = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::NanDirection => {
                            nan_direction = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let feature_index = match feature_index {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("feature_index")),
                };
                let split_bin = match split_bin {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("split_bin")),
                };
                let gain = match gain {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("gain")),
                };
                let left_gradient_sum = match left_gradient_sum {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("left_gradient_sum")),
                };
                let left_hessian_sum = match left_hessian_sum {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("left_hessian_sum")),
                };
                let left_count = match left_count {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("left_count")),
                };
                let right_gradient_sum = match right_gradient_sum {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("right_gradient_sum")),
                };
                let right_hessian_sum = match right_hessian_sum {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("right_hessian_sum")),
                };
                let right_count = match right_count {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("right_count")),
                };
                let nan_direction = match nan_direction {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("nan_direction")),
                };

                Ok(SplitResult::new(SplitResultConfig {
                    feature_index,
                    split_bin,
                    gain,
                    left_gradient_sum,
                    left_hessian_sum,
                    left_count,
                    right_gradient_sum,
                    right_hessian_sum,
                    right_count,
                    nan_direction,
                }))
            }
        }

        deserializer.deserialize_struct("SplitResult", SPLIT_RESULT_FIELDS, SplitResultVisitor)
    }
}
