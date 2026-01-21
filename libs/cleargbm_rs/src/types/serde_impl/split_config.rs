//! Manual serde implementation for `SplitConfig`.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::types::SplitConfig;

impl Serialize for SplitConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("SplitConfig", 5) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
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
        match state.serialize_field("reg_lambda", &self.reg_lambda()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("min_gain", &self.min_gain()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `SplitConfig` deserialization.
enum SplitConfigField {
    /// The min_samples_split field.
    MinSamplesSplit,
    /// The min_samples_leaf field.
    MinSamplesLeaf,
    /// The max_bins field.
    MaxBins,
    /// The reg_lambda field.
    RegLambda,
    /// The min_gain field.
    MinGain,
}

/// Visitor for deserializing `SplitConfigField` from string.
struct SplitConfigFieldVisitor;

impl<'de> Visitor<'de> for SplitConfigFieldVisitor {
    type Value = SplitConfigField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "min_samples_split" => Ok(SplitConfigField::MinSamplesSplit),
            "min_samples_leaf" => Ok(SplitConfigField::MinSamplesLeaf),
            "max_bins" => Ok(SplitConfigField::MaxBins),
            "reg_lambda" => Ok(SplitConfigField::RegLambda),
            "min_gain" => Ok(SplitConfigField::MinGain),
            _ => Err(E::unknown_field(value, SPLIT_CONFIG_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for SplitConfigField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(SplitConfigFieldVisitor)
    }
}

/// Field names for `SplitConfig` serialization.
const SPLIT_CONFIG_FIELDS: &[&str] = &[
    "min_samples_split",
    "min_samples_leaf",
    "max_bins",
    "reg_lambda",
    "min_gain",
];

impl<'de> Deserialize<'de> for SplitConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SplitConfigVisitor;

        impl<'de> Visitor<'de> for SplitConfigVisitor {
            type Value = SplitConfig;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct SplitConfig")
            }

            fn visit_map<V>(self, mut map: V) -> Result<SplitConfig, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut min_samples_split = None;
                let mut min_samples_leaf = None;
                let mut max_bins = None;
                let mut reg_lambda = None;
                let mut min_gain = None;

                loop {
                    let key: Option<SplitConfigField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        SplitConfigField::MinSamplesSplit => {
                            min_samples_split = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitConfigField::MinSamplesLeaf => {
                            min_samples_leaf = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitConfigField::MaxBins => {
                            max_bins = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitConfigField::RegLambda => {
                            reg_lambda = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitConfigField::MinGain => {
                            min_gain = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

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
                let reg_lambda = match reg_lambda {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("reg_lambda")),
                };
                let min_gain = match min_gain {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("min_gain")),
                };

                Ok(SplitConfig {
                    min_samples_split,
                    min_samples_leaf,
                    max_bins,
                    reg_lambda,
                    min_gain,
                })
            }
        }

        deserializer.deserialize_struct("SplitConfig", SPLIT_CONFIG_FIELDS, SplitConfigVisitor)
    }
}
