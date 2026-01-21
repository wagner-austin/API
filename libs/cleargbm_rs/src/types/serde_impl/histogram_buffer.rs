//! Manual serde implementation for `HistogramBuffer`.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::types::HistogramBuffer;

impl Serialize for HistogramBuffer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("HistogramBuffer", 4) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("n_bins", &self.n_bins()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("gradient_sums", self.gradient_sums()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("hessian_sums", self.hessian_sums()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("counts", self.counts()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `HistogramBuffer` deserialization.
pub(crate) enum HistogramBufferField {
    /// The n_bins field.
    NBins,
    /// The gradient_sums field.
    GradientSums,
    /// The hessian_sums field.
    HessianSums,
    /// The counts field.
    Counts,
}

/// Visitor for deserializing `HistogramBufferField` from string.
///
/// Exposed for testing the expecting() error path.
pub(crate) struct HistogramBufferFieldVisitor;

impl<'de> Visitor<'de> for HistogramBufferFieldVisitor {
    type Value = HistogramBufferField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match formatter.write_str("field identifier") {
            Ok(()) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "n_bins" => Ok(HistogramBufferField::NBins),
            "gradient_sums" => Ok(HistogramBufferField::GradientSums),
            "hessian_sums" => Ok(HistogramBufferField::HessianSums),
            "counts" => Ok(HistogramBufferField::Counts),
            _ => Err(E::unknown_field(value, HISTOGRAM_BUFFER_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for HistogramBufferField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(HistogramBufferFieldVisitor)
    }
}

/// Field names for `HistogramBuffer` serialization.
const HISTOGRAM_BUFFER_FIELDS: &[&str] = &["n_bins", "gradient_sums", "hessian_sums", "counts"];

/// Visitor for deserializing `HistogramBuffer` from a map.
///
/// Exposed for testing the expecting() error path.
pub(crate) struct HistogramBufferVisitor;

impl<'de> Visitor<'de> for HistogramBufferVisitor {
    type Value = HistogramBuffer;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match formatter.write_str("struct HistogramBuffer") {
            Ok(()) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn visit_map<V>(self, mut map: V) -> Result<HistogramBuffer, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut n_bins = None;
        let mut gradient_sums = None;
        let mut hessian_sums = None;
        let mut counts = None;

        loop {
            let key: Option<HistogramBufferField> = match map.next_key() {
                Ok(k) => k,
                Err(e) => return Err(e),
            };
            let key = match key {
                Some(k) => k,
                None => break,
            };
            match key {
                HistogramBufferField::NBins => {
                    if n_bins.is_some() {
                        return Err(de::Error::duplicate_field("n_bins"));
                    }
                    n_bins = Some(match map.next_value() {
                        Ok(v) => v,
                        Err(e) => return Err(e),
                    });
                }
                HistogramBufferField::GradientSums => {
                    if gradient_sums.is_some() {
                        return Err(de::Error::duplicate_field("gradient_sums"));
                    }
                    gradient_sums = Some(match map.next_value() {
                        Ok(v) => v,
                        Err(e) => return Err(e),
                    });
                }
                HistogramBufferField::HessianSums => {
                    if hessian_sums.is_some() {
                        return Err(de::Error::duplicate_field("hessian_sums"));
                    }
                    hessian_sums = Some(match map.next_value() {
                        Ok(v) => v,
                        Err(e) => return Err(e),
                    });
                }
                HistogramBufferField::Counts => {
                    if counts.is_some() {
                        return Err(de::Error::duplicate_field("counts"));
                    }
                    counts = Some(match map.next_value() {
                        Ok(v) => v,
                        Err(e) => return Err(e),
                    });
                }
            }
        }

        let n_bins = match n_bins {
            Some(v) => v,
            None => return Err(de::Error::missing_field("n_bins")),
        };
        let gradient_sums = match gradient_sums {
            Some(v) => v,
            None => return Err(de::Error::missing_field("gradient_sums")),
        };
        let hessian_sums = match hessian_sums {
            Some(v) => v,
            None => return Err(de::Error::missing_field("hessian_sums")),
        };
        let counts = match counts {
            Some(v) => v,
            None => return Err(de::Error::missing_field("counts")),
        };

        Ok(HistogramBuffer {
            n_bins,
            gradient_sums,
            hessian_sums,
            counts,
        })
    }
}

impl<'de> Deserialize<'de> for HistogramBuffer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "HistogramBuffer",
            HISTOGRAM_BUFFER_FIELDS,
            HistogramBufferVisitor,
        )
    }
}
