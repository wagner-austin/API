//! PyO3 bindings for LambdaMART ranking training.
//!
//! Mirrors [`super::training_fns`] for the `lambdarank` objective: relevance
//! labels arrive as numpy i64 grades and convert to `u32`, query group sizes
//! arrive as numpy i64 counts and convert to `usize`, and prediction reuses
//! the existing single-score `predict_raw` surface — the raw score IS the
//! ranking key.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::pyo3_module::config_extract::{dict_get_i64, extract_config};
use crate::pyo3_module::model_fns::PyGbmModel;
use crate::pyo3_module::training_fns::{extract_feature_names, extract_rows, extract_targets};
use crate::pyo3_module::training_multiclass_fns::extract_class_labels;
use crate::training::{
    train_gradient_boosting_ranking, RankingTrainingData, RankingValidationData,
};
use crate::training::{Parallelism, TrainingRuntime};

/// The ranking entry's training-side arrays.
pub(crate) struct RankingTrainingArrays<'py, 'a> {
    /// Training feature matrix.
    pub x: &'a PyReadonlyArray2<'py, f64>,
    /// Relevance labels (i64 grades).
    pub y: &'a PyReadonlyArray1<'py, i64>,
    /// Documents per query, in row order.
    pub group: &'a PyReadonlyArray1<'py, i64>,
    /// Optional per-row training weights.
    pub weight: Option<&'a PyReadonlyArray1<'py, f64>>,
}

/// The ranking entry's optional validation-side arrays.
pub(crate) struct RankingValidationArrays<'py, 'a> {
    /// Validation feature matrix.
    pub x: Option<&'a PyReadonlyArray2<'py, f64>>,
    /// Validation relevance labels.
    pub y: Option<&'a PyReadonlyArray1<'py, i64>>,
    /// Documents per validation query.
    pub group: Option<&'a PyReadonlyArray1<'py, i64>>,
}

/// Extracts query group sizes from a numpy i64 array to `Vec<usize>`.
///
/// Content validation (non-empty, per-query bounds, sum against the row
/// count) is owned by the core's ranking entry; this rejects only values a
/// `usize` cannot hold.
///
/// # Errors
///
/// Returns `PyErr` if the array is non-contiguous or a count is negative.
fn extract_group_sizes(group: &PyReadonlyArray1<'_, i64>, name: &str) -> PyResult<Vec<usize>> {
    let slice = propagate_into!(group.as_slice());
    let mut result = Vec::with_capacity(slice.len());
    for (i, &val) in slice.iter().enumerate() {
        let converted = match usize::try_from(val) {
            Ok(v) => v,
            Err(_) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: name.to_string(),
                    reason: format!("group size at index {i} must be >= 0, got {val}"),
                }
                .into())
            }
        };
        result.push(converted);
    }
    Ok(result)
}

/// Trains a LambdaMART ranking model from Python data.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `train` - Training arrays: features, i64 relevance labels, i64 query
///   group sizes, and optional per-row weights (`None` weighs every row 1).
/// * `val` - Optional validation arrays: features, labels, and group
///   sizes — all three or none. There is no validation-weight argument:
///   NDCG is a per-query metric, and a per-document evaluation weight has
///   no defined meaning for it.
/// * `config_dict` - Python dict with training hyperparameters; its
///   `objective` must be `"lambdarank"` with `lambdarank_truncation_level`
///   set.
/// * `feature_names` - Python list of feature name strings.
///
/// # Returns
///
/// A [`PyGbmModel`] wrapping the trained model; score it with the existing
/// `predict_raw_model_rs` — the raw score is the ranking key.
///
/// # Errors
///
/// Returns `PyErr` on argument extraction, validation, or training errors.
pub(crate) fn train_gradient_boosting_ranking_rs(
    py: Python<'_>,
    train: &RankingTrainingArrays<'_, '_>,
    val: &RankingValidationArrays<'_, '_>,
    config_dict: &Bound<'_, PyDict>,
    feature_names: &Bound<'_, PyList>,
) -> PyResult<Py<PyGbmModel>> {
    let train_rows = propagate_into!(extract_rows(train.x));
    let train_slices: Vec<&[f64]> = train_rows.iter().map(Vec::as_slice).collect();

    let y_train_u32 = propagate!(extract_class_labels(train.y));
    let groups = propagate!(extract_group_sizes(train.group, "group"));
    let weights: Option<Vec<f64>> = match train.weight {
        Some(w) => Some(propagate!(extract_targets(w))),
        None => None,
    };

    // Validation is all-three-or-none: features, labels and groups
    // describe one split together.
    let val_rows: Option<Vec<Vec<f64>>> = match val.x {
        Some(xv) => Some(propagate_into!(extract_rows(xv))),
        None => None,
    };
    let val_slices: Option<Vec<&[f64]>> = val_rows
        .as_ref()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());
    let y_val_u32: Option<Vec<u32>> = match val.y {
        Some(yv) => Some(propagate!(extract_class_labels(yv))),
        None => None,
    };
    let val_groups: Option<Vec<usize>> = match val.group {
        Some(g) => Some(propagate!(extract_group_sizes(g, "val_group"))),
        None => None,
    };

    let validation: Option<RankingValidationData<'_>> = match (&val_slices, &y_val_u32, &val_groups)
    {
        (Some(xv), Some(yv), Some(gv)) => Some(RankingValidationData {
            x: xv,
            y: yv,
            groups: gv,
        }),
        (None, None, None) => None,
        _ => {
            return Err(ClearGbmError::InvalidParameter {
                name: "x_val".to_string(),
                reason: "ranking validation requires x_val, y_val and val_group \
                             together, or none of them"
                    .to_string(),
            }
            .into())
        }
    };

    let config = propagate!(extract_config(config_dict));
    let names = propagate!(extract_feature_names(feature_names));
    let parallelism = propagate_into!(Parallelism::from_n_jobs(propagate!(dict_get_i64(
        config_dict,
        "n_jobs"
    ))));

    let model = propagate_into!(train_gradient_boosting_ranking(
        &train_slices,
        &RankingTrainingData {
            y: &y_train_u32,
            groups: &groups,
            weight: weights.as_deref(),
        },
        validation,
        &config,
        &names,
        &TrainingRuntime {
            parallelism,
            hooks: &Hooks::default(),
        },
    ));

    Py::new(py, PyGbmModel { inner: model })
}
