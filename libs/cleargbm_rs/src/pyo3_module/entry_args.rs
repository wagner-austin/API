//! Positional-argument extraction for the registered training and
//! prediction entry points.
//!
//! Each `*_from_args` function unpacks the `PyTuple` a
//! `PyCFunction::new_closure` registration receives and delegates to the
//! typed core wrapper in [`super::training_fns`].

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::pyo3_module::model_fns::PyGbmModel;
use crate::pyo3_module::training_fns::{
    predict_proba_model_rs, predict_raw_model_rs, train_gradient_boosting_regression_rs,
    train_gradient_boosting_rs, TrainingArrays, ValidationArrays,
};
use crate::pyo3_module::training_multiclass_fns::{
    predict_class_model_rs, predict_proba_multiclass_model_rs, predict_raw_multiclass_model_rs,
    train_gradient_boosting_multiclass_rs,
};
use crate::pyo3_module::training_ranking_fns::{
    train_gradient_boosting_ranking_rs, RankingTrainingArrays, RankingValidationArrays,
};

/// Extracts arguments and delegates to [`train_gradient_boosting_rs`].
///
/// # Args (positional)
///
/// 0. `x_train` (numpy f64 2D array) - Training features.
/// 1. `y_train` (numpy i64 1D array) - Training labels.
/// 2. `sample_weight` (numpy f64 1D array or None) - Optional per-row
///    training weights.
/// 3. `x_val` (numpy f64 2D array or None) - Optional validation features.
/// 4. `y_val` (numpy i64 1D array or None) - Optional validation labels.
/// 5. `val_sample_weight` (numpy f64 1D array or None) - Optional per-row
///    evaluation weights.
/// 6. `config` (dict) - Training hyperparameters.
/// 7. `feature_names` (list of str) - Feature names.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or training fails.
pub(crate) fn train_gradient_boosting_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let x_train: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let y_train: PyReadonlyArray1<'_, i64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let arg2 = propagate!(args.get_item(2_usize));
    let sample_weight: Option<PyReadonlyArray1<'_, f64>> = if arg2.is_none() {
        None
    } else {
        Some(propagate_into!(arg2.extract()))
    };

    let arg3 = propagate!(args.get_item(3_usize));
    let x_val: Option<PyReadonlyArray2<'_, f64>> = if arg3.is_none() {
        None
    } else {
        Some(propagate_into!(arg3.extract()))
    };

    let arg4 = propagate!(args.get_item(4_usize));
    let y_val: Option<PyReadonlyArray1<'_, i64>> = if arg4.is_none() {
        None
    } else {
        Some(propagate_into!(arg4.extract()))
    };

    let arg5 = propagate!(args.get_item(5_usize));
    let val_sample_weight: Option<PyReadonlyArray1<'_, f64>> = if arg5.is_none() {
        None
    } else {
        Some(propagate_into!(arg5.extract()))
    };

    let config_dict: Bound<'_, PyDict> =
        propagate_into!(propagate!(args.get_item(6_usize)).extract());
    let feature_names: Bound<'_, PyList> =
        propagate_into!(propagate!(args.get_item(7_usize)).extract());

    let model = propagate!(train_gradient_boosting_rs(
        py,
        &TrainingArrays {
            x: &x_train,
            y: &y_train,
            weight: sample_weight.as_ref(),
        },
        &ValidationArrays {
            x: x_val.as_ref(),
            y: y_val.as_ref(),
            weight: val_sample_weight.as_ref(),
        },
        &config_dict,
        &feature_names,
    ));

    Ok(model.into_any())
}

/// Extracts arguments and delegates to
/// [`train_gradient_boosting_regression_rs`].
///
/// # Args (positional)
///
/// 0. `x_train` (numpy f64 2D array) - Training features.
/// 1. `y_train` (numpy f64 1D array) - Continuous training targets.
/// 2. `sample_weight` (numpy f64 1D array or None) - Optional per-row
///    training weights.
/// 3. `x_val` (numpy f64 2D array or None) - Optional validation features.
/// 4. `y_val` (numpy f64 1D array or None) - Optional validation targets.
/// 5. `val_sample_weight` (numpy f64 1D array or None) - Optional per-row
///    evaluation weights.
/// 6. `config` (dict) - Training hyperparameters.
/// 7. `feature_names` (list of str) - Feature names.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or training fails.
pub(crate) fn train_gradient_boosting_regression_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let x_train: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let y_train: PyReadonlyArray1<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let arg2 = propagate!(args.get_item(2_usize));
    let sample_weight: Option<PyReadonlyArray1<'_, f64>> = if arg2.is_none() {
        None
    } else {
        Some(propagate_into!(arg2.extract()))
    };

    let arg3 = propagate!(args.get_item(3_usize));
    let x_val: Option<PyReadonlyArray2<'_, f64>> = if arg3.is_none() {
        None
    } else {
        Some(propagate_into!(arg3.extract()))
    };

    let arg4 = propagate!(args.get_item(4_usize));
    let y_val: Option<PyReadonlyArray1<'_, f64>> = if arg4.is_none() {
        None
    } else {
        Some(propagate_into!(arg4.extract()))
    };

    let arg5 = propagate!(args.get_item(5_usize));
    let val_sample_weight: Option<PyReadonlyArray1<'_, f64>> = if arg5.is_none() {
        None
    } else {
        Some(propagate_into!(arg5.extract()))
    };

    let config_dict: Bound<'_, PyDict> =
        propagate_into!(propagate!(args.get_item(6_usize)).extract());
    let feature_names: Bound<'_, PyList> =
        propagate_into!(propagate!(args.get_item(7_usize)).extract());

    let model = propagate!(train_gradient_boosting_regression_rs(
        py,
        &TrainingArrays {
            x: &x_train,
            y: &y_train,
            weight: sample_weight.as_ref(),
        },
        &ValidationArrays {
            x: x_val.as_ref(),
            y: y_val.as_ref(),
            weight: val_sample_weight.as_ref(),
        },
        &config_dict,
        &feature_names,
    ));

    Ok(model.into_any())
}

/// Extracts arguments and delegates to [`predict_proba_model_rs`].
///
/// # Args (positional)
///
/// 0. `model` (PyGbmModel) - Trained model.
/// 1. `features` (numpy f64 2D array) - Feature matrix.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_proba_model_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let model: PyRef<'_, PyGbmModel> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let features: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let result = propagate!(predict_proba_model_rs(py, &model, &features));
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to [`predict_raw_model_rs`].
///
/// # Args (positional)
///
/// 0. `model` (PyGbmModel) - Trained model.
/// 1. `features` (numpy f64 2D array) - Feature matrix.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_raw_model_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let model: PyRef<'_, PyGbmModel> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let features: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let result = propagate!(predict_raw_model_rs(py, &model, &features));
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to
/// [`train_gradient_boosting_multiclass_rs`].
///
/// # Args (positional)
///
/// 0. `x_train` (numpy f64 2D array) - Training features.
/// 1. `y_train` (numpy i64 1D array) - Class labels.
/// 2. `sample_weight` (numpy f64 1D array or None) - Optional per-row
///    training weights.
/// 3. `x_val` (numpy f64 2D array or None) - Optional validation features.
/// 4. `y_val` (numpy i64 1D array or None) - Optional validation labels.
/// 5. `val_sample_weight` (numpy f64 1D array or None) - Optional per-row
///    evaluation weights.
/// 6. `config` (dict) - Training hyperparameters.
/// 7. `feature_names` (list of str) - Feature names.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or training fails.
pub(crate) fn train_gradient_boosting_multiclass_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let x_train: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let y_train: PyReadonlyArray1<'_, i64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let arg2 = propagate!(args.get_item(2_usize));
    let sample_weight: Option<PyReadonlyArray1<'_, f64>> = if arg2.is_none() {
        None
    } else {
        Some(propagate_into!(arg2.extract()))
    };

    let arg3 = propagate!(args.get_item(3_usize));
    let x_val: Option<PyReadonlyArray2<'_, f64>> = if arg3.is_none() {
        None
    } else {
        Some(propagate_into!(arg3.extract()))
    };

    let arg4 = propagate!(args.get_item(4_usize));
    let y_val: Option<PyReadonlyArray1<'_, i64>> = if arg4.is_none() {
        None
    } else {
        Some(propagate_into!(arg4.extract()))
    };

    let arg5 = propagate!(args.get_item(5_usize));
    let val_sample_weight: Option<PyReadonlyArray1<'_, f64>> = if arg5.is_none() {
        None
    } else {
        Some(propagate_into!(arg5.extract()))
    };

    let config_dict: Bound<'_, PyDict> =
        propagate_into!(propagate!(args.get_item(6_usize)).extract());
    let feature_names: Bound<'_, PyList> =
        propagate_into!(propagate!(args.get_item(7_usize)).extract());

    let model = propagate!(train_gradient_boosting_multiclass_rs(
        py,
        &TrainingArrays {
            x: &x_train,
            y: &y_train,
            weight: sample_weight.as_ref(),
        },
        &ValidationArrays {
            x: x_val.as_ref(),
            y: y_val.as_ref(),
            weight: val_sample_weight.as_ref(),
        },
        &config_dict,
        &feature_names,
    ));

    Ok(model.into_any())
}

/// Extracts arguments and delegates to
/// [`train_gradient_boosting_ranking_rs`].
///
/// # Args (positional)
///
/// 0. `x_train` (numpy f64 2D array) - Training features.
/// 1. `y_train` (numpy i64 1D array) - Relevance labels.
/// 2. `group` (numpy i64 1D array) - Documents per query, in row order.
/// 3. `sample_weight` (numpy f64 1D array or None) - Optional per-row
///    training weights.
/// 4. `x_val` (numpy f64 2D array or None) - Optional validation features.
/// 5. `y_val` (numpy i64 1D array or None) - Optional validation labels.
/// 6. `val_group` (numpy i64 1D array or None) - Optional validation
///    query group sizes.
/// 7. `config` (dict) - Training hyperparameters.
/// 8. `feature_names` (list of str) - Feature names.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or training fails.
pub(crate) fn train_gradient_boosting_ranking_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let x_train: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let y_train: PyReadonlyArray1<'_, i64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());
    let group: PyReadonlyArray1<'_, i64> =
        propagate_into!(propagate!(args.get_item(2_usize)).extract());

    let arg3 = propagate!(args.get_item(3_usize));
    let sample_weight: Option<PyReadonlyArray1<'_, f64>> = if arg3.is_none() {
        None
    } else {
        Some(propagate_into!(arg3.extract()))
    };

    let arg4 = propagate!(args.get_item(4_usize));
    let x_val: Option<PyReadonlyArray2<'_, f64>> = if arg4.is_none() {
        None
    } else {
        Some(propagate_into!(arg4.extract()))
    };

    let arg5 = propagate!(args.get_item(5_usize));
    let y_val: Option<PyReadonlyArray1<'_, i64>> = if arg5.is_none() {
        None
    } else {
        Some(propagate_into!(arg5.extract()))
    };

    let arg6 = propagate!(args.get_item(6_usize));
    let val_group: Option<PyReadonlyArray1<'_, i64>> = if arg6.is_none() {
        None
    } else {
        Some(propagate_into!(arg6.extract()))
    };

    let config_dict: Bound<'_, PyDict> =
        propagate_into!(propagate!(args.get_item(7_usize)).extract());
    let feature_names: Bound<'_, PyList> =
        propagate_into!(propagate!(args.get_item(8_usize)).extract());

    let model = propagate!(train_gradient_boosting_ranking_rs(
        py,
        &RankingTrainingArrays {
            x: &x_train,
            y: &y_train,
            group: &group,
            weight: sample_weight.as_ref(),
        },
        &RankingValidationArrays {
            x: x_val.as_ref(),
            y: y_val.as_ref(),
            group: val_group.as_ref(),
        },
        &config_dict,
        &feature_names,
    ));

    Ok(model.into_any())
}

/// Extracts arguments and delegates to [`predict_raw_multiclass_model_rs`].
///
/// # Args (positional)
///
/// 0. `model` (PyGbmModel) - Trained multiclass model.
/// 1. `features` (numpy f64 2D array) - Feature matrix.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_raw_multiclass_model_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let py = args.py();
    let model: PyRef<'_, PyGbmModel> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let features: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());
    let result = propagate!(predict_raw_multiclass_model_rs(py, &model, &features));
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to
/// [`predict_proba_multiclass_model_rs`].
///
/// # Args (positional)
///
/// 0. `model` (PyGbmModel) - Trained multiclass model.
/// 1. `features` (numpy f64 2D array) - Feature matrix.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_proba_multiclass_model_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let py = args.py();
    let model: PyRef<'_, PyGbmModel> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let features: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());
    let result = propagate!(predict_proba_multiclass_model_rs(py, &model, &features));
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to [`predict_class_model_rs`].
///
/// # Args (positional)
///
/// 0. `model` (PyGbmModel) - Trained multiclass model.
/// 1. `features` (numpy f64 2D array) - Feature matrix.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_class_model_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();
    let model: PyRef<'_, PyGbmModel> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let features: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());
    let result = propagate!(predict_class_model_rs(py, &model, &features));
    Ok(result.unbind().into_any())
}
