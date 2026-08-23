//! Tests for the multiclass training entry and its predict trio, driven
//! through the real registered bindings.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::helpers::{fail, make_config_dict, set_config_str, wrap_py_err};
use crate::error::ClearGbmError;
use crate::pyo3_module::entry_args::{
    predict_class_model_from_args, predict_proba_multiclass_model_from_args,
    predict_raw_multiclass_model_from_args, train_gradient_boosting_multiclass_from_args,
};

/// Nine rows on two features, three clusters by class on feature 0.
fn multiclass_rows() -> Vec<Vec<f64>> {
    vec![
        vec![0.0_f64, 0.0_f64],
        vec![1.0_f64, 0.0_f64],
        vec![2.0_f64, 0.0_f64],
        vec![10.0_f64, 0.0_f64],
        vec![11.0_f64, 0.0_f64],
        vec![12.0_f64, 0.0_f64],
        vec![20.0_f64, 0.0_f64],
        vec![21.0_f64, 0.0_f64],
        vec![22.0_f64, 0.0_f64],
    ]
}

/// The nine class labels for [`multiclass_rows`].
fn multiclass_labels() -> Vec<i64> {
    vec![0, 0, 0, 1, 1, 1, 2, 2, 2]
}

/// Builds the 8-tuple for the multiclass entry: the shared config dict
/// with the objective flipped to softmax, n_classes = 3, a wider bin
/// budget so the clusters stay separable, and no class weight.
fn make_multiclass_args(py: Python<'_>) -> Result<Bound<'_, PyTuple>, ClearGbmError> {
    let x_train = match PyArray2::from_vec2(py, &multiclass_rows()) {
        Ok(f) => f,
        Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
    };
    let y_train = PyArray1::from_vec(py, multiclass_labels());
    let config = match make_config_dict(py) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    match set_config_str(&config, "objective", "multiclass_softmax") {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match config.set_item("scale_pos_weight", py.None()) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match config.set_item("n_classes", 3_i64) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match config.set_item("max_bins", 16_i64) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let names = match PyList::new(py, ["f0", "f1"]) {
        Ok(l) => l,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match PyTuple::new(
        py,
        [
            x_train.into_any(),
            y_train.into_any(),
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
            config.into_any(),
            names.into_any(),
        ],
    ) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Builds a `(model, features)` predict tuple over the training rows.
fn make_predict_args<'py>(
    py: Python<'py>,
    model: &Py<PyAny>,
) -> Result<Bound<'py, PyTuple>, ClearGbmError> {
    let features = match PyArray2::from_vec2(py, &multiclass_rows()) {
        Ok(f) => f,
        Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
    };
    match PyTuple::new(py, [model.bind(py).clone().into_any(), features.into_any()]) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

#[test]
fn test_multiclass_entry_trains_and_predicts() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_multiclass_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let model = match train_gradient_boosting_multiclass_from_args(&args) {
            Ok(m) => m,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Classes: the argmax recovers the clusters.
        let predict_args = match make_predict_args(py, &model) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let classes_any = match predict_class_model_from_args(&predict_args) {
            Ok(c) => c,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let classes: PyReadonlyArray1<'_, i64> = match classes_any.bind(py).extract() {
            Ok(c) => c,
            Err(e) => return Err(fail(format!("class extraction failed: {e}"))),
        };
        let class_slice = match classes.as_slice() {
            Ok(s) => s,
            Err(e) => return Err(fail(format!("class slice failed: {e}"))),
        };
        assert_eq!(class_slice, multiclass_labels().as_slice());

        // Probabilities: shape (9, 3), rows sum to 1.
        let predict_args = match make_predict_args(py, &model) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let probas_any = match predict_proba_multiclass_model_from_args(&predict_args) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let probas: PyReadonlyArray2<'_, f64> = match probas_any.bind(py).extract() {
            Ok(p) => p,
            Err(e) => return Err(fail(format!("proba extraction failed: {e}"))),
        };
        let shape = (probas.shape()[0_usize], probas.shape()[1_usize]);
        assert_eq!(shape, (9_usize, 3_usize));
        let arr = probas.as_array();
        for row in 0_usize..9_usize {
            let sum: f64 = (0_usize..3_usize).map(|k| arr[[row, k]]).sum();
            assert!((sum - 1.0_f64).abs() < 1e-12_f64);
        }

        // Raw scores: shape (9, 3).
        let predict_args = match make_predict_args(py, &model) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let raw_any = match predict_raw_multiclass_model_from_args(&predict_args) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let raw: PyReadonlyArray2<'_, f64> = match raw_any.bind(py).extract() {
            Ok(r) => r,
            Err(e) => return Err(fail(format!("raw extraction failed: {e}"))),
        };
        assert_eq!(raw.shape()[1_usize], 3_usize);
        Ok(())
    })
}

#[test]
fn test_multiclass_entry_rejects_a_negative_label() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_multiclass_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        // Rebuild arg 1 with a negative label.
        let mut labels = multiclass_labels();
        labels[0] = -1_i64;
        let bad_labels = PyArray1::from_vec(py, labels);
        let item0 = match args.get_item(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let item6 = match args.get_item(6_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let item7 = match args.get_item(7_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let bad_args = match PyTuple::new(
            py,
            [
                item0,
                bad_labels.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                item6,
                item7,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_multiclass_from_args(&bad_args) {
            Ok(_) => Err(fail("a negative class label must be rejected".to_string())),
            Err(e) => {
                let text = e.to_string();
                assert!(text.contains("label"), "got: {text}");
                Ok(())
            }
        }
    })
}

#[test]
fn test_binary_model_rejects_the_multiclass_predict_surface() -> Result<(), ClearGbmError> {
    use super::helpers::train_model;

    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let features = match PyArray2::from_vec2(py, &multiclass_rows()) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let predict_args =
            match PyTuple::new(py, [model.bind(py).clone().into_any(), features.into_any()]) {
                Ok(t) => t,
                Err(e) => return Err(wrap_py_err(&e)),
            };
        match predict_proba_multiclass_model_from_args(&predict_args) {
            Ok(_) => Err(fail(
                "a binary model must reject the multiclass surface".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(text.contains("multiclass_softmax"), "got: {text}");
                Ok(())
            }
        }
    })
}

/// The config dict helpers are re-exported via `super::helpers`; this
/// module additionally needs a dict handle type in scope for signatures.
#[test]
fn test_multiclass_entry_requires_the_n_classes_pairing() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_multiclass_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(6_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.set_item("n_classes", py.None()) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_multiclass_from_args(&args) {
            Ok(_) => Err(fail(
                "multiclass without n_classes must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(text.contains("n_classes"), "got: {text}");
                Ok(())
            }
        }
    })
}

#[test]
fn test_multiclass_entry_accepts_validation_and_weights() -> Result<(), ClearGbmError> {
    // The fully-populated 8-tuple: weights, a validation split with its own
    // labels, and validation weights all cross the boundary.
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let base_args = match make_multiclass_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let x_val = match PyArray2::from_vec2(py, &multiclass_rows()) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_val = PyArray1::from_vec(py, multiclass_labels());
        let weights = PyArray1::from_vec(py, vec![1.0_f64; 9]);
        let val_weights = PyArray1::from_vec(py, vec![1.0_f64; 9]);
        let item0 = match base_args.get_item(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let item1 = match base_args.get_item(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let item6 = match base_args.get_item(6_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let item7 = match base_args.get_item(7_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let args = match PyTuple::new(
            py,
            [
                item0,
                item1,
                weights.into_any(),
                x_val.into_any(),
                y_val.into_any(),
                val_weights.into_any(),
                item6,
                item7,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_multiclass_from_args(&args) {
            Ok(_) => Ok(()),
            Err(e) => Err(wrap_py_err(&e)),
        }
    })
}

#[test]
fn test_multiclass_entry_enforces_the_validation_pairing() -> Result<(), ClearGbmError> {
    // x_val without y_val, y_val without x_val, and a validation weight
    // alone are each rejected with the pairing named.
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let variants: [(usize, &str); 3] =
            [(3_usize, "y_val"), (4_usize, "x_val"), (5_usize, "y_val")];
        for (slot, expected) in variants {
            let base_args = match make_multiclass_args(py) {
                Ok(a) => a,
                Err(e) => return Err(e),
            };
            let mut items: Vec<Bound<'_, pyo3::PyAny>> = Vec::with_capacity(8_usize);
            for i in 0_usize..8_usize {
                let item = match base_args.get_item(i) {
                    Ok(v) => v,
                    Err(e) => return Err(wrap_py_err(&e)),
                };
                items.push(item);
            }
            let filler: Bound<'_, pyo3::PyAny> = if slot == 3_usize {
                match PyArray2::from_vec2(py, &multiclass_rows()) {
                    Ok(f) => f.into_any(),
                    Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
                }
            } else if slot == 4_usize {
                PyArray1::from_vec(py, multiclass_labels()).into_any()
            } else {
                PyArray1::from_vec(py, vec![1.0_f64; 9]).into_any()
            };
            items[slot] = filler;
            let args = match PyTuple::new(py, items) {
                Ok(t) => t,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            match train_gradient_boosting_multiclass_from_args(&args) {
                Ok(_) => {
                    return Err(fail(format!(
                        "a lone slot-{slot} validation argument must be rejected"
                    )))
                }
                Err(e) => {
                    let text = e.to_string();
                    assert!(text.contains(expected), "slot {slot}: got {text}");
                }
            }
        }
        Ok(())
    })
}

#[test]
fn test_multiclass_entry_rejects_a_missing_n_classes_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_multiclass_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(6_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.del_item("n_classes") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_multiclass_from_args(&args) {
            Ok(_) => Err(fail("a missing n_classes key must be rejected".to_string())),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("missing required key 'n_classes'"),
                    "got: {text}"
                );
                Ok(())
            }
        }
    })
}
