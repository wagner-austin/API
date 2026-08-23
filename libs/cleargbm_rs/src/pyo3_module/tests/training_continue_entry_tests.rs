//! Tests for the continued-training entries, driven through the real
//! registered bindings.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::helpers::{fail, make_config_dict, make_regression_config_dict, wrap_py_err};
use crate::error::ClearGbmError;
use crate::pyo3_module::entry_args::{
    continue_gradient_boosting_from_args, continue_gradient_boosting_regression_from_args,
    train_gradient_boosting_from_args, train_gradient_boosting_regression_from_args,
};

/// Eight linearly separable rows on two features.
fn binary_rows() -> Vec<Vec<f64>> {
    vec![
        vec![0.0_f64, 0.1_f64],
        vec![0.1_f64, 0.0_f64],
        vec![0.2_f64, 0.2_f64],
        vec![0.3_f64, 0.1_f64],
        vec![0.8_f64, 0.9_f64],
        vec![0.9_f64, 0.8_f64],
        vec![1.0_f64, 1.0_f64],
        vec![0.7_f64, 0.9_f64],
    ]
}

/// The binary labels for [`binary_rows`].
fn binary_labels() -> Vec<i64> {
    vec![0, 0, 0, 0, 1, 1, 1, 1]
}

/// Trains a small binary model through the registered entry.
fn train_binary_model(py: Python<'_>) -> Result<Py<PyAny>, ClearGbmError> {
    let x_train = match PyArray2::from_vec2(py, &binary_rows()) {
        Ok(f) => f,
        Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
    };
    let y_train = PyArray1::from_vec(py, binary_labels());
    let config = match make_config_dict(py) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let names = match pyo3::types::PyList::new(py, ["f0", "f1"]) {
        Ok(l) => l,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let args = match PyTuple::new(
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
        Ok(t) => t,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match train_gradient_boosting_from_args(&args) {
        Ok(m) => Ok(m),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Builds the continuation 9-tuple over the training rows.
fn make_continue_args<'py>(
    py: Python<'py>,
    model: &Py<PyAny>,
    additional_rounds: i64,
) -> Result<Bound<'py, PyTuple>, ClearGbmError> {
    let x_train = match PyArray2::from_vec2(py, &binary_rows()) {
        Ok(f) => f,
        Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
    };
    let y_train = PyArray1::from_vec(py, binary_labels());
    match PyTuple::new(
        py,
        [
            model.bind(py).clone().into_any(),
            x_train.into_any(),
            y_train.into_any(),
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
            match additional_rounds.into_pyobject(py) {
                Ok(v) => v.into_any(),
                Err(never) => match never {},
            },
            match 1_i64.into_pyobject(py) {
                Ok(v) => v.into_any(),
                Err(never) => match never {},
            },
        ],
    ) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

#[test]
fn test_continue_entry_grows_the_binary_model() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_binary_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let args = match make_continue_args(py, &model, 2_i64) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let base_args = match PyTuple::new(py, [model.bind(py).clone().into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let base_trees =
            match crate::pyo3_module::model_fns::py_gbm_model_n_trees_from_args(&base_args) {
                Ok(v) => v,
                Err(e) => return Err(wrap_py_err(&e)),
            };
        let continued = match continue_gradient_boosting_from_args(&args) {
            Ok(m) => m,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        // Continuation adds exactly the requested rounds.
        let n_trees_args = match PyTuple::new(py, [continued.bind(py).clone().into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let n_trees =
            match crate::pyo3_module::model_fns::py_gbm_model_n_trees_from_args(&n_trees_args) {
                Ok(v) => v,
                Err(e) => return Err(wrap_py_err(&e)),
            };
        assert_eq!(n_trees, base_trees + 2_usize);
        Ok(())
    })
}

#[test]
fn test_continue_entry_rejects_negative_rounds_and_a_lone_val_slot() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_binary_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        // Negative rounds are rejected at the conversion boundary.
        let args = match make_continue_args(py, &model, -1_i64) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        match continue_gradient_boosting_from_args(&args) {
            Ok(_) => return Err(fail("negative rounds must be rejected".to_string())),
            Err(e) => {
                let text = e.to_string();
                assert!(text.contains("must be >= 1, got -1"), "got: {text}");
            }
        }

        // A lone x_val is rejected with the pairing named.
        let base_args = match make_continue_args(py, &model, 1_i64) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let mut items: Vec<Bound<'_, pyo3::PyAny>> = Vec::with_capacity(9_usize);
        for i in 0_usize..9_usize {
            let item = match base_args.get_item(i) {
                Ok(v) => v,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            items.push(item);
        }
        items[4] = match PyArray2::from_vec2(py, &binary_rows()) {
            Ok(f) => f.into_any(),
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let args = match PyTuple::new(py, items) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match continue_gradient_boosting_from_args(&args) {
            Ok(_) => return Err(fail("a lone x_val must be rejected".to_string())),
            Err(e) => {
                let text = e.to_string();
                assert!(text.contains("y_val"), "got: {text}");
            }
        }

        // A lone y_val and a lone val weight are each rejected too.
        for (slot, expected) in [(5_usize, "x_val"), (6_usize, "y_val")] {
            let base_args = match make_continue_args(py, &model, 1_i64) {
                Ok(a) => a,
                Err(e) => return Err(e),
            };
            let mut items: Vec<Bound<'_, pyo3::PyAny>> = Vec::with_capacity(9_usize);
            for i in 0_usize..9_usize {
                let item = match base_args.get_item(i) {
                    Ok(v) => v,
                    Err(e) => return Err(wrap_py_err(&e)),
                };
                items.push(item);
            }
            items[slot] = if slot == 5_usize {
                PyArray1::from_vec(py, binary_labels()).into_any()
            } else {
                PyArray1::from_vec(py, vec![1.0_f64; 8]).into_any()
            };
            let args = match PyTuple::new(py, items) {
                Ok(t) => t,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            match continue_gradient_boosting_from_args(&args) {
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
fn test_continue_entry_accepts_weights_and_validation() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_binary_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let base_args = match make_continue_args(py, &model, 1_i64) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let mut items: Vec<Bound<'_, pyo3::PyAny>> = Vec::with_capacity(9_usize);
        for i in 0_usize..9_usize {
            let item = match base_args.get_item(i) {
                Ok(v) => v,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            items.push(item);
        }
        items[3] = PyArray1::from_vec(py, vec![1.0_f64; 8]).into_any();
        items[4] = match PyArray2::from_vec2(py, &binary_rows()) {
            Ok(f) => f.into_any(),
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        items[5] = PyArray1::from_vec(py, binary_labels()).into_any();
        items[6] = PyArray1::from_vec(py, vec![1.0_f64; 8]).into_any();
        let args = match PyTuple::new(py, items) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match continue_gradient_boosting_from_args(&args) {
            Ok(_) => Ok(()),
            Err(e) => Err(wrap_py_err(&e)),
        }
    })
}

#[test]
fn test_regression_continue_entry_round_trips() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let rows: Vec<Vec<f64>> = (0_u32..8_u32)
            .map(|i| vec![f64::from(i), 0.5_f64])
            .collect();
        let targets: Vec<f64> = (0_u32..8_u32).map(|i| f64::from(i) * 2.0_f64).collect();
        let x_train = match PyArray2::from_vec2(py, &rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_train = PyArray1::from_vec(py, targets.clone());
        let config = match make_regression_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let names = match pyo3::types::PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let train_args = match PyTuple::new(
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
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let model = match train_gradient_boosting_regression_from_args(&train_args) {
            Ok(m) => m,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let x_again = match PyArray2::from_vec2(py, &rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_again = PyArray1::from_vec(py, targets);
        let continue_args = match PyTuple::new(
            py,
            [
                model.bind(py).clone().into_any(),
                x_again.into_any(),
                y_again.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                match 2_i64.into_pyobject(py) {
                    Ok(v) => v.into_any(),
                    Err(never) => match never {},
                },
                match 1_i64.into_pyobject(py) {
                    Ok(v) => v.into_any(),
                    Err(never) => match never {},
                },
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match continue_gradient_boosting_regression_from_args(&continue_args) {
            Ok(_) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        }

        // The fully-populated call: weights plus the validation pair with
        // its own weights.
        let mut items: Vec<Bound<'_, pyo3::PyAny>> = Vec::with_capacity(9_usize);
        for i in 0_usize..9_usize {
            let item = match continue_args.get_item(i) {
                Ok(v) => v,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            items.push(item);
        }
        let targets2: Vec<f64> = (0_u32..8_u32).map(|i| f64::from(i) * 2.0_f64).collect();
        items[3] = PyArray1::from_vec(py, vec![1.0_f64; 8]).into_any();
        items[4] = match PyArray2::from_vec2(py, &rows) {
            Ok(f) => f.into_any(),
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        items[5] = PyArray1::from_vec(py, targets2.clone()).into_any();
        items[6] = PyArray1::from_vec(py, vec![1.0_f64; 8]).into_any();
        let full_args = match PyTuple::new(py, items) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match continue_gradient_boosting_regression_from_args(&full_args) {
            Ok(_) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        }

        // Each lone validation slot is rejected with the pairing named.
        let variants: [(usize, &str); 3] =
            [(4_usize, "y_val"), (5_usize, "x_val"), (6_usize, "y_val")];
        for (slot, expected) in variants {
            let mut items: Vec<Bound<'_, pyo3::PyAny>> = Vec::with_capacity(9_usize);
            for i in 0_usize..9_usize {
                let item = match continue_args.get_item(i) {
                    Ok(v) => v,
                    Err(e) => return Err(wrap_py_err(&e)),
                };
                items.push(item);
            }
            items[slot] = if slot == 4_usize {
                match PyArray2::from_vec2(py, &rows) {
                    Ok(f) => f.into_any(),
                    Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
                }
            } else if slot == 5_usize {
                PyArray1::from_vec(py, targets2.clone()).into_any()
            } else {
                PyArray1::from_vec(py, vec![1.0_f64; 8]).into_any()
            };
            let args = match PyTuple::new(py, items) {
                Ok(t) => t,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            match continue_gradient_boosting_regression_from_args(&args) {
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

/// A binary model refuses regression continuation through the pairing.
#[test]
fn test_wrong_objective_continuation_is_refused() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_binary_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let rows = binary_rows();
        let x_train = match PyArray2::from_vec2(py, &rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let targets: Vec<f64> = binary_labels()
            .into_iter()
            .map(|l| if l == 1_i64 { 1.0_f64 } else { 0.0_f64 })
            .collect();
        let y_train = PyArray1::from_vec(py, targets);
        let args = match PyTuple::new(
            py,
            [
                model.bind(py).clone().into_any(),
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                match 1_i64.into_pyobject(py) {
                    Ok(v) => v.into_any(),
                    Err(never) => match never {},
                },
                match 1_i64.into_pyobject(py) {
                    Ok(v) => v.into_any(),
                    Err(never) => match never {},
                },
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match continue_gradient_boosting_regression_from_args(&args) {
            Ok(_) => Err(fail(
                "continuing a binary model with continuous labels must fail".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(text.contains("binary (u8) labels"), "got: {text}");
                Ok(())
            }
        }
    })
}
