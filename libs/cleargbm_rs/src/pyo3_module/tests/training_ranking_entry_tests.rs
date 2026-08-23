//! Tests for the LambdaMART ranking training entry, driven through the
//! real registered binding. Prediction goes through the existing
//! single-score `predict_raw` surface — the raw score is the ranking key.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::helpers::{fail, make_config_dict, set_config_str, wrap_py_err};
use crate::error::ClearGbmError;
use crate::pyo3_module::entry_args::{
    predict_raw_model_from_args, train_gradient_boosting_ranking_from_args,
};

/// Two queries of four documents each; feature 0 carries the relevance
/// signal offset per query, feature 1 is constant.
fn ranking_rows() -> Vec<Vec<f64>> {
    vec![
        vec![0.0_f64, 0.5_f64],
        vec![1.0_f64, 0.5_f64],
        vec![2.0_f64, 0.5_f64],
        vec![3.0_f64, 0.5_f64],
        vec![0.1_f64, 0.5_f64],
        vec![1.1_f64, 0.5_f64],
        vec![2.1_f64, 0.5_f64],
        vec![3.1_f64, 0.5_f64],
    ]
}

/// The relevance labels for [`ranking_rows`].
fn ranking_labels() -> Vec<i64> {
    vec![0, 1, 2, 3, 0, 1, 2, 3]
}

/// Builds the 9-tuple for the ranking entry: the shared config dict with
/// the objective flipped to lambdarank, the truncation level set, no class
/// weight, and a wider bin budget so the signal stays separable.
fn make_ranking_args(py: Python<'_>) -> Result<Bound<'_, PyTuple>, ClearGbmError> {
    let x_train = match PyArray2::from_vec2(py, &ranking_rows()) {
        Ok(f) => f,
        Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
    };
    let y_train = PyArray1::from_vec(py, ranking_labels());
    let group = PyArray1::from_vec(py, vec![4_i64, 4_i64]);
    let config = match make_config_dict(py) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    match set_config_str(&config, "objective", "lambdarank") {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match config.set_item("scale_pos_weight", py.None()) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match config.set_item("lambdarank_truncation_level", 4_i64) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match config.set_item("max_bins", 16_i64) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    // Enough rounds to separate the low-relevance pair, whose NDCG
    // contribution (and therefore lambda) is smallest.
    match config.set_item("n_estimators", 50_i64) {
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
            group.into_any(),
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

#[test]
fn test_ranking_entry_trains_and_scores_in_label_order() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_ranking_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let model = match train_gradient_boosting_ranking_from_args(&args) {
            Ok(m) => m,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let features = match PyArray2::from_vec2(py, &ranking_rows()) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let predict_args =
            match PyTuple::new(py, [model.bind(py).clone().into_any(), features.into_any()]) {
                Ok(t) => t,
                Err(e) => return Err(wrap_py_err(&e)),
            };
        let scores_any = match predict_raw_model_from_args(&predict_args) {
            Ok(s) => s,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let scores: PyReadonlyArray1<'_, f64> = match scores_any.bind(py).extract() {
            Ok(s) => s,
            Err(e) => return Err(fail(format!("scores extraction failed: {e}"))),
        };
        let slice = match scores.as_slice() {
            Ok(s) => s,
            Err(e) => return Err(fail(format!("scores not contiguous: {e}"))),
        };
        // Within each query the scores must ascend with the labels.
        for query in 0_usize..2_usize {
            let base = query * 4_usize;
            for i in 0_usize..3_usize {
                assert!(
                    slice[base + i] < slice[base + i + 1_usize],
                    "query {query} misordered at {i}: {slice:?}"
                );
            }
        }
        Ok(())
    })
}

#[test]
fn test_ranking_entry_accepts_weights_and_a_validation_split() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let base_args = match make_ranking_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let weights = PyArray1::from_vec(py, vec![1.0_f64; 8]);
        let x_val = match PyArray2::from_vec2(py, &ranking_rows()) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_val = PyArray1::from_vec(py, ranking_labels());
        let val_group = PyArray1::from_vec(py, vec![4_i64, 4_i64]);
        let mut items: Vec<Bound<'_, pyo3::PyAny>> = Vec::with_capacity(9_usize);
        for i in 0_usize..9_usize {
            let item = match base_args.get_item(i) {
                Ok(v) => v,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            items.push(item);
        }
        items[3] = weights.into_any();
        items[4] = x_val.into_any();
        items[5] = y_val.into_any();
        items[6] = val_group.into_any();
        let args = match PyTuple::new(py, items) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_ranking_from_args(&args) {
            Ok(_) => Ok(()),
            Err(e) => Err(wrap_py_err(&e)),
        }
    })
}

#[test]
fn test_ranking_entry_enforces_the_validation_triple() -> Result<(), ClearGbmError> {
    // A lone validation argument in any of the three slots is rejected
    // with the triple named.
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        for slot in [4_usize, 5_usize, 6_usize] {
            let base_args = match make_ranking_args(py) {
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
            let filler: Bound<'_, pyo3::PyAny> = if slot == 4_usize {
                match PyArray2::from_vec2(py, &ranking_rows()) {
                    Ok(f) => f.into_any(),
                    Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
                }
            } else if slot == 5_usize {
                PyArray1::from_vec(py, ranking_labels()).into_any()
            } else {
                PyArray1::from_vec(py, vec![4_i64, 4_i64]).into_any()
            };
            items[slot] = filler;
            let args = match PyTuple::new(py, items) {
                Ok(t) => t,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            match train_gradient_boosting_ranking_from_args(&args) {
                Ok(_) => {
                    return Err(fail(format!(
                        "a lone slot-{slot} validation argument must be rejected"
                    )))
                }
                Err(e) => {
                    let text = e.to_string();
                    assert!(text.contains("together"), "slot {slot}: got {text}");
                }
            }
        }
        Ok(())
    })
}

#[test]
fn test_ranking_entry_rejects_a_negative_group_size() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let base_args = match make_ranking_args(py) {
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
        items[2] = PyArray1::from_vec(py, vec![-4_i64, 12_i64]).into_any();
        let args = match PyTuple::new(py, items) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_ranking_from_args(&args) {
            Ok(_) => Err(fail("a negative group size must be rejected".to_string())),
            Err(e) => {
                let text = e.to_string();
                assert!(text.contains("must be >= 0, got -4"), "got: {text}");
                Ok(())
            }
        }
    })
}

#[test]
fn test_ranking_entry_rejects_a_missing_truncation_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_ranking_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(7_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.del_item("lambdarank_truncation_level") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_ranking_from_args(&args) {
            Ok(_) => Err(fail(
                "a missing lambdarank_truncation_level key must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("missing required key 'lambdarank_truncation_level'"),
                    "got: {text}"
                );
                Ok(())
            }
        }
    })
}
