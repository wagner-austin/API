"""Random Forest backend hyperparameter-variation tests with actual sklearn training.

Each test trains on real US bankruptcy data under one configuration change and
checks the outcome the backend reports for it.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.backends.random_forest import create_random_forest_backend
from covenant_ml.explainers.adapters import try_extract_native_tree_model
from tests.backends.random_forest._rf_fixtures import _invoke_rf_train, _make_rf_config

from ...conftest import load_us_bankruptcy_data


def test_rf_backend_with_oob_score(tmp_path: Path) -> None:
    """RandomForestBackend works with OOB score enabled."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=100, bootstrap=True, oob_score=True)

    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5
    assert Path(outcome["model_path"]).exists()


def test_rf_backend_without_bootstrap(tmp_path: Path) -> None:
    """RandomForestBackend works without bootstrap (full dataset)."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=10, bootstrap=False, oob_score=False)

    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_rf_backend_with_no_max_depth(tmp_path: Path) -> None:
    """RandomForestBackend works with unlimited depth."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=10, max_depth=None)

    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_rf_backend_with_log2_features(tmp_path: Path) -> None:
    """RandomForestBackend works with log2 max_features."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=10, max_features="log2")

    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_rf_backend_with_high_min_samples(tmp_path: Path) -> None:
    """RandomForestBackend works with high min_samples parameters."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(
        n_estimators=10,
        min_samples_split=10,
        min_samples_leaf=5,
    )

    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Should still produce valid output
    assert 0.0 <= outcome["best_val_auc"] <= 1.0
    assert Path(outcome["model_path"]).exists()


def test_rf_backend_without_class_weight_balance(tmp_path: Path) -> None:
    """RandomForestBackend works without class weight balancing."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=10)
    config["class_weight_balanced"] = False

    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_rf_backend_feature_importance_ranking(tmp_path: Path) -> None:
    """RandomForestBackend produces correctly ranked feature importances."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=20)
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    importances = outcome["feature_importances"]

    # Verify ranks are correct
    for i, feat in enumerate(importances):
        assert feat["rank"] == i + 1

    # Verify sorted by importance (descending)
    for i in range(len(importances) - 1):
        assert importances[i]["importance"] >= importances[i + 1]["importance"]


def test_rf_backend_config_stored_in_outcome(tmp_path: Path) -> None:
    """RandomForestBackend stores config in TrainOutcome."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=15, max_depth=7)
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Verify config matches what was passed in (compare against input)
    assert outcome["config"]["random_state"] == config["random_state"]
    assert outcome["config"]["train_ratio"] == config["train_ratio"]


def test_rf_backend_scale_pos_weight_computed(tmp_path: Path) -> None:
    """RandomForestBackend computes and stores scale_pos_weight."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config()
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Verify scale_pos_weight_computed is positive
    assert outcome["scale_pos_weight_computed"] > 0.0


def test_rf_backend_single_round_training(tmp_path: Path) -> None:
    """RandomForestBackend always reports single round training."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config()
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # RF is single-round training
    assert outcome["best_round"] == 1
    assert outcome["total_rounds"] == 1
    assert outcome["early_stopped"] is False


def test_rf_backend_different_tree_counts(tmp_path: Path) -> None:
    """RandomForestBackend works with various n_estimators values."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    for n_trees in [5, 10, 25]:
        config = _make_rf_config(n_estimators=n_trees)
        outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)
        assert outcome["best_val_auc"] > 0.5, f"Failed for n_estimators={n_trees}"


def test_rf_backend_different_depths(tmp_path: Path) -> None:
    """RandomForestBackend works with various max_depth values."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    for max_depth in [2, 5, 10]:
        config = _make_rf_config(n_estimators=10, max_depth=max_depth)
        outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)
        assert outcome["best_val_auc"] > 0.5, f"Failed for max_depth={max_depth}"


def test_rf_prepared_exposes_the_native_sklearn_model(tmp_path: Path) -> None:
    """The prepared model surrenders the sklearn ensemble SHAP needs.

    shap.TreeExplainer accepts sklearn ensembles directly and rejects
    wrappers with "Model type not yet supported by TreeExplainer", so the
    native handle has to be reachable or shap_tree cannot work here.
    """
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=5)
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)
    loaded = backend.load(path=outcome["model_path"])

    native = try_extract_native_tree_model(loaded)

    # Names the concrete type: a None here would read as "NoneType" and
    # still fail, so a separate not-None assertion adds nothing.
    assert type(native).__name__ == "RandomForestClassifier"
