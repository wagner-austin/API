"""Tests for optimizer search space factory functions."""

from __future__ import annotations

from covenant_ml.optimizer.search_spaces import (
    make_lstm_default_space,
    make_lstm_focused_space,
    make_mlp_default_space,
    make_mlp_focused_space,
)


def test_make_mlp_default_space_returns_complete_space() -> None:
    """make_mlp_default_space returns space with all required parameters."""
    space = make_mlp_default_space()

    assert "n_layers" in space
    assert "hidden_size" in space
    assert "learning_rate" in space
    assert "dropout" in space
    assert "batch_size" in space


def test_make_mlp_default_space_param_types() -> None:
    """make_mlp_default_space uses correct param types."""
    space = make_mlp_default_space()

    assert space["n_layers"]["param_type"] == "int"
    assert space["hidden_size"]["param_type"] == "categorical_int"
    assert space["learning_rate"]["param_type"] == "float"
    assert space["dropout"]["param_type"] == "float"
    assert space["batch_size"]["param_type"] == "categorical_int"


def test_make_mlp_default_space_ranges() -> None:
    """make_mlp_default_space has sensible default ranges."""
    space = make_mlp_default_space()

    n_layers = space["n_layers"]
    if n_layers["param_type"] == "int":
        assert n_layers["low"] == 1
        assert n_layers["high"] == 4

    hidden_size = space["hidden_size"]
    if hidden_size["param_type"] == "categorical_int":
        assert hidden_size["choices"] == (64, 128, 256, 512)

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["log_scale"] is True
        assert lr["low"] == 1e-5
        assert lr["high"] == 1e-2

    dropout = space["dropout"]
    if dropout["param_type"] == "float":
        assert dropout["low"] == 0.0
        assert dropout["high"] == 0.5


def test_make_mlp_focused_space_narrows_around_best() -> None:
    """make_mlp_focused_space creates narrower ranges around best values."""
    space = make_mlp_focused_space(best_n_layers=2, best_hidden_size=128, best_learning_rate=1e-3)

    n_layers = space["n_layers"]
    if n_layers["param_type"] == "int":
        assert n_layers["low"] == 1  # max(1, 2-1)
        assert n_layers["high"] == 3  # min(5, 2+1)

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["low"] == 1e-4  # 1e-3 * 0.1
        assert lr["high"] == 1e-2  # 1e-3 * 10.0


def test_make_mlp_focused_space_clamps_layers() -> None:
    """make_mlp_focused_space clamps layers to valid range."""
    space_low = make_mlp_focused_space(
        best_n_layers=1, best_hidden_size=64, best_learning_rate=1e-3
    )
    n_layers_low = space_low["n_layers"]
    if n_layers_low["param_type"] == "int":
        assert n_layers_low["low"] == 1  # max(1, 1-1) = 1

    space_high = make_mlp_focused_space(
        best_n_layers=5, best_hidden_size=64, best_learning_rate=1e-3
    )
    n_layers_high = space_high["n_layers"]
    if n_layers_high["param_type"] == "int":
        assert n_layers_high["high"] == 5  # min(5, 5+1) = 5


def test_make_mlp_focused_space_clamps_learning_rate() -> None:
    """make_mlp_focused_space clamps learning rate to valid range."""
    space_low = make_mlp_focused_space(
        best_n_layers=2, best_hidden_size=128, best_learning_rate=1e-7
    )
    lr_low = space_low["learning_rate"]
    if lr_low["param_type"] == "float":
        assert lr_low["low"] == 1e-6  # max(1e-6, 1e-7*0.1) = 1e-6

    space_high = make_mlp_focused_space(
        best_n_layers=2, best_hidden_size=128, best_learning_rate=0.5
    )
    lr_high = space_high["learning_rate"]
    if lr_high["param_type"] == "float":
        assert lr_high["high"] == 0.1  # min(0.1, 0.5*10) = 0.1


def test_make_mlp_focused_space_hidden_size_fallback() -> None:
    """make_mlp_focused_space falls back to best hidden size when no matches."""
    # Using a very small value (10) ensures no sizes pass the abs(size - best) <= best check
    space = make_mlp_focused_space(best_n_layers=2, best_hidden_size=10, best_learning_rate=1e-3)
    hidden = space["hidden_size"]
    if hidden["param_type"] == "categorical_int":
        assert hidden["choices"] == (10,)


def test_make_lstm_default_space_returns_complete_space() -> None:
    """make_lstm_default_space returns space with all required parameters."""
    space = make_lstm_default_space()

    assert "hidden_size" in space
    assert "num_layers" in space
    assert "dropout" in space
    assert "learning_rate" in space
    assert "batch_size" in space


def test_make_lstm_default_space_param_types() -> None:
    """make_lstm_default_space uses correct param types."""
    space = make_lstm_default_space()

    assert space["hidden_size"]["param_type"] == "categorical_int"
    assert space["num_layers"]["param_type"] == "int"
    assert space["dropout"]["param_type"] == "float"
    assert space["learning_rate"]["param_type"] == "float"
    assert space["batch_size"]["param_type"] == "categorical_int"


def test_make_lstm_default_space_ranges() -> None:
    """make_lstm_default_space has sensible default ranges."""
    space = make_lstm_default_space()

    hidden = space["hidden_size"]
    if hidden["param_type"] == "categorical_int":
        assert hidden["choices"] == (64, 128, 256)

    num_layers = space["num_layers"]
    if num_layers["param_type"] == "int":
        assert num_layers["low"] == 1
        assert num_layers["high"] == 3

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["log_scale"] is True
        assert lr["low"] == 1e-5
        assert lr["high"] == 1e-2

    batch = space["batch_size"]
    if batch["param_type"] == "categorical_int":
        assert batch["choices"] == (16, 32, 64)


def test_make_lstm_focused_space_narrows_around_best() -> None:
    """make_lstm_focused_space creates narrower ranges around best values."""
    space = make_lstm_focused_space(
        best_hidden_size=128, best_num_layers=2, best_learning_rate=1e-3
    )

    num_layers = space["num_layers"]
    if num_layers["param_type"] == "int":
        assert num_layers["low"] == 1  # max(1, 2-1)
        assert num_layers["high"] == 3  # min(4, 2+1)

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["low"] == 1e-4  # 1e-3 * 0.1
        assert lr["high"] == 1e-2  # 1e-3 * 10.0


def test_make_lstm_focused_space_clamps_layers() -> None:
    """make_lstm_focused_space clamps num_layers to valid range."""
    space_low = make_lstm_focused_space(
        best_hidden_size=64, best_num_layers=1, best_learning_rate=1e-3
    )
    layers_low = space_low["num_layers"]
    if layers_low["param_type"] == "int":
        assert layers_low["low"] == 1  # max(1, 1-1) = 1

    space_high = make_lstm_focused_space(
        best_hidden_size=64, best_num_layers=4, best_learning_rate=1e-3
    )
    layers_high = space_high["num_layers"]
    if layers_high["param_type"] == "int":
        assert layers_high["high"] == 4  # min(4, 4+1) = 4


def test_make_lstm_focused_space_clamps_learning_rate() -> None:
    """make_lstm_focused_space clamps learning rate to valid range."""
    space_low = make_lstm_focused_space(
        best_hidden_size=64, best_num_layers=2, best_learning_rate=1e-7
    )
    lr_low = space_low["learning_rate"]
    if lr_low["param_type"] == "float":
        assert lr_low["low"] == 1e-6  # max(1e-6, 1e-7*0.1) = 1e-6

    space_high = make_lstm_focused_space(
        best_hidden_size=64, best_num_layers=2, best_learning_rate=0.5
    )
    lr_high = space_high["learning_rate"]
    if lr_high["param_type"] == "float":
        assert lr_high["high"] == 0.1  # min(0.1, 0.5*10) = 0.1


def test_make_lstm_focused_space_hidden_size_fallback() -> None:
    """make_lstm_focused_space falls back to best hidden size when no matches."""
    # Using a very small value (10) ensures no sizes pass the abs(size - best) <= best check
    space = make_lstm_focused_space(best_hidden_size=10, best_num_layers=2, best_learning_rate=1e-3)
    hidden = space["hidden_size"]
    if hidden["param_type"] == "categorical_int":
        assert hidden["choices"] == (10,)
