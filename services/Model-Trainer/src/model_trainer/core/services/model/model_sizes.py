"""The model sizes each backend implements.

WHY THIS IS NOT INSIDE THE BACKEND PACKAGES
-------------------------------------------
These tables belong, conceptually, next to the code that builds the models. They
cannot live there. ``backends/gpt2/__init__.py`` and ``backends/char_lstm/__init__.py``
re-export their whole backend -- train, evaluate, generate, io, prepare, score --
so importing ANY submodule of those packages executes all of it, including
``train``, which imports ``base_trainer``. That is the cycle
``tests/test_import_cycles.py`` exists to prevent: it used to raise
``ImportError: cannot import name 'BaseTrainer' from partially initialized
module`` on every hf_lm training run.

backend_factory declares each backend's capabilities and must therefore know
which sizes exist, but it must not drag a backend package in to find out -- which
is why every other backend import in that module is deferred into a function.
A neutral module outside ``backends/`` is the only place both can safely read.

WHY THE TABLES EXIST AT ALL
---------------------------
``supported_sizes`` in each capability declaration is DERIVED from these tables
rather than hand-written. When it was a literal maintained alongside the
implementation, the two drifted in both directions at once and nothing compared
them:

* GPT-2 advertised a ``"tiny"`` its table did not implement, so asking for the
  advertised size raised a bare ``KeyError`` from a dict index.
* The same table implemented an ``"xl"`` the registry never advertised.
* char_lstm advertised only ``("small",)`` while its size lookup accepted
  ``tiny`` and ``medium`` too, hiding two working sizes from every caller that
  consulted capabilities.

The test that looked like it covered this asserted the constant against a copy of
itself, so it passed throughout. Deriving from one table makes both drift
directions unrepresentable; the ``capability-sizes`` guard rule keeps the literal
form from returning.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict


class GPT2ModelSizeConfig(TypedDict, total=True):
    """Configuration for GPT-2 model architecture by size."""

    hidden_size: int
    n_layer: int
    n_head: int


class CharLSTMSizeConfig(TypedDict, total=True):
    """Configuration for char-LSTM architecture by size."""

    embed_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float


# GPT-2 model size configurations
# Reference: https://huggingface.co/docs/transformers/model_doc/gpt2
#
# "tiny" is NOT a GPT-2 published size. It exists because the capability registry
# had always advertised it, and because every test that touches this backend
# builds a real transformer: with the suite's 128-token test vocabulary "small"
# measures 85.2M params and 362 MiB of VRAM, and under `pytest -n auto` that is
# ~31 workers each training a real language model on CPU.
#
# 2 layers keeps the block stacking real, and hidden 128 over 2 heads keeps the
# 64-wide head dimension the larger sizes use, so attention shapes stay
# representative rather than degenerate.
GPT2_MODEL_SIZES: Final[dict[str, GPT2ModelSizeConfig]] = {
    "tiny": {"hidden_size": 128, "n_layer": 2, "n_head": 2},  # ~0.4M params; tests
    "small": {"hidden_size": 768, "n_layer": 12, "n_head": 12},  # ~124M params
    "medium": {"hidden_size": 1024, "n_layer": 24, "n_head": 16},  # ~355M params
    "large": {"hidden_size": 1280, "n_layer": 36, "n_head": 20},  # ~774M params
    "xl": {"hidden_size": 1600, "n_layer": 48, "n_head": 25},  # ~1.5B params
}


# These dimensions were an if-chain inside char_lstm's ``_size_to_dims``. As a
# chain they were invisible to anything that wanted to know which sizes exist,
# which is how the capability declaration drifted away from them.
CHAR_LSTM_MODEL_SIZES: Final[dict[str, CharLSTMSizeConfig]] = {
    "tiny": {"embed_dim": 128, "hidden_dim": 256, "num_layers": 2, "dropout": 0.10},
    "small": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.10},
    "medium": {"embed_dim": 384, "hidden_dim": 768, "num_layers": 3, "dropout": 0.10},
}


__all__ = [
    "CHAR_LSTM_MODEL_SIZES",
    "GPT2_MODEL_SIZES",
    "CharLSTMSizeConfig",
    "GPT2ModelSizeConfig",
]
