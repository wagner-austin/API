---
title: ClearGBM's Python-Rust boundary after the shim removal
tags: [ml, cleargbm, rust, pyo3, packaging, maturin]
related:
  - "[[cleargbm-histogram-split-path]]"
  - "[[cleargbm-perf-leaf-wise-growth]]"
source_paths:
  - libs/cleargbm/src/cleargbm/ensemble.py
  - libs/cleargbm/src/cleargbm/_rust.py
  - libs/cleargbm_rs/pyproject.toml
  - libs/cleargbm_rs/Cargo.toml
  - libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py
  - libs/covenant_ml/src/covenant_ml/explainers/cleargbm_shap.py
source_git_blobs:
  "libs/cleargbm/src/cleargbm/ensemble.py": d855968ef0fd5be83716ae0a331765004dee690f
  "libs/cleargbm/src/cleargbm/_rust.py": 5f8ba08dec7197ffe2a203a44385d3337f0b47db
  "libs/cleargbm_rs/pyproject.toml": 4082850e2dbc7a7d9b962066ac5d1b44d46ac2b2
  "libs/cleargbm_rs/Cargo.toml": 472c6cc568ce46dba53caba924b7fa1b7a3cf0d8
  "libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py": 6f29219931d6bb9d551b5bb779732244686d2dcf
  "libs/covenant_ml/src/covenant_ml/explainers/cleargbm_shap.py": b1b17fa189ff6fae7a1af1527e186aff840802e5
fact_checked: "2026-08-17"
confidence: high
hubs: [libs]
---

# ClearGBM's Python-Rust boundary after the shim removal

As of commit `ea7835d2` (2026-08-17) there is **no hand-maintained Python inside the
Rust crate**. Two changes four weeks apart produced this state, and they are distinct:
`f7c61172` (2026-07-21) deleted the Python *compute path* (~10k lines — histogram,
split, tree, losses, parallel, explain, buffers); `ea7835d2` deleted the Python
*packaging shim* (a 50-line PEP 562 `__getattr__` forwarder that existed only because
`[tool.maturin] python-source = "python"` nested the extension at
`cleargbm_rs.cleargbm_rs`). Dropping `python-source` lets maturin build `cleargbm_rs`
as a top-level module and generate its own wrapper into site-packages[^2].

## The dependency points one way

The crate does not need Python as a library consumer: `crate-type = ["cdylib", "rlib"]`
makes it both an extension module and a plain Rust lib[^1], with the pyo3 dependency at
0.27.2 and `extension-module` as an opt-in feature[^2]. The *Python* package hard-needs
the Rust: `_rust.py` states that if `cleargbm_rs` is not installed the module raises
`ImportError` at import time, and that "there is no Python fallback"[^3].

## What each Python file is for

`_rust.py` (201 lines) imports the extension exactly once and pins each native callable
to a `Protocol` type so mypy sees precise signatures instead of `Any` leaking from the
dynamic import[^4]. `ensemble.py` (222 lines) is the whole public training surface:
validate a `GradientBoostingConfig`, marshal it to a dict shaped for
`cleargbm_rs.train_gradient_boosting_rs` via `_config_to_rust_dict`, and make one
call[^5]. The remaining ~1,800 lines of the package are the typed model/tree/JSON/
explain surface (`_types_*.py`), not compute.

## Why the boundary is shaped this way

Signature drift between hand-written Python stubs and Rust exports was the failure mode
the shim removal closed: with maturin generating the wrapper, the only Python that
mirrors Rust signatures is `_rust.py`'s Protocol layer, which mypy checks against every
caller[^4]. The `ea7835d2` verification rebuilt the extension into all three consuming
venvs (cleargbm, cleargbm_rs dev, covenant_ml) and ran each `make check` green:
4387/4387 coverage segments in the crate, 189 tests in cleargbm, 2240 in covenant_ml.

## Consumers

`covenant_ml` reaches the extension the same way `cleargbm/_rust.py` does, by importing
`cleargbm_rs` as a top-level module (its `backends/cleargbm/backend.py` and
`explainers/cleargbm_shap.py` were repointed in `ea7835d2`)[^6]. No consumer imports
through a nested `cleargbm_rs.cleargbm_rs` path any more; that spelling died with the
shim.

[^1]: libs/cleargbm_rs/Cargo.toml:14 — `crate-type = ["cdylib", "rlib"]`.
[^2]: libs/cleargbm_rs/Cargo.toml:18,21 — `extension-module = ["pyo3/extension-module"]`; `pyo3 = { version = "0.27.2" }`. Build backend and module name: libs/cleargbm_rs/pyproject.toml:3,32-33 — `build-backend = "maturin"`, `features = ["extension-module"]`, `module-name = "cleargbm_rs"`, with no `python-source` key remaining.
[^3]: libs/cleargbm/src/cleargbm/_rust.py:10-11 — the docstring's guarantee, verbatim on line 11: "there is no Python fallback".
[^4]: libs/cleargbm/src/cleargbm/_rust.py:1-15 — module docstring: built by maturin as a top-level module, imported exactly once, each callable pinned to a Protocol; strict typing, no `Any`, no `cast`.
[^6]: libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py:121 and libs/covenant_ml/src/covenant_ml/explainers/cleargbm_shap.py:72 — both read `_native_mod: types.ModuleType = __import__("cleargbm_rs")`.
[^5]: libs/cleargbm/src/cleargbm/ensemble.py:24-29,63,139-141 — the import from `cleargbm._rust`, `_config_to_rust_dict`, and the single `train_gradient_boosting_rs(...)` call that is the entire hand-off.
