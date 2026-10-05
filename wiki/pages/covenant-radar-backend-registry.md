---
title: Covenant Radar's backend registry — one interface, two registries, and a count that drifted
tags: [services, covenant-radar, ml, backends, registry, drift]
related:
  - "[[cleargbm-program-charter]]"
  - "[[service-port-map]]"
  - "[[monorepo-discipline]]"
  - "[[covenant-radar-service-architecture]]"
  - "[[covenant-radar-optuna-optimisation]]"
source_paths:
  - libs/covenant_ml/src/covenant_ml/types_regression.py
  - libs/covenant_ml/src/covenant_ml/backends/registry.py
  - libs/covenant_ml/src/covenant_ml/backends/regressor_registry.py
  - libs/covenant_nn/src/covenant_nn/backends/mlp/regressor.py
  - libs/covenant_nn/src/covenant_nn/backends/lstm/regressor.py
  - services/covenant-radar-api/README.md
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/types_regression.py": 02863fc0a9ce27664d2cbd0ed4bbb86d156fdf9b
  "libs/covenant_ml/src/covenant_ml/backends/registry.py": 5ac90a8a212d7de66ab5ddf32123381c4735389a
  "libs/covenant_ml/src/covenant_ml/backends/regressor_registry.py": a94a0e196da7b8ea54b14a5174b65122169f373c
  "libs/covenant_nn/src/covenant_nn/backends/mlp/regressor.py": 46434118c3b3223895bee60871739844c3f84450
  "libs/covenant_nn/src/covenant_nn/backends/lstm/regressor.py": d713fc382efb1018248fe18fa855b77d45d3dde4
  "services/covenant-radar-api/README.md": 34f48ed308cbe2cbffc9e867e2b4a41748c9a90b
provenance:
  - "the two covenant_nn regressors repinned 2026-09-11 on a mechanical argument rather than a re-reading: each diff is +3/-2 and the whole of both is one type alias being inlined -- RegressorProgressCallback became Callable[[RegressionTrainProgress], None] on one parameter. Same signature, no logic, no registry entry touched, and no page in this wiki cites the removed name. This page's claim is about WHICH backends the registry names and that is untouched by a parameter annotation."
  - "all five library pins re-read and repinned 2026-10-05 (board task 418aba78) after 2821e0fae (2026-09-26) turned BackendName and RegressorBackendName from Literal aliases into StrEnums. Every hunk in the five diffs is a string literal replaced by the matching enum member; the classifier registry still registers five, the regressor registry three, and RegressorBackendName still has five members. The README blob is unchanged, so its 'four regressors' is still wrong."
fact_checked: "2026-10-05"
confidence: high
hubs: [services]
---

# Covenant Radar's backend registry

Covenant Radar selects a model backend **per request**, behind one interface.
That is what makes [[cleargbm-program-charter]] a structural comparison rather
than a curated benchmark: ClearGBM sits behind the same protocol as XGBoost and
LightGBM, is trained by the same machinery, and is explained by the same
endpoint.[^clsreg][^readme]

The backends are declared in **two places** — `covenant_ml` and `covenant_nn`,
which register tree/linear and neural families respectively — and that split is
the source of every counting mistake on this page's subject.[^clsreg][^nnbackends]

## Classifiers — seven, across two libraries

`covenant_ml`'s `default_registry` registers five: `xgboost`, `lightgbm`,
`cleargbm`, `logreg`, `random_forest`.[^clsreg] The neural pair — `mlp` and
`lstm` — lives in `covenant_nn/backends/` and is composed in
separately.[^nnbackends] Seven total.

## Regressors — five by the type, three by that registry

The canonical name set is a `StrEnum` in `covenant_ml` (a `Literal` alias until
`2821e0fae` on 2026-09-26, which converted it and the classifier `BackendName`
without changing a member):[^regtype]

```python
class RegressorBackendName(StrEnum):
    XGBOOST_REG = "xgboost_reg"
    LIGHTGBM_REG = "lightgbm_reg"
    CLEARGBM_REG = "cleargbm_reg"
    MLP_REG = "mlp_reg"
    LSTM_REG = "lstm_reg"
```

But `default_regressor_registry` registers only the three tree backends —
`xgboost_reg`, `lightgbm_reg`, `cleargbm_reg`.[^regreg] `mlp_reg` and `lstm_reg`
are implemented in `covenant_nn/backends/{mlp,lstm}/regressor.py` and enter by
the same composition route as their classifier siblings.

**Read the enum, not a registry, when you need the answer.** A registry
holds what one library wired; the type holds what the system accepts — which is
why `regressor_registry.py` can carry three entries while
`RegressorBackendName` admits five without either being wrong.[^regtype][^regreg]

## The count in the README is out of date

`services/covenant-radar-api/README.md` says *"Eleven model backends behind one
interface — seven classifiers … plus four regressors"*, and elsewhere *"all four
`*_reg` backends"*.[^readme] The seven is right. **The four is not: the type
declares five, so the total is twelve.**

Two commits date the divergence. The README's "Eleven model backends" wording
was introduced by `7e9b23d0` on 2026-08-05; the five-name declaration (then a
`Literal`) was introduced to `types_regression.py` by `46b8d4a5` on 2026-08-21, a
commit whose subject is "covenant_ml: 32 of 36 over-ceiling files split by
role".[^commits] Because `46b8d4a5` is a file split, it dates the arrival of the
declaration *in this file* — the names may be older than the file.

This matters beyond tidiness because the figure travels: the README is the
service's front door, and its exact "eleven model backends" phrasing has been
copied outward into portfolio material.[^readme] A number cited more often than
it is recomputed is exactly the kind that goes quietly wrong.

Editing the README to twelve restores agreement only until someone adds a
backend. The durable form is for the README to name the enum as its source,
or for a test to assert the README's count against
`len(RegressorBackendName)` — the same instinct as the guard rules in
[[monorepo-discipline]], which exist because a convention nothing checks is a
convention that drifts.[^regtype]

## Explainability follows the backend

One `/ml/explain` endpoint, three strategies chosen by model type: permutation
importance for anything, SHAP `TreeExplainer` for tree models, and input /
integrated gradients for the neural backends.[^readme] So adding a backend is
not only a registry entry — it lands in a family whose explainer already exists,
which is why the tree/neural split shows up in the type names at all.

[^clsreg]: `libs/covenant_ml/src/covenant_ml/backends/registry.py` —
    `default_registry` registers `BackendName.XGBOOST`, `LIGHTGBM`, `CLEARGBM`,
    `LOGREG`, `RANDOM_FOREST` (lines 77-109). Directory siblings confirm: `backends/`
    holds `cleargbm/`, `lightgbm/`, `logreg/`, `random_forest/`, `xgboost/`.
[^nnbackends]: `libs/covenant_nn/src/covenant_nn/backends/mlp/regressor.py`
    /`create_mlp_regressor_backend` and
    `libs/covenant_nn/src/covenant_nn/backends/lstm/regressor.py`
    /`create_lstm_regressor_backend`; their classifier siblings
    `create_mlp_backend` and `create_lstm_backend` sit in the same two package
    directories.
[^commits]: `git log -S'Eleven model backends' -- services/covenant-radar-api/README.md`
    → `7e9b23d0`, 2026-08-05, "covenant-radar-api README: lead with the
    multi-domain protocol, not the loan origin".
    `git log -S'"mlp_reg"' -- libs/covenant_ml/src/covenant_ml/types_regression.py`
    → `46b8d4a5`, 2026-08-21, "covenant_ml: 32 of 36 over-ceiling files split by
    role; 4 held for a concurrent session".
[^regtype]: `libs/covenant_ml/src/covenant_ml/types_regression.py:129` — `class RegressorBackendName(StrEnum)`.
[^regreg]: `libs/covenant_ml/src/covenant_ml/backends/regressor_registry.py`
    :123-134 — three `reg.register(...)` calls.
[^readme]: `services/covenant-radar-api/README.md:38-45` (backend count and
    explainability strategies) and `:123` ("all four `*_reg` backends").
