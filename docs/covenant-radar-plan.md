# Covenant Radar - Implementation Plan

## Progress Summary

| Milestone | Status | Tests | Coverage |
|-----------|--------|-------|----------|
| 1. covenant_domain | ✅ Complete | 113 | 100% |
| 2. covenant_ml | ✅ Complete | 18 | 100% |
| 3. covenant_persistence | ✅ Complete | 96 | 100% |
| 4. Service Shell | ✅ Complete | - | 100% |
| 5. CRUD Endpoints | ✅ Complete | 79 | 100% |
| 6. ML Endpoints | ✅ Complete | 146 | 100% |
| 7. Documentation | ✅ Complete | - | - |

---

## Overview

Covenant Radar is a loan covenant monitoring and breach prediction system. It stores loan deals and covenant rules, ingests borrower financial data, computes covenant status using deterministic rules, and serves an XGBoost-based risk model for predicting future breaches.

**Key Differentiator from Model-Trainer:** This is tabular ML (gradient boosted trees), not sequence modeling. Requires a standalone service.

---

## Architecture

### New Components

```
libs/
  covenant_domain/       # Pure business logic (TypedDict models, rule engine)
  covenant_ml/           # XGBoost training and prediction
  covenant_persistence/  # PostgreSQL repositories

services/
  covenant-radar-api/    # FastAPI service (port 8007)
```

### Reused Components

| Existing Library | Usage in Covenant Radar |
|------------------|------------------------|
| `platform_core` | Logging, config, error handling, `DataBankClient`, `json_utils` |
| `platform_workers` | Redis Protocols, RQ harness for training jobs |
| `platform_ml` | `ArtifactStore` for model storage, manifest patterns |

---

## Strict Typing Standards

All code follows monorepo standards:

```toml
[tool.mypy]
strict = true
disallow_any_unimported = true
disallow_any_expr = true
disallow_any_decorated = true
disallow_any_explicit = true
```

**Banned:**
- `typing.Any`
- `typing.cast`
- `# type: ignore`
- `.pyi` stub files
- `dataclasses`
- `try/except` in core logic (exceptions propagate)

---

## New Libraries

### libs/covenant_domain/

Pure business logic with no IO, no framework dependencies. TypedDict only.

```
covenant_domain/
├── __init__.py
├── py.typed
├── models.py          # Deal, Covenant, Measurement, CovenantResult, type aliases
├── decode.py          # JSONObject → TypedDict decoders (uses platform_core.json_utils)
├── encode.py          # TypedDict → dict[str, JSONValue] encoders
├── rules.py           # Deterministic covenant evaluation
├── features.py        # Feature extraction for ML
└── formula_parser.py  # Shunting-yard expression evaluator
```

#### models.py

Type aliases are inlined as Literals for strict typing without separate types.py file.

```python
from __future__ import annotations

from typing import Literal, TypedDict


class DealId(TypedDict, total=True):
    """Immutable deal identifier."""
    value: str  # UUID string


class Deal(TypedDict, total=True):
    """Loan deal record. Immutable by convention."""
    id: DealId
    name: str
    borrower: str
    sector: str
    region: str
    commitment_amount_cents: int  # Store as cents to avoid Decimal
    currency: str
    maturity_date_iso: str  # ISO 8601 date string


class CovenantId(TypedDict, total=True):
    """Immutable covenant identifier."""
    value: str  # UUID string


class Covenant(TypedDict, total=True):
    """Covenant rule definition. Immutable by convention."""
    id: CovenantId
    deal_id: DealId
    name: str
    formula: str  # e.g., "total_debt / ebitda"
    threshold_value_scaled: int  # Scaled integer (multiply by 1_000_000)
    threshold_direction: Literal["<=", ">="]
    frequency: Literal["QUARTERLY", "ANNUAL"]


class Measurement(TypedDict, total=True):
    """Financial metric measurement for a period."""
    deal_id: DealId
    period_start_iso: str  # ISO 8601 date
    period_end_iso: str    # ISO 8601 date
    metric_name: str       # e.g., "total_debt", "ebitda"
    metric_value_scaled: int  # Scaled integer (multiply by 1_000_000)


class CovenantResult(TypedDict, total=True):
    """Computed covenant status for a period."""
    covenant_id: CovenantId
    period_start_iso: str
    period_end_iso: str
    calculated_value_scaled: int
    status: Literal["OK", "NEAR_BREACH", "BREACH"]
```

#### decode.py

Uses centralized `platform_core.json_utils` helpers (`require_str`, `require_int`, `require_dict`, `JSONObject`, `JSONTypeError`) for standardized JSON validation. Domain-specific literal validators are thin wrappers.

```python
from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_dict,
    require_int,
    require_str,
)

from .models import Covenant, CovenantId, CovenantResult, Deal, DealId, Measurement


def _require_covenant_status(data: JSONObject, key: str) -> Literal["OK", "NEAR_BREACH", "BREACH"]:
    """Extract and validate CovenantStatus literal."""
    value = require_str(data, key)
    if value == "OK":
        return "OK"
    if value == "NEAR_BREACH":
        return "NEAR_BREACH"
    if value == "BREACH":
        return "BREACH"
    raise JSONTypeError(f"Invalid CovenantStatus: {value}")


def _require_threshold_direction(data: JSONObject, key: str) -> Literal["<=", ">="]:
    """Extract and validate ThresholdDirection literal."""
    value = require_str(data, key)
    if value == "<=":
        return "<="
    if value == ">=":
        return ">="
    raise JSONTypeError(f"Invalid ThresholdDirection: {value}")


def _require_frequency(data: JSONObject, key: str) -> Literal["QUARTERLY", "ANNUAL"]:
    """Extract and validate CovenantFrequency literal."""
    value = require_str(data, key)
    if value == "QUARTERLY":
        return "QUARTERLY"
    if value == "ANNUAL":
        return "ANNUAL"
    raise JSONTypeError(f"Invalid CovenantFrequency: {value}")


def decode_deal_id(data: JSONObject) -> DealId:
    """Decode DealId from JSON dict."""
    return DealId(value=require_str(data, "value"))


def decode_deal(data: JSONObject) -> Deal:
    """Decode Deal from JSON dict. Raises on invalid data."""
    return Deal(
        id=decode_deal_id(require_dict(data, "id")),
        name=require_str(data, "name"),
        borrower=require_str(data, "borrower"),
        sector=require_str(data, "sector"),
        region=require_str(data, "region"),
        commitment_amount_cents=require_int(data, "commitment_amount_cents"),
        currency=require_str(data, "currency"),
        maturity_date_iso=require_str(data, "maturity_date_iso"),
    )


def decode_covenant_id(data: JSONObject) -> CovenantId:
    """Decode CovenantId from JSON dict."""
    return CovenantId(value=require_str(data, "value"))


def decode_covenant(data: JSONObject) -> Covenant:
    """Decode Covenant from JSON dict. Raises on invalid data."""
    return Covenant(
        id=decode_covenant_id(require_dict(data, "id")),
        deal_id=decode_deal_id(require_dict(data, "deal_id")),
        name=require_str(data, "name"),
        formula=require_str(data, "formula"),
        threshold_value_scaled=require_int(data, "threshold_value_scaled"),
        threshold_direction=_require_threshold_direction(data, "threshold_direction"),
        frequency=_require_frequency(data, "frequency"),
    )


def decode_measurement(data: JSONObject) -> Measurement:
    """Decode Measurement from JSON dict. Raises on invalid data."""
    return Measurement(
        deal_id=decode_deal_id(require_dict(data, "deal_id")),
        period_start_iso=require_str(data, "period_start_iso"),
        period_end_iso=require_str(data, "period_end_iso"),
        metric_name=require_str(data, "metric_name"),
        metric_value_scaled=require_int(data, "metric_value_scaled"),
    )


def decode_covenant_result(data: JSONObject) -> CovenantResult:
    """Decode CovenantResult from JSON dict. Raises on invalid data."""
    return CovenantResult(
        covenant_id=decode_covenant_id(require_dict(data, "covenant_id")),
        period_start_iso=require_str(data, "period_start_iso"),
        period_end_iso=require_str(data, "period_end_iso"),
        calculated_value_scaled=require_int(data, "calculated_value_scaled"),
        status=_require_covenant_status(data, "status"),
    )
```

#### encode.py

```python
from __future__ import annotations

from platform_core.json_utils import JSONValue
from .models import Deal, DealId, Covenant, CovenantId, Measurement, CovenantResult


def encode_deal_id(deal_id: DealId) -> dict[str, JSONValue]:
    """Encode DealId to JSON-serializable dict."""
    return {"value": deal_id["value"]}


def encode_deal(deal: Deal) -> dict[str, JSONValue]:
    """Encode Deal to JSON-serializable dict."""
    return {
        "id": encode_deal_id(deal["id"]),
        "name": deal["name"],
        "borrower": deal["borrower"],
        "sector": deal["sector"],
        "region": deal["region"],
        "commitment_amount_cents": deal["commitment_amount_cents"],
        "currency": deal["currency"],
        "maturity_date_iso": deal["maturity_date_iso"],
    }


def encode_covenant_id(covenant_id: CovenantId) -> dict[str, JSONValue]:
    """Encode CovenantId to JSON-serializable dict."""
    return {"value": covenant_id["value"]}


def encode_covenant(covenant: Covenant) -> dict[str, JSONValue]:
    """Encode Covenant to JSON-serializable dict."""
    return {
        "id": encode_covenant_id(covenant["id"]),
        "deal_id": encode_deal_id(covenant["deal_id"]),
        "name": covenant["name"],
        "formula": covenant["formula"],
        "threshold_value_scaled": covenant["threshold_value_scaled"],
        "threshold_direction": covenant["threshold_direction"],
        "frequency": covenant["frequency"],
    }


def encode_measurement(measurement: Measurement) -> dict[str, JSONValue]:
    """Encode Measurement to JSON-serializable dict."""
    return {
        "deal_id": encode_deal_id(measurement["deal_id"]),
        "period_start_iso": measurement["period_start_iso"],
        "period_end_iso": measurement["period_end_iso"],
        "metric_name": measurement["metric_name"],
        "metric_value_scaled": measurement["metric_value_scaled"],
    }


def encode_covenant_result(result: CovenantResult) -> dict[str, JSONValue]:
    """Encode CovenantResult to JSON-serializable dict."""
    return {
        "covenant_id": encode_covenant_id(result["covenant_id"]),
        "period_start_iso": result["period_start_iso"],
        "period_end_iso": result["period_end_iso"],
        "calculated_value_scaled": result["calculated_value_scaled"],
        "status": result["status"],
    }
```

#### formula_parser.py

Shunting-yard algorithm for safe expression evaluation (no `eval()`).

```python
from __future__ import annotations

from typing import Mapping, Sequence

# Token types
_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})
_PRECEDENCE: dict[str, int] = {"+": 1, "-": 1, "*": 2, "/": 2}


class FormulaParseError(Exception):
    """Raised when formula parsing fails."""


class FormulaEvalError(Exception):
    """Raised when formula evaluation fails."""


def _tokenize(formula: str) -> Sequence[str]:
    """
    Tokenize formula into operators, parentheses, and metric names.

    Raises FormulaParseError on invalid characters.
    """
    tokens: list[str] = []
    current: list[str] = []

    for char in formula:
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        elif char in _OPERATORS or char in ("(", ")"):
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
        elif char.isalnum() or char == "_":
            current.append(char)
        else:
            raise FormulaParseError(f"Invalid character in formula: {char!r}")

    if current:
        tokens.append("".join(current))

    return tokens


def _to_rpn(tokens: Sequence[str]) -> Sequence[str]:
    """
    Convert infix tokens to Reverse Polish Notation using shunting-yard.

    Raises FormulaParseError on mismatched parentheses.
    """
    output: list[str] = []
    operator_stack: list[str] = []

    for token in tokens:
        if token in _OPERATORS:
            while (
                operator_stack
                and operator_stack[-1] != "("
                and operator_stack[-1] in _OPERATORS
                and _PRECEDENCE[operator_stack[-1]] >= _PRECEDENCE[token]
            ):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack and operator_stack[-1] != "(":
                output.append(operator_stack.pop())
            if not operator_stack:
                raise FormulaParseError("Mismatched parentheses")
            operator_stack.pop()  # Remove the "("
        else:
            # Metric name or number
            output.append(token)

    while operator_stack:
        op = operator_stack.pop()
        if op == "(":
            raise FormulaParseError("Mismatched parentheses")
        output.append(op)

    return output


def _evaluate_rpn(rpn: Sequence[str], metrics: Mapping[str, int]) -> int:
    """
    Evaluate RPN expression with scaled integer arithmetic.

    Raises:
        KeyError: Unknown metric name
        FormulaEvalError: Division by zero or invalid expression
    """
    stack: list[int] = []

    for token in rpn:
        if token in _OPERATORS:
            if len(stack) < 2:
                raise FormulaEvalError("Invalid expression: insufficient operands")
            b = stack.pop()
            a = stack.pop()
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                # For multiplication of scaled values, adjust scale
                stack.append((a * b) // 1_000_000)
            elif token == "/":
                if b == 0:
                    raise FormulaEvalError("Division by zero")
                # For division of scaled values, adjust scale
                stack.append((a * 1_000_000) // b)
        else:
            # Try to parse as integer literal first
            if token.isdigit():
                stack.append(int(token) * 1_000_000)
            else:
                # Metric name lookup - raises KeyError if not found
                stack.append(metrics[token])

    if len(stack) != 1:
        raise FormulaEvalError("Invalid expression: wrong number of values")

    return stack[0]


def evaluate_formula(formula: str, metrics: Mapping[str, int]) -> int:
    """
    Evaluate arithmetic formula against metric values.

    All values are scaled integers (multiply by 1_000_000 for 6 decimal places).

    Supported operators: +, -, *, /
    Supported: parentheses, metric names, integer literals

    Raises:
        FormulaParseError: Invalid formula syntax
        FormulaEvalError: Division by zero or invalid expression
        KeyError: Unknown metric name
    """
    tokens = _tokenize(formula)
    rpn = _to_rpn(tokens)
    return _evaluate_rpn(rpn, metrics)
```

#### rules.py

```python
from __future__ import annotations

from typing import Mapping, Sequence

from .models import Covenant, Measurement, CovenantResult, CovenantId
from .types import CovenantStatus
from .formula_parser import evaluate_formula


def classify_status(
    threshold_value_scaled: int,
    threshold_direction: str,
    calculated_value_scaled: int,
    tolerance_ratio_scaled: int,
) -> CovenantStatus:
    """
    Classify covenant status based on calculated value vs threshold.

    tolerance_ratio_scaled: The tolerance band width (e.g., 100_000 for 10%).
    All values are scaled integers (multiply by 1_000_000).

    For "<=": BREACH if calculated > threshold
              NEAR_BREACH if calculated > threshold * (1 - tolerance)
              OK otherwise

    For ">=": BREACH if calculated < threshold
              NEAR_BREACH if calculated < threshold * (1 + tolerance)
              OK otherwise
    """
    # Calculate tolerance band
    tolerance_amount = (threshold_value_scaled * tolerance_ratio_scaled) // 1_000_000

    if threshold_direction == "<=":
        if calculated_value_scaled > threshold_value_scaled:
            return "BREACH"
        if calculated_value_scaled > threshold_value_scaled - tolerance_amount:
            return "NEAR_BREACH"
        return "OK"

    # threshold_direction == ">="
    if calculated_value_scaled < threshold_value_scaled:
        return "BREACH"
    if calculated_value_scaled < threshold_value_scaled + tolerance_amount:
        return "NEAR_BREACH"
    return "OK"


def _build_metrics_for_period(
    measurements: Sequence[Measurement],
    period_start_iso: str,
    period_end_iso: str,
) -> dict[str, int]:
    """
    Build metric name → scaled value mapping for a specific period.

    Raises ValueError if duplicate metric names for same period.
    """
    metrics: dict[str, int] = {}

    for m in measurements:
        if m["period_start_iso"] == period_start_iso and m["period_end_iso"] == period_end_iso:
            name = m["metric_name"]
            if name in metrics:
                raise ValueError(f"Duplicate metric {name} for period {period_start_iso}")
            metrics[name] = m["metric_value_scaled"]

    return metrics


def evaluate_covenant_for_period(
    covenant: Covenant,
    period_start_iso: str,
    period_end_iso: str,
    measurements: Sequence[Measurement],
    tolerance_ratio_scaled: int,
) -> CovenantResult:
    """
    Evaluate a covenant for a specific period.

    Raises:
        KeyError: Required metric missing from measurements
        FormulaParseError: Invalid formula
        FormulaEvalError: Division by zero
        ValueError: Duplicate metrics
    """
    metrics = _build_metrics_for_period(measurements, period_start_iso, period_end_iso)

    calculated_value_scaled = evaluate_formula(covenant["formula"], metrics)

    status = classify_status(
        threshold_value_scaled=covenant["threshold_value_scaled"],
        threshold_direction=covenant["threshold_direction"],
        calculated_value_scaled=calculated_value_scaled,
        tolerance_ratio_scaled=tolerance_ratio_scaled,
    )

    return CovenantResult(
        covenant_id=covenant["id"],
        period_start_iso=period_start_iso,
        period_end_iso=period_end_iso,
        calculated_value_scaled=calculated_value_scaled,
        status=status,
    )


def evaluate_all_covenants_for_period(
    covenants: Sequence[Covenant],
    period_start_iso: str,
    period_end_iso: str,
    measurements: Sequence[Measurement],
    tolerance_ratio_scaled: int,
) -> Sequence[CovenantResult]:
    """
    Evaluate all covenants for a period. Returns results in same order as covenants.

    Raises on first failure - no partial results.
    """
    results: list[CovenantResult] = []

    for covenant in covenants:
        result = evaluate_covenant_for_period(
            covenant=covenant,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            measurements=measurements,
            tolerance_ratio_scaled=tolerance_ratio_scaled,
        )
        results.append(result)

    return results
```

#### features.py

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from .models import CovenantResult, Deal


class LoanFeatures(TypedDict, total=True):
    """Feature vector for ML risk prediction. All values are floats."""
    debt_to_ebitda: float
    interest_cover: float
    current_ratio: float
    leverage_change_1p: float   # vs 1 period ago
    leverage_change_4p: float   # vs 4 periods ago
    sector_encoded: int
    region_encoded: int
    near_breach_count_4p: int   # near breaches in last 4 periods


class RiskPrediction(TypedDict, total=True):
    """ML model prediction output."""
    probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]


# Feature column order for numpy array conversion
FEATURE_ORDER: tuple[str, ...] = (
    "debt_to_ebitda",
    "interest_cover",
    "current_ratio",
    "leverage_change_1p",
    "leverage_change_4p",
    "sector_encoded",
    "region_encoded",
    "near_breach_count_4p",
)


def classify_risk_tier(probability: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    """Map probability to risk tier. Pure function."""
    if probability < 0.3:
        return "LOW"
    if probability < 0.7:
        return "MEDIUM"
    return "HIGH"


def _count_near_breaches(results: Sequence[CovenantResult], periods: int) -> int:
    """Count NEAR_BREACH status in last N periods."""
    count = 0
    for result in results[:periods]:
        if result["status"] == "NEAR_BREACH":
            count += 1
    return count


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def extract_features(
    deal: Deal,
    metrics_current: Mapping[str, int],
    metrics_1p_ago: Mapping[str, int],
    metrics_4p_ago: Mapping[str, int],
    recent_results: Sequence[CovenantResult],
    sector_encoder: Mapping[str, int],
    region_encoder: Mapping[str, int],
) -> LoanFeatures:
    """
    Extract ML features from domain data. Pure function.

    metrics_*: metric_name -> scaled_value (multiply by 1_000_000)
    recent_results: Last N covenant results, sorted by period descending

    Raises KeyError if required metrics missing or sector/region unknown.
    """
    # Current period ratios (convert from scaled int to float)
    debt = metrics_current["total_debt"] / 1_000_000
    ebitda = metrics_current["ebitda"] / 1_000_000
    interest = metrics_current["interest_expense"] / 1_000_000
    current_assets = metrics_current["current_assets"] / 1_000_000
    current_liab = metrics_current["current_liabilities"] / 1_000_000

    debt_to_ebitda = _safe_divide(debt, ebitda)
    interest_cover = _safe_divide(ebitda, interest)
    current_ratio = _safe_divide(current_assets, current_liab)

    # Historical leverage for change calculation
    debt_1p = metrics_1p_ago.get("total_debt", 0) / 1_000_000
    ebitda_1p = metrics_1p_ago.get("ebitda", 1_000_000) / 1_000_000
    leverage_1p = _safe_divide(debt_1p, ebitda_1p)

    debt_4p = metrics_4p_ago.get("total_debt", 0) / 1_000_000
    ebitda_4p = metrics_4p_ago.get("ebitda", 1_000_000) / 1_000_000
    leverage_4p = _safe_divide(debt_4p, ebitda_4p)

    leverage_change_1p = debt_to_ebitda - leverage_1p
    leverage_change_4p = debt_to_ebitda - leverage_4p

    # Categorical encoding (raises KeyError if unknown)
    sector_encoded = sector_encoder[deal["sector"]]
    region_encoded = region_encoder[deal["region"]]

    # Near breach count
    near_breach_count = _count_near_breaches(recent_results, 4)

    return LoanFeatures(
        debt_to_ebitda=debt_to_ebitda,
        interest_cover=interest_cover,
        current_ratio=current_ratio,
        leverage_change_1p=leverage_change_1p,
        leverage_change_4p=leverage_change_4p,
        sector_encoded=sector_encoded,
        region_encoded=region_encoded,
        near_breach_count_4p=near_breach_count,
    )
```

---

### libs/covenant_ml/ ✅ IMPLEMENTED

XGBoost training and prediction with Protocol-based dynamic imports.

```
covenant_ml/
├── __init__.py
├── py.typed
├── types.py           # TrainConfig, XGBModelProtocol, Proba2DProtocol, Factory/Loader Protocols
├── trainer.py         # train_model(), save_model()
└── predictor.py       # load_model(), predict_probabilities()
```

**Dependencies:** xgboost ^3.1, scikit-learn ^1.7, numpy ^2.3

#### types.py

```python
from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray


class TrainConfig(TypedDict, total=True):
    """Configuration for XGBoost model training."""
    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float
    colsample_bytree: float
    random_state: int


class Proba2DProtocol(Protocol):
    """Protocol for 2D probability array from predict_proba."""
    @property
    def shape(self) -> tuple[int, int]: ...
    def __getitem__(self, idx: tuple[int, int]) -> float: ...


class XGBModelProtocol(Protocol):
    """Protocol for XGBoost classifier interface."""
    def fit(self, x_features: NDArray[np.float64], y_labels: NDArray[np.int64]) -> "XGBModelProtocol": ...
    def predict_proba(self, x_features: NDArray[np.float64]) -> Proba2DProtocol: ...
    def save_model(self, fname: str) -> None: ...
    def load_model(self, fname: str) -> None: ...


class XGBClassifierFactory(Protocol):
    """Protocol for XGBClassifier constructor."""
    def __call__(
        self, *, learning_rate: float, max_depth: int, n_estimators: int,
        subsample: float, colsample_bytree: float, random_state: int,
        objective: str, eval_metric: str,
    ) -> XGBModelProtocol: ...


class XGBClassifierLoader(Protocol):
    """Protocol for XGBClassifier loader (no-arg constructor)."""
    def __call__(self) -> XGBModelProtocol: ...
```

#### trainer.py

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import TrainConfig, XGBClassifierFactory, XGBModelProtocol


def train_model(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    config: TrainConfig,
) -> XGBModelProtocol:
    """Train XGBoost classifier for breach prediction."""
    xgb = __import__("xgboost")
    classifier_factory: XGBClassifierFactory = xgb.XGBClassifier

    model = classifier_factory(
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        random_state=config["random_state"],
        objective="binary:logistic",
        eval_metric="logloss",
    )

    model.fit(x_features, y_labels)
    return model


def save_model(model: XGBModelProtocol, path: str) -> None:
    """Save trained model to file path."""
    model.save_model(path)
```

#### predictor.py

```python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from covenant_domain.features import LoanFeatures
from numpy.typing import NDArray

from .types import Proba2DProtocol, XGBClassifierLoader, XGBModelProtocol


def predict_probabilities(
    model: XGBModelProtocol,
    features: Sequence[LoanFeatures],
) -> list[float]:
    """Predict breach probabilities for loan features."""
    if len(features) == 0:
        return []

    x_array = _features_to_array(features)
    proba = model.predict_proba(x_array)
    return _extract_positive_class_probabilities(proba)


def load_model(model_path: str) -> XGBModelProtocol:
    """Load XGBoost model from file path."""
    xgb = __import__("xgboost")
    classifier_loader: XGBClassifierLoader = xgb.XGBClassifier
    model = classifier_loader()
    model.load_model(model_path)
    return model
```

---

### libs/covenant_persistence/

PostgreSQL repository layer using Protocol-based psycopg.

```
covenant_persistence/
├── __init__.py
├── py.typed
├── protocols.py       # Database Protocol definitions
├── repositories.py    # Repository Protocol definitions
├── postgres.py        # psycopg implementations
└── schema.sql         # DDL
```

#### protocols.py

```python
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol


class CursorProtocol(Protocol):
    """Protocol for psycopg cursor."""

    def execute(self, query: str, params: tuple[str | int | bool | None, ...] = ()) -> None: ...
    def fetchone(self) -> tuple[str | int | bool | None, ...] | None: ...
    def fetchall(self) -> Sequence[tuple[str | int | bool | None, ...]]: ...

    @property
    def rowcount(self) -> int: ...


class ConnectionProtocol(Protocol):
    """Protocol for psycopg connection."""

    def cursor(self) -> CursorProtocol: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


class ConnectCallable(Protocol):
    """Protocol for psycopg.connect function."""

    def __call__(self, conninfo: str) -> ConnectionProtocol: ...


@contextmanager
def connect(dsn: str) -> Iterator[ConnectionProtocol]:
    """Context manager for database connection with typed interface."""
    psycopg = __import__("psycopg")
    connect_fn: ConnectCallable = psycopg.connect
    conn = connect_fn(dsn)
    try:
        yield conn
    finally:
        conn.close()
```

#### repositories.py

```python
from __future__ import annotations

from typing import Protocol, Sequence

from covenant_domain.models import (
    Deal, DealId, Covenant, CovenantId, Measurement, CovenantResult
)


class DealRepository(Protocol):
    """Repository for Deal operations."""

    def create(self, deal: Deal) -> None:
        """Insert new deal. Raises on duplicate ID."""
        ...

    def get(self, deal_id: DealId) -> Deal:
        """Get deal by ID. Raises KeyError if not found."""
        ...

    def list_all(self) -> Sequence[Deal]:
        """List all deals."""
        ...

    def update(self, deal: Deal) -> None:
        """Update existing deal. Raises KeyError if not found."""
        ...

    def delete(self, deal_id: DealId) -> None:
        """Delete deal. Raises KeyError if not found."""
        ...


class CovenantRepository(Protocol):
    """Repository for Covenant operations."""

    def create(self, covenant: Covenant) -> None:
        """Insert new covenant. Raises on duplicate ID."""
        ...

    def list_for_deal(self, deal_id: DealId) -> Sequence[Covenant]:
        """List all covenants for a deal."""
        ...

    def delete(self, covenant_id: CovenantId) -> None:
        """Delete covenant. Raises KeyError if not found."""
        ...


class MeasurementRepository(Protocol):
    """Repository for Measurement operations."""

    def add_many(self, measurements: Sequence[Measurement]) -> int:
        """Insert measurements. Returns count inserted. Raises on duplicate."""
        ...

    def list_for_deal_and_period(
        self,
        deal_id: DealId,
        period_start_iso: str,
        period_end_iso: str,
    ) -> Sequence[Measurement]:
        """List measurements for deal and period."""
        ...

    def list_for_deal(self, deal_id: DealId) -> Sequence[Measurement]:
        """List all measurements for a deal."""
        ...


class CovenantResultRepository(Protocol):
    """Repository for CovenantResult operations."""

    def save(self, result: CovenantResult) -> None:
        """Insert or update result."""
        ...

    def save_many(self, results: Sequence[CovenantResult]) -> int:
        """Insert or update multiple results. Returns count."""
        ...

    def list_for_deal(self, deal_id: DealId) -> Sequence[CovenantResult]:
        """List all results for a deal's covenants."""
        ...

    def list_for_covenant(self, covenant_id: CovenantId) -> Sequence[CovenantResult]:
        """List results for a specific covenant."""
        ...
```

#### schema.sql

```sql
-- Covenant Radar Database Schema
-- All monetary values stored as integers (cents or scaled by 1_000_000)

CREATE TABLE deals (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    borrower TEXT NOT NULL,
    sector TEXT NOT NULL,
    region TEXT NOT NULL,
    commitment_amount_cents BIGINT NOT NULL,
    currency TEXT NOT NULL,
    maturity_date DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE covenants (
    id UUID PRIMARY KEY,
    deal_id UUID NOT NULL REFERENCES deals(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    formula TEXT NOT NULL,
    threshold_value_scaled BIGINT NOT NULL,
    threshold_direction TEXT NOT NULL CHECK (threshold_direction IN ('<=', '>=')),
    frequency TEXT NOT NULL CHECK (frequency IN ('QUARTERLY', 'ANNUAL')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_id UUID NOT NULL REFERENCES deals(id) ON DELETE CASCADE,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value_scaled BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (deal_id, period_start, period_end, metric_name)
);

CREATE TABLE covenant_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    covenant_id UUID NOT NULL REFERENCES covenants(id) ON DELETE CASCADE,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    calculated_value_scaled BIGINT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OK', 'NEAR_BREACH', 'BREACH')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (covenant_id, period_start, period_end)
);

CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    feature_names JSONB NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);

-- Indexes
CREATE INDEX idx_covenants_deal_id ON covenants(deal_id);
CREATE INDEX idx_measurements_deal_period ON measurements(deal_id, period_start, period_end);
CREATE INDEX idx_measurements_deal_id ON measurements(deal_id);
CREATE INDEX idx_results_covenant_id ON covenant_results(covenant_id);
CREATE INDEX idx_ml_models_active ON ml_models(is_active) WHERE is_active = TRUE;
```

---

## Service: covenant-radar-api

### Directory Structure

```
services/covenant-radar-api/
├── src/covenant_radar_api/
│   ├── __init__.py
│   ├── py.typed
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   ├── deals.py
│   │   │   ├── covenants.py
│   │   │   ├── measurements.py
│   │   │   ├── evaluate.py
│   │   │   └── ml.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── requests.py
│   │   └── decode.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── container.py
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── train_job.py
│   │   ├── evaluate_job.py
│   │   └── worker_entry.py
│   └── infra/
│       ├── __init__.py
│       └── postgres_repos.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_domain/
│   │   ├── __init__.py
│   │   ├── test_formula_parser.py
│   │   ├── test_rules.py
│   │   ├── test_features.py
│   │   └── test_decode.py
│   ├── test_ml/
│   │   ├── __init__.py
│   │   ├── test_train.py
│   │   └── test_predict.py
│   ├── test_persistence/
│   │   ├── __init__.py
│   │   └── test_repositories.py
│   └── test_api/
│       ├── __init__.py
│       └── test_endpoints.py
├── scripts/
│   └── seed.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

### pyproject.toml

```toml
[build-system]
requires = ["poetry-core>=1.3.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "covenant-radar-api"
version = "0.1.0"
description = "Loan covenant monitoring and breach prediction API"
authors = ["Austin Wagner <austinwagner@msn.com>"]
packages = [
  { include = "covenant_radar_api", from = "src" }
]
include = ["src/covenant_radar_api/py.typed"]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.124"
hypercorn = "^0.18"
redis = "^7.1"
rq = "^2.3"
psycopg = {version = "^3.3", extras = ["binary", "pool"]}
httpx = "^0.28"
xgboost = "^3.1"
numpy = "^2.3"
scikit-learn = "^1.7"

# Monorepo libs
platform-core = { path = "../../libs/platform_core", develop = true }
platform-workers = { path = "../../libs/platform_workers", develop = true }
platform-ml = { path = "../../libs/platform_ml", develop = true }
covenant-domain = { path = "../../libs/covenant_domain", develop = true }
covenant-ml = { path = "../../libs/covenant_ml", develop = true }
covenant-persistence = { path = "../../libs/covenant_persistence", develop = true }

[tool.poetry.group.dev.dependencies]
pytest = "^9.0.0"
pytest-asyncio = "^1.3.0"
pytest-timeout = "^2.3.1"
mypy = "^1.13.0"
ruff = "^0.14.4"
fakeredis = {version = "^2.26", extras = ["lua"]}
pytest-cov = "^7.0.0"
pytest-xdist = "^3.6.1"
typing-extensions = "^4.12.0"

[tool.poetry.scripts]
covenant-rq-worker = "covenant_radar_api.worker.worker_entry:main"

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true
disallow_subclassing_any = true
disallow_any_generics = true
no_implicit_optional = true
check_untyped_defs = true
no_implicit_reexport = true
show_error_codes = true
pretty = true
files = ["src", "tests", "scripts"]
mypy_path = ["src"]
explicit_package_bases = true
disallow_any_unimported = true
disallow_any_expr = true
disallow_any_decorated = true
disallow_any_explicit = true

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests", "scripts"]
exclude = [".venv"]

[tool.ruff.lint]
select = [
    "E","F","I","B","BLE","UP","N","C4","SIM","RET","C90","RUF","ANN"
]
ignore = []
fixable = ["ALL"]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any" = { msg = "Do not use typing.Any; prefer precise types or Protocols/TypedDicts." }
"typing.cast" = { msg = "Do not use typing.cast; prefer adapters or precise types." }
"typing.TypeAlias" = { msg = "Do not use TypeAlias; use Literal types or expand unions explicitly." }

[tool.ruff.lint.flake8-annotations]
allow-star-arg-any = false
mypy-init-return = true

[tool.coverage.run]
source = ["src", "scripts"]
omit = []
branch = true

[tool.coverage.report]
precision = 2
show_missing = true
fail_under = 100

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-v -n auto --dist loadscope"
timeout = 60
timeout_method = "thread"
```

---

## Implementation Milestones

### Milestone 1: covenant_domain Library ✅ COMPLETE

**Files:**
- [x] `libs/covenant_domain/pyproject.toml`
- [x] `libs/covenant_domain/src/covenant_domain/__init__.py`
- [x] `libs/covenant_domain/src/covenant_domain/py.typed`
- [x] `libs/covenant_domain/src/covenant_domain/models.py` (includes type aliases as Literals)
- [x] `libs/covenant_domain/src/covenant_domain/decode.py` (uses platform_core.json_utils)
- [x] `libs/covenant_domain/src/covenant_domain/encode.py`
- [x] `libs/covenant_domain/src/covenant_domain/formula_parser.py`
- [x] `libs/covenant_domain/src/covenant_domain/rules.py`
- [x] `libs/covenant_domain/src/covenant_domain/features.py`

**Tests (100% coverage):**
- [x] `libs/covenant_domain/tests/test_decode.py`
- [x] `libs/covenant_domain/tests/test_encode.py`
- [x] `libs/covenant_domain/tests/test_formula_parser.py`
- [x] `libs/covenant_domain/tests/test_rules.py`
- [x] `libs/covenant_domain/tests/test_features.py`
- [x] `libs/covenant_domain/tests/test_script_guard_entrypoint.py`

**Verification:** `cd libs/covenant_domain && make check` ✅ 113 tests, 100% coverage

---

### Milestone 2: covenant_ml Library ✅ COMPLETE

**Files:**
- [x] `libs/covenant_ml/pyproject.toml`
- [x] `libs/covenant_ml/src/covenant_ml/__init__.py`
- [x] `libs/covenant_ml/src/covenant_ml/py.typed`
- [x] `libs/covenant_ml/src/covenant_ml/types.py` - TrainConfig TypedDict, XGBModelProtocol, XGBClassifierFactory, XGBClassifierLoader, Proba2DProtocol
- [x] `libs/covenant_ml/src/covenant_ml/trainer.py` - train_model(), save_model()
- [x] `libs/covenant_ml/src/covenant_ml/predictor.py` - load_model(), predict_probabilities()

**Dependencies (latest stable):**
- xgboost ^3.1 (3.1.2)
- scikit-learn ^1.7 (1.7.2)
- numpy ^2.3 (2.3.5)

**Tests (100% coverage):**
- [x] `libs/covenant_ml/tests/test_trainer.py`
- [x] `libs/covenant_ml/tests/test_predictor.py`
- [x] `libs/covenant_ml/tests/test_types.py`
- [x] `libs/covenant_ml/tests/test_script_guard_entrypoint.py`

**Verification:** `cd libs/covenant_ml && make check` ✅ 18 tests, 100% coverage

---

### Milestone 3: covenant_persistence Library ✅ COMPLETE

**Files:**
- [x] `libs/covenant_persistence/pyproject.toml`
- [x] `libs/covenant_persistence/src/covenant_persistence/__init__.py`
- [x] `libs/covenant_persistence/src/covenant_persistence/py.typed`
- [x] `libs/covenant_persistence/src/covenant_persistence/protocols.py` - CursorProtocol, ConnectionProtocol, ConnectCallable, connect()
- [x] `libs/covenant_persistence/src/covenant_persistence/repositories.py` - DealRepository, CovenantRepository, MeasurementRepository, CovenantResultRepository protocols
- [x] `libs/covenant_persistence/src/covenant_persistence/postgres.py` - PostgresDealRepository, PostgresCovenantRepository, PostgresMeasurementRepository, PostgresCovenantResultRepository
- [x] `libs/covenant_persistence/src/covenant_persistence/schema.sql`

**Tests (100% coverage, uses in-memory connection implementation):**
- [x] `libs/covenant_persistence/tests/test_protocols.py` - Protocol structural typing and connect context manager
- [x] `libs/covenant_persistence/tests/test_repositories.py` - Repository protocol verification
- [x] `libs/covenant_persistence/tests/test_postgres_repos.py` - In-memory PostgreSQL repository tests (no mocks)
- [x] `libs/covenant_persistence/tests/test_row_converters.py` - Row to TypedDict conversion tests
- [x] `libs/covenant_persistence/tests/test_init.py` - Package exports
- [x] `libs/covenant_persistence/tests/test_script_guard_entrypoint.py` - Guard script tests

**Verification:** `cd libs/covenant_persistence && make check` ✅ 96 tests, 100% coverage

---

### Milestone 4: covenant-radar-api Service Shell ✅ COMPLETE

**Service Files:**
- [x] `services/covenant-radar-api/pyproject.toml` - Dependencies, strict mypy/ruff config, banned APIs
- [x] `services/covenant-radar-api/Makefile` - PowerShell lint/test/check targets
- [x] `services/covenant-radar-api/Dockerfile` - Multi-stage build (api + worker targets)
- [x] `services/covenant-radar-api/docker-compose.yml` - Port 8007, platform-network
- [x] `services/covenant-radar-api/README.md`
- [x] `services/covenant-radar-api/src/covenant_radar_api/__init__.py`
- [x] `services/covenant-radar-api/src/covenant_radar_api/py.typed`
- [x] `services/covenant-radar-api/src/covenant_radar_api/main.py` - FastAPI factory with platform_core integration
- [x] `services/covenant-radar-api/src/covenant_radar_api/health.py` - Health check utilities
- [x] `services/covenant-radar-api/src/covenant_radar_api/core/__init__.py`
- [x] `services/covenant-radar-api/src/covenant_radar_api/core/config.py` - Re-exports CovenantRadarSettings
- [x] `services/covenant-radar-api/src/covenant_radar_api/core/container.py` - ServiceContainer for dependency injection
- [x] `services/covenant-radar-api/src/covenant_radar_api/api/__init__.py`
- [x] `services/covenant-radar-api/src/covenant_radar_api/api/routes/__init__.py`
- [x] `services/covenant-radar-api/src/covenant_radar_api/api/routes/health.py` - /healthz, /readyz endpoints
- [x] `services/covenant-radar-api/src/covenant_radar_api/infra/__init__.py`
- [x] `services/covenant-radar-api/src/covenant_radar_api/worker/__init__.py`
- [x] `services/covenant-radar-api/src/covenant_radar_api/worker/worker_entry.py` - RQ worker using centralized platform_workers.rq_harness
- [x] `services/covenant-radar-api/scripts/__init__.py`

**Platform Core Additions:**
- [x] `libs/platform_core/src/platform_core/config/covenant_radar.py` - CovenantRadarSettings TypedDict
- [x] `libs/platform_core/src/platform_core/queues.py` - Added COVENANT_QUEUE constant
- [x] `libs/platform_core/src/platform_core/job_events.py` - Added "covenant" to JobDomain
- [x] `libs/platform_core/tests/test_covenant_radar_settings.py` - Full test coverage

**Tests (100% coverage):**
- [x] `services/covenant-radar-api/tests/__init__.py`
- [x] `services/covenant-radar-api/tests/conftest.py` - FakeRedis fixture
- [x] `services/covenant-radar-api/tests/test_health.py` - Endpoint tests
- [x] `services/covenant-radar-api/tests/test_health_utils.py` - Unit tests
- [x] `services/covenant-radar-api/tests/test_main.py` - Factory tests
- [x] `services/covenant-radar-api/tests/test_config.py` - Settings tests
- [x] `services/covenant-radar-api/tests/test_imports.py` - Package import tests
- [x] `services/covenant-radar-api/tests/test_container.py` - ServiceContainer tests with FakeRedis
- [x] `services/covenant-radar-api/tests/test_worker_entry.py` - RQ worker entry point tests

**Verification:** `cd services/covenant-radar-api && make check` ✅ 100% coverage (tests included in Milestone 5 count)

---

### Milestone 5: CRUD API Endpoints ✅ COMPLETE

**Files:**
- [x] `src/covenant_radar_api/api/routes/deals.py`
- [x] `src/covenant_radar_api/api/routes/covenants.py`
- [x] `src/covenant_radar_api/api/routes/measurements.py`
- [x] `src/covenant_radar_api/api/decode.py`

**Tests (100% coverage):**
- [x] `tests/test_routes_deals.py`
- [x] `tests/test_routes_covenants.py`
- [x] `tests/test_routes_measurements.py`
- [x] `tests/test_decode.py`

**Verification:** `cd services/covenant-radar-api && make check` ✅ 79 tests, 100% coverage

---

### Milestone 6: Evaluation and ML Endpoints ✅ COMPLETE

**Files:**
- [x] `src/covenant_radar_api/api/routes/evaluate.py`
- [x] `src/covenant_radar_api/api/routes/ml.py`
- [x] `src/covenant_radar_api/worker/train_job.py`
- [x] `src/covenant_radar_api/worker/evaluate_job.py`
- [x] `src/covenant_radar_api/worker_entry.py` (relocated to package root)
- [x] `src/covenant_radar_api/_test_hooks.py` (worker runner injection)
- [x] `src/covenant_radar_api/core/_test_hooks.py` (container/config hooks)
- [x] `scripts/guard.py` (MockBanRule integration)

**Tests (100% coverage):**
- [x] `tests/test_routes_evaluate.py` - Evaluation endpoint tests
- [x] `tests/test_routes_ml.py` - Prediction and training endpoint tests
- [x] `tests/test_train_job.py` - Training job integration tests
- [x] `tests/test_evaluate_job.py` - Batch evaluation job tests
- [x] `tests/test_worker_entry.py` - Worker entry point tests
- [x] `tests/test_guard_checks.py` - Guard script tests

**Verification:** `cd services/covenant-radar-api && make check` ✅ 146 tests, 100% coverage, 0 guard violations

---

### Milestone 7: Documentation and Demo ✅ COMPLETE

**Files:**
- [x] `services/covenant-radar-api/README.md` - Basic documentation
- [x] Update `docs/services.md` - Added covenant-radar-api entry
- [x] Update `docs/architecture.md` - Added to all sections

**Optional (not required for completion):**
- [ ] `services/covenant-radar-api/scripts/seed.py` - Demo data seeder (future enhancement)

---

## Port Assignment

| Service | Port |
|---------|------|
| covenant-radar-api | 8007 |

---

## Testing Strategy

All tests follow the monorepo pattern:
- 100% statement coverage
- 100% branch coverage
- `fail_under = 100` enforced
- Parallel execution with pytest-xdist

**Unit Tests:** Pure functions in domain/ml libs
**Integration Tests:** Repository tests with test database
**API Tests:** FastAPI TestClient through HTTP boundary

No mocking of core logic - test through actual code paths.
