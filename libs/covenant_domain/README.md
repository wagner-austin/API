# covenant-domain

Pure business logic for covenant monitoring: TypedDict models, rule engine, formula parser, and feature extraction.

## Installation

```bash
poetry add covenant-domain
```

No external dependencies - this is a pure domain library.

## Domain Models

All models are TypedDicts with no mutable state:

```python
from covenant_domain import Deal, DealId, Covenant, CovenantId, Measurement, CovenantResult

# Deal - a loan facility
deal: Deal = {
    "id": DealId(value="abc-123"),
    "name": "Tech Corp Credit Facility",
    "borrower": "Tech Corp",
    "sector": "Technology",
    "region": "North America",
    "commitment_amount_cents": 50_000_000_00,  # $500M in cents
    "currency": "USD",
    "maturity_date_iso": "2027-12-31",
}

# Covenant - a rule attached to a deal
covenant: Covenant = {
    "id": CovenantId(value="cov-456"),
    "deal_id": DealId(value="abc-123"),
    "name": "Max Leverage Ratio",
    "formula": "total_debt / ebitda",
    "threshold_value_scaled": 4_500_000,  # 4.5x scaled by 1M
    "threshold_direction": "<=",
    "frequency": "QUARTERLY",
}

# Measurement - a financial metric for a period
measurement: Measurement = {
    "deal_id": DealId(value="abc-123"),
    "period_start_iso": "2024-01-01",
    "period_end_iso": "2024-03-31",
    "metric_name": "total_debt",
    "metric_value_scaled": 1_000_000_000_000,  # $1B scaled
}
```

## Covenant Evaluation

Evaluate covenants against measurements:

```python
from covenant_domain import (
    evaluate_covenant_for_period,
    evaluate_all_covenants_for_period,
    classify_status,
)

# Evaluate a single covenant
result: CovenantResult = evaluate_covenant_for_period(
    covenant=covenant,
    period_start_iso="2024-01-01",
    period_end_iso="2024-03-31",
    measurements=[total_debt_measurement, ebitda_measurement],
    tolerance_ratio_scaled=100_000,  # 10% tolerance
)

print(result["status"])  # "OK", "NEAR_BREACH", or "BREACH"
print(result["calculated_value_scaled"])

# Evaluate all covenants for a deal
results = evaluate_all_covenants_for_period(
    covenants=[covenant1, covenant2],
    period_start_iso="2024-01-01",
    period_end_iso="2024-03-31",
    measurements=all_measurements,
    tolerance_ratio_scaled=100_000,
)
```

### Status Classification

| Status | Description |
|--------|-------------|
| `OK` | Covenant in compliance |
| `NEAR_BREACH` | Within tolerance threshold |
| `BREACH` | Threshold exceeded |

## Formula Parser

Safe arithmetic expression evaluator using shunting-yard algorithm:

```python
from covenant_domain import evaluate_formula, FormulaParseError, FormulaEvalError

# Supported operators: +, -, *, /
# Supported: parentheses, variable names

metrics = {
    "total_debt": 1_000_000_000,
    "ebitda": 250_000_000,
}

result = evaluate_formula("total_debt / ebitda", metrics)
# Returns 4.0

# Handles errors gracefully
try:
    evaluate_formula("invalid ++ formula", {})
except FormulaParseError as e:
    print(f"Parse error: {e}")

try:
    evaluate_formula("a / b", {"a": 1, "b": 0})
except FormulaEvalError as e:
    print(f"Division by zero: {e}")
```

## ML Feature Extraction

Extract features for breach prediction:

```python
from covenant_domain import extract_features, classify_risk_tier, LoanFeatures

features: LoanFeatures = extract_features(
    deal=deal,
    metrics_current={"total_debt": 1000, "ebitda": 250, ...},
    metrics_1p_ago={"total_debt": 950, ...},
    metrics_4p_ago={"total_debt": 800, ...},
    recent_results=[result1, result2, ...],
    sector_encoder={"Technology": 0, "Finance": 1, ...},
    region_encoder={"North America": 0, "Europe": 1, ...},
)

# Features include:
# - debt_to_ebitda, interest_cover, current_ratio
# - leverage_change_1p, leverage_change_4p
# - sector_encoded, region_encoded
# - near_breach_count_4p

# Classify risk tier from probability
risk_tier = classify_risk_tier(0.75)  # Returns "HIGH"
```

## JSON Encoding/Decoding

Convert between JSON dicts and TypedDicts:

```python
from covenant_domain import (
    decode_deal, encode_deal,
    decode_covenant, encode_covenant,
    decode_measurement, encode_measurement,
    decode_covenant_result, encode_covenant_result,
)

# Decode from JSON dict
json_dict = {"id": {"value": "abc"}, "name": "Deal", ...}
deal = decode_deal(json_dict)

# Encode to JSON dict
json_dict = encode_deal(deal)
```

## Design Principles

- All types are `TypedDict` (no dataclasses, no classes with state)
- All monetary values stored as scaled integers (`* 1_000_000`)
- No `Any`, `cast`, `type: ignore`, or `.pyi` stubs
- No `try/except` in core logic - exceptions propagate
- 100% test coverage (statement + branch)

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- No external dependencies
- 100% test coverage enforced
