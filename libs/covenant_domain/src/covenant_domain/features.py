from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from .models import CovenantResult, Deal


class LoanFeatures(TypedDict, total=True):
    """Feature vector for ML risk prediction. All values are floats."""

    debt_to_ebitda: float
    interest_cover: float
    current_ratio: float
    leverage_change_1p: float  # vs 1 period ago
    leverage_change_4p: float  # vs 4 periods ago
    sector_encoded: int
    region_encoded: int
    near_breach_count_4p: int  # near breaches in last 4 periods


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
    # Assume results are sorted by period descending
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


__all__ = [
    "FEATURE_ORDER",
    "LoanFeatures",
    "RiskPrediction",
    "classify_risk_tier",
    "extract_features",
]
