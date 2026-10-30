"""The selected candidate's single final-period score under the frozen monthly design.

Only ``research_final.execute_final`` calls this, after the research journal has
consumed the selected candidate's one final release. The candidate is fitted
once on every decision whose outcome is known before ``holdout_start``: the
declared stopping months immediately before the boundary drive early stopping,
and ``purge_label_overlap`` drops each label not available strictly before the
next boundary. Every decision from ``holdout_start`` through ``holdout_end`` is
then ranked from decision-time features alone, and every arm runs through the
validation study's holdings accounting and paired moving-block comparison.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..benchmarking.protocols import TrainerProto
from .crsp_extract import CrspSources
from .crsp_holdings import index_crsp_outcomes
from .crsp_panel import build_crsp_panel
from .crsp_study import CrspScored, crsp_arms, crsp_fold, crsp_known_panel, crsp_study_rows


def run_crsp_final(
    sources: CrspSources,
    trainers: Sequence[TrainerProto],
    seed: int,
    *,
    features: tuple[str, ...],
    stopping_months: int,
    confidence: float,
) -> CrspScored:
    """Fit once before the final period and score every final-period decision.

    Args:
        sources: Verified CRSP evidence covering the final period and its exits.
        trainers: The selected candidate's distinctly named trainers.
        seed: The selected attempt's model seed.
        features: The selected attempt's decision-time features.
        stopping_months: Decision months before the boundary used for early stopping.
        confidence: Coverage of each moving-block interval.

    Returns:
        The final-period decisions, the single fit's purged counts, every arm's
        accounting at every frozen cost over ``holdout_start`` through
        ``holdout_end`` and each non-momentum arm paired with momentum.

    Raises:
        ValueError: CRSP-009 for a nonpositive stopping schedule or incomplete
            evidence; calendar, fold, holdings and moving-block refusals propagate.
    """
    if stopping_months < 1:
        raise ValueError("[CRSP-009] stopping months must be positive")
    protocol = sources.protocol
    known = crsp_known_panel(sources)
    final = build_crsp_panel(sources, protocol["holdout_start"], protocol["holdout_end"])
    panel = known + final
    outcomes = index_crsp_outcomes(sources)
    rows = crsp_study_rows(outcomes, panel, features)
    fit, scores = crsp_fold(
        rows, panel, range(len(known), len(panel)), stopping_months, trainers, seed
    )
    series, comparisons = crsp_arms(outcomes, final, scores, protocol, confidence)
    return CrspScored(final, (fit,), series, comparisons)


__all__ = ["run_crsp_final"]
