"""The single final-period fit and score on real files, trainers and holdings accounting."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import require_str

from covenant_ml.benchmarking.factory import make_baseline_trainers, make_benchmark_config
from covenant_ml.markets.crsp_extract import CrspSources, load_crsp_sources
from covenant_ml.markets.crsp_final import run_crsp_final
from covenant_ml.markets.crsp_holdings import evaluate_crsp_holdings, index_crsp_outcomes
from covenant_ml.markets.crsp_panel import build_crsp_panel
from covenant_ml.markets.crsp_study import (
    EQUAL_WEIGHT,
    MOMENTUM,
    CrspScored,
    crsp_known_panel,
    run_crsp_validation,
)

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import (
    FINAL_END,
    final_manifest,
    study_protocol,
    study_rows,
    study_sources,
)

TRAINERS = make_baseline_trainers(make_benchmark_config(3, 2, 16, 4, 1, 0))
ARMS = tuple(trainer.model_name.value for trainer in TRAINERS)
FEATURES = ("momentum", "liquidity")


def score(sources: CrspSources, stopping: int = 6) -> CrspScored:
    """Run the final study with seed 11 and 90% intervals."""
    return run_crsp_final(
        sources, TRAINERS, 11, features=FEATURES, stopping_months=stopping, confidence=0.9
    )


def test_one_fit_uses_only_outcomes_known_before_the_final_period(tmp_path: Path) -> None:
    sources = load_crsp_sources(final_manifest(tmp_path))
    result = score(sources)
    known = build_crsp_panel(sources, "2017-01-01", "2019-12-31")
    sizes = [len(decision.features) for decision in known]
    # Training starts in January 2017; November and December 2019 exit in 2020.
    assert len(known) == 36
    assert [decision.window.exit for decision in known[-2:]] == ["2020-01-01", "2020-02-03"]
    assert crsp_known_panel(sources) == known[:34]
    final = build_crsp_panel(sources, "2020-01-01", "2020-06-30")
    assert [decision.window.decision for decision in result.decisions] == [
        decision.window.decision for decision in final
    ]
    assert len(result.decisions) == 6
    assert result.decisions[-1].window.exit == "2020-08-03"
    (fit,) = result.folds
    # Independent index arithmetic: stopping months are known decisions 28-33; only
    # decision 27, whose exit follows the stopping boundary, is purged; no stopping
    # label is open at the first final decision because every known exit precedes 2020.
    assert fit.first_stopping == known[28].window.decision
    assert fit.first_scored == "2020-01-31"
    assert known[26].window.exit < fit.first_stopping < known[27].window.exit
    assert known[33].window.exit < "2020-01-01" < fit.first_scored
    assert fit.n_train == sum(sizes[:27])
    assert fit.n_stopping == sum(sizes[28:34])
    assert fit.n_purged == sizes[27]
    assert fit.n_scored == sum(len(decision.features) for decision in final)


def _oracle_top(scores: list[float], permnos: list[int]) -> tuple[int, ...]:
    """Independent half-universe selection, ties by PERMNO."""
    pairs = sorted((-value, permno) for value, permno in zip(scores, permnos, strict=True))
    return tuple(sorted(permno for _, permno in pairs[: (len(permnos) + 1) // 2]))


def test_final_arms_match_independent_oracles(tmp_path: Path) -> None:
    sources = load_crsp_sources(final_manifest(tmp_path))
    result = score(sources)
    series = {(item.arm, item.cost_bps): item.periods for item in result.series}
    assert set(series) == {
        (arm, cost) for arm in (MOMENTUM, EQUAL_WEIGHT, *ARMS) for cost in (10, 5, 25)
    }
    expected = tuple(
        _oracle_top(
            [item.momentum for item in decision.features],
            [item.permno for item in decision.features],
        )
        for decision in result.decisions
    )
    momentum = series[(MOMENTUM, 10)]
    assert tuple(period.selected for period in momentum) == expected
    windows = tuple(decision.window for decision in result.decisions)
    assert momentum == evaluate_crsp_holdings(index_crsp_outcomes(sources), windows, expected, 10)
    for arm in ARMS:
        for period in series[(arm, 10)]:
            assert len(period.selected) == 3
    assert [comparison["arm"] for comparison in result.comparisons] == [EQUAL_WEIGHT, *ARMS]
    for comparison in result.comparisons:
        assert comparison["n_rows"] == 6
        assert comparison["difference"]["n_units"] == 2


def test_final_outcomes_never_reach_the_fit_or_any_selection(tmp_path: Path) -> None:
    (tmp_path / "original").mkdir()
    (tmp_path / "changed").mkdir()
    original = score(load_crsp_sources(final_manifest(tmp_path / "original")))
    rows = study_rows(FINAL_END)
    changed = 0
    for row in rows["returns"]:
        # After the last final decision: every feature is unchanged, only outcomes move.
        if require_str(row, "dlycaldt") > "2020-06-30":
            row["dlyret"] = 0.9
            changed += 1
    assert changed == 7 * 44  # Seven surviving securities over July and August sessions.
    path, _ = write_sources(tmp_path / "changed", rows, study_protocol())
    result = score(load_crsp_sources(path))
    assert result.folds == original.folds
    assert [item.arm for item in result.series] == [item.arm for item in original.series]
    for before, after in zip(original.series, result.series, strict=True):
        assert [period.selected for period in after.periods] == [
            period.selected for period in before.periods
        ]
    last = {item.arm: item.periods[-1].net_return for item in result.series}
    assert last != {item.arm: item.periods[-1].net_return for item in original.series}


def test_staging_the_final_period_cannot_change_validation(tmp_path: Path) -> None:
    (tmp_path / "validation").mkdir()
    (tmp_path / "final").mkdir()
    results = [
        run_crsp_validation(
            sources,
            TRAINERS,
            11,
            features=FEATURES,
            stopping_months=6,
            fold_months=12,
            confidence=0.9,
        )
        for sources in (
            load_crsp_sources(final_manifest(tmp_path / "final")),
            study_sources(tmp_path / "validation"),
        )
    ]
    assert results[0] == results[1]
    assert results[0].decisions[-1].window.exit < "2020-01-01"


@pytest.mark.parametrize(
    ("stopping", "message"),
    [
        (0, "stopping months must be positive"),
        (34, "training decisions before the stopping months"),
    ],
)
def test_unsupported_stopping_schedule_is_refused(
    tmp_path: Path, stopping: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score(load_crsp_sources(final_manifest(tmp_path)), stopping)


def test_validation_extract_cannot_be_scored_as_final(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="calendar must cover both decision bounds"):
        score(study_sources(tmp_path))
