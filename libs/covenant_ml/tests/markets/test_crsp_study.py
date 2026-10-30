"""The monthly validation study on real files, trainers and holdings accounting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from platform_core.json_utils import JSONObject, require_float, require_str

from covenant_ml.benchmarking.factory import make_baseline_trainers, make_benchmark_config
from covenant_ml.evaluation.paired import compare_moving_block
from covenant_ml.evaluation.types import ComparisonMetric, IntervalMethod
from covenant_ml.markets.crsp_calendar import crsp_month
from covenant_ml.markets.crsp_extract import CrspSources, load_crsp_sources
from covenant_ml.markets.crsp_holdings import (
    crsp_window_path,
    evaluate_crsp_holdings,
    index_crsp_outcomes,
)
from covenant_ml.markets.crsp_panel import CrspFeature, build_crsp_panel
from covenant_ml.markets.crsp_study import (
    EQUAL_WEIGHT,
    MOMENTUM,
    CrspValidation,
    crsp_selected_count,
    crsp_study_rows,
    crsp_top,
    run_crsp_validation,
)

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import DELISTED, study_protocol, study_rows, study_sources

TRAINERS = make_baseline_trainers(make_benchmark_config(3, 2, 16, 4, 1, 0))
ARMS = tuple(trainer.model_name.value for trainer in TRAINERS)
FEATURES = ("momentum", "liquidity")


def validate(sources: CrspSources, stopping: int, features: tuple[str, ...]) -> CrspValidation:
    """Run the study with annual folds, seed 11 and 90% intervals."""
    return run_crsp_validation(
        sources,
        TRAINERS,
        11,
        features=features,
        stopping_months=stopping,
        fold_months=12,
        confidence=0.9,
    )


def test_folds_drop_exactly_the_label_still_open_at_each_boundary(tmp_path: Path) -> None:
    sources = study_sources(tmp_path)
    result = validate(sources, 6, FEATURES)
    panel = build_crsp_panel(sources, "2017-01-01", "2019-10-31")
    sizes = [len(decision.features) for decision in panel]
    assert len(panel) == 34
    # Validation starts at index 18 (July 2018); annual folds start at 18 and 30.
    for fold, start in zip(result.folds, (18, 30), strict=True):
        stop = start - 6
        assert fold.first_stopping == panel[stop].window.decision
        assert fold.first_scored == panel[start].window.decision
        # The decision just before each boundary exits after it; the one before that does not.
        assert panel[stop - 2].window.exit < fold.first_stopping < panel[stop - 1].window.exit
        assert panel[start - 2].window.exit < fold.first_scored < panel[start - 1].window.exit
        assert fold.n_train == sum(sizes[: stop - 1])
        assert fold.n_stopping == sum(sizes[stop : start - 1])
        assert fold.n_purged == sizes[stop - 1] + sizes[start - 1]
        assert fold.n_scored == sum(sizes[start : start + 12])
    assert [decision.window.decision for decision in result.decisions] == [
        decision.window.decision for decision in panel[18:]
    ]
    # The November and December decisions need final-period outcomes and are omitted.
    assert result.decisions[-1].window.decision == "2019-10-31"
    assert result.decisions[-1].window.exit == "2019-12-02"


def _oracle_top(features: tuple[CrspFeature, ...], scores: list[float]) -> tuple[int, ...]:
    """Independent half-universe selection, ties by PERMNO."""
    pairs = sorted(
        (-score, feature.permno) for score, feature in zip(scores, features, strict=True)
    )
    return tuple(sorted(permno for _, permno in pairs[: (len(features) + 1) // 2]))


def test_baselines_costs_and_pairing_match_independent_oracles(tmp_path: Path) -> None:
    sources = study_sources(tmp_path)
    result = validate(sources, 6, FEATURES)
    series = {(item.arm, item.cost_bps): item.periods for item in result.series}
    assert set(series) == {
        (arm, cost) for arm in (MOMENTUM, EQUAL_WEIGHT, *ARMS) for cost in (10, 5, 25)
    }
    momentum = series[(MOMENTUM, 10)]
    expected = tuple(
        _oracle_top(decision.features, [item.momentum for item in decision.features])
        for decision in result.decisions
    )
    assert tuple(period.selected for period in momentum) == expected
    windows = tuple(decision.window for decision in result.decisions)
    assert momentum == evaluate_crsp_holdings(index_crsp_outcomes(sources), windows, expected, 10)
    universe = series[(EQUAL_WEIGHT, 10)]
    for period, decision in zip(universe, result.decisions, strict=True):
        assert period.selected == tuple(sorted(item.permno for item in decision.features))
    # Higher declared cost never raises a month's net return.
    for cheap, dear in zip(series[(MOMENTUM, 5)], series[(MOMENTUM, 25)], strict=True):
        assert dear.net_return <= cheap.net_return
    assert sum(item.net_return for item in series[(MOMENTUM, 25)]) < sum(
        item.net_return for item in series[(MOMENTUM, 5)]
    )
    # Every non-momentum arm is paired with momentum on the identical 16 months.
    assert [comparison["arm"] for comparison in result.comparisons] == [EQUAL_WEIGHT, *ARMS]
    reference = [period.net_return for period in momentum]
    for comparison in result.comparisons:
        arm = [period.net_return for period in series[(comparison["arm"], 10)]]
        assert comparison["reference"] == MOMENTUM
        assert comparison["metric"] is ComparisonMetric.UTILITY
        assert comparison["n_rows"] == 16
        assert comparison["value_arm"] == pytest.approx(sum(arm) / 16)
        assert comparison["value_reference"] == pytest.approx(sum(reference) / 16)
        difference = comparison["difference"]
        assert difference["method"] is IntervalMethod.MOVING_BLOCK_BOOTSTRAP
        assert difference["estimate"] == pytest.approx(
            sum(a - b for a, b in zip(arm, reference, strict=True)) / 16
        )
        assert difference["n_units"] == 5
    months = [crsp_month(period.window.decision) for period in momentum]
    assert months == list(range(months[0], months[0] + 16))
    # The interval is the frozen block length, draw count and seed applied to these months.
    frozen = sources.protocol
    universe_returns: list[float] = [period.net_return for period in universe]
    assert result.comparisons[0] == compare_moving_block(
        EQUAL_WEIGHT,
        MOMENTUM,
        np.asarray(universe_returns, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        np.asarray(months, dtype=np.int64),
        np.asarray(months, dtype=np.int64),
        block_length=frozen["bootstrap_block_months"],
        n_resamples=frozen["bootstrap_draws"],
        seed=frozen["bootstrap_seed"],
        confidence=0.9,
    )


def test_delisted_holding_is_valued_once_then_reinvested(tmp_path: Path) -> None:
    result = validate(study_sources(tmp_path), 6, FEATURES)
    universe = {item.arm: item.periods for item in result.series if item.cost_bps == 10}
    months = {period.window.decision: period for period in universe[EQUAL_WEIGHT]}
    delisting = months["2019-02-28"]
    assert DELISTED in delisting.selected
    (owed,) = delisting.receivables
    assert (owed.permno, owed.valued) == (DELISTED, "2019-03-20")
    assert DELISTED not in dict(delisting.held)
    assert months["2019-03-29"].receivables == ()
    assert DELISTED not in months["2019-03-29"].selected


@pytest.mark.parametrize(
    ("boundary", "last_decision", "last_exit"),
    [
        ("2020-01-01", "2019-10-31", "2019-12-02"),
        ("2019-12-02", "2019-09-30", "2019-11-01"),
        ("2019-12-03", "2019-10-31", "2019-12-02"),
    ],
)
def test_holdout_boundary_excludes_equal_and_later_exits(
    tmp_path: Path, boundary: str, last_decision: str, last_exit: str
) -> None:
    protocol = study_protocol()
    protocol["holdout_start"] = boundary
    path, _ = write_sources(tmp_path, study_rows(), protocol)
    result = validate(load_crsp_sources(path), 6, FEATURES)
    assert result.decisions[-1].window.decision == last_decision
    assert result.decisions[-1].window.exit == last_exit
    assert all(decision.window.exit < boundary for decision in result.decisions)


def test_changed_holdout_outcomes_cannot_change_validation(tmp_path: Path) -> None:
    (tmp_path / "original").mkdir()
    (tmp_path / "changed").mkdir()
    original = validate(study_sources(tmp_path / "original"), 6, FEATURES)
    rows = study_rows()
    changed = 0
    for row in rows["returns"]:
        if require_str(row, "dlycaldt") >= "2020-01-01":
            row["dlyret"] = 0.9
            row["dlyprc"] = 200.0
            changed += 1
    assert changed == 301  # Seven surviving securities over 43 weekday sessions.
    path, _ = write_sources(tmp_path / "changed", rows, study_protocol())
    result = validate(load_crsp_sources(path), 6, FEATURES)
    assert result == original


def test_rows_label_against_the_universe_and_date_availability(tmp_path: Path) -> None:
    sources = study_sources(tmp_path)
    panel = build_crsp_panel(sources, "2019-02-01", "2019-02-28")
    outcomes = index_crsp_outcomes(sources)
    rows = crsp_study_rows(outcomes, panel, ("liquidity",))
    assert rows.features.shape == (6, 1)
    # Independent label oracle compounds the raw fixture rows over (entry, exit], stops the
    # delisted security at its delisting date and applies its -25% terminal value once.
    window = panel[0].window
    growth: list[float] = []
    for feature in panel[0].features:
        end = "2019-03-14" if feature.permno == DELISTED else window.exit
        value = 0.75 if feature.permno == DELISTED else 1.0
        for row in study_rows()["returns"]:
            day = require_str(row, "dlycaldt")
            if row["permno"] == feature.permno and window.entry < day <= end:
                value *= 1 + require_float(row, "dlyret")
        growth.append(value)
    assert crsp_window_path(outcomes, DELISTED, window).terminal == ("2019-03-20", 0.75)
    mean = sum(growth) / len(growth)
    labels = [int(rows.labels.flat[index]) for index in range(6)]
    assert labels == [1 if value > mean else 0 for value in growth]
    assert 0 < sum(labels) < 6
    assert {int(rows.available.flat[index]) for index in range(6)} == {737150}  # 2019-04-01
    assert {int(rows.times.flat[index]) for index in range(6)} == {737118}  # 2019-02-28


@pytest.mark.parametrize(
    ("fraction", "eligible", "count"), [(0.1, 30, 3), (0.1, 1000, 100), (0.5, 5, 3), (0.1, 1, 1)]
)
def test_selection_count_rounds_the_declared_decimal_up(
    fraction: float, eligible: int, count: int
) -> None:
    assert crsp_selected_count(fraction, eligible) == count


def test_top_selection_breaks_ties_by_permno() -> None:
    features = tuple(CrspFeature(permno, 0.0, 1.0) for permno in (9, 3, 7, 5))
    assert crsp_top(features, [0.5, 0.5, 0.9, 0.1], 2) == (3, 7)


def _refusal(tmp_path: Path, case: str) -> tuple[CrspSources, int, tuple[str, ...]]:
    """Build one invalid study request."""
    rows: dict[str, list[JSONObject]] = study_rows()
    protocol: JSONObject = study_protocol()
    stopping: int = 6
    features: tuple[str, ...] = ("momentum",)
    if case == "stopping":
        stopping = 18
    elif case == "nonpositive":
        stopping = 0
    elif case == "feature":
        features = ("volume",)
    elif case == "no_validation":
        protocol.update({"validation_start": "2019-12-30", "holdout_start": "2019-12-31"})
    elif case == "empty":
        for row in rows["membership"]:
            row["shrcd"] = 12
    else:
        rows["returns"] = [
            row for row in rows["returns"] if (row["permno"], row["dlycaldt"]) != (3, "2019-05-15")
        ]
    path, _ = write_sources(tmp_path, rows, protocol)
    return load_crsp_sources(path), stopping, features


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("stopping", "training decisions before the stopping months"),
        ("nonpositive", "must be positive"),
        ("feature", "nonempty subset"),
        ("no_validation", "no validation decision"),
        ("empty", "no eligible securities"),
        ("missing", "held return for 3 on 2019-05-15"),
    ],
)
def test_unsupported_schedules_and_incomplete_evidence_are_refused(
    tmp_path: Path, case: str, message: str
) -> None:
    sources, stopping, features = _refusal(tmp_path, case)
    with pytest.raises(ValueError, match=message):
        validate(sources, stopping, features)


def test_requested_feature_order_is_the_column_order(tmp_path: Path) -> None:
    sources = study_sources(tmp_path)
    panel = build_crsp_panel(sources, "2019-02-01", "2019-02-28")
    rows = crsp_study_rows(index_crsp_outcomes(sources), panel, ("liquidity", "momentum"))
    leader = panel[0].features[0]
    assert float(rows.features.flat[0]) == leader.liquidity
    assert float(rows.features.flat[1]) == leader.momentum
