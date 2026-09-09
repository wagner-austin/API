"""The design effect, and the two ways a clustering correction lies.

The arithmetic here is four lines and the ways it goes wrong are not, so the
tests are organised around the failure modes rather than around the
functions.

WHAT IS CHECKED AGAINST WHAT. The ICC is not compared to a value this module
produced. Every pinned number below is either hand-computed from the ANOVA
definition in the test itself -- ``_icc_by_definition`` builds the mean
squares from scratch, in a different expression from the implementation --
or it is a case where the answer is forced by construction (perfectly
nested clusters give exactly 1.0, identical clusters give exactly 0.0
whatever the data). A test that asserted the implementation's own output
would pass a rewrite that changed the meaning.
"""

from __future__ import annotations

import pytest

from platform_core.clustering import (
    MINIMUM_CLUSTERS,
    average_cluster_size,
    clustered_paired_power,
    intracluster_correlation,
)
from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.power_records import (
    decode_clustered_paired_power,
    encode_clustered_paired_power,
)
from platform_core.power_types import PowerInstrument


def _icc_by_definition(groups: list[list[float]]) -> float:
    """Recompute the ICC from the ANOVA definition, independently.

    Written as a separate expression from the implementation on purpose: it
    builds the sums of squares with explicit loops and divides at the end,
    where the implementation folds the division into the comprehension. Two
    routes to one number is the only way a pinned constant proves anything
    about the arithmetic rather than about the author's arithmetic.

    Args:
        groups: Per-cluster values.

    Returns:
        The one-way random-effects ICC.
    """
    clusters = len(groups)
    total = sum(len(group) for group in groups)
    grand = sum(value for group in groups for value in group) / total
    ssb = 0.0
    ssw = 0.0
    for group in groups:
        mean = sum(group) / len(group)
        ssb += len(group) * (mean - grand) ** 2
        for value in group:
            ssw += (value - mean) ** 2
    msb = ssb / (clusters - 1)
    msw = ssw / (total - clusters)
    m0 = (total - sum(len(group) ** 2 for group in groups) / total) / (clusters - 1)
    return (msb - msw) / (msb + (m0 - 1) * msw)


class TestTheUnequalClusterSize:
    """``m0``, which is the whole reason this is not a one-liner."""

    def test_equal_clusters_give_exactly_the_arithmetic_mean(self) -> None:
        """The special case Killip states the formula for.

        Four clusters of five: ``(20 - 100/20) / 3 = 5.0``, which is also the
        plain mean. If these two ever disagree on equal sizes the unequal
        form is wrong.
        """
        assert average_cluster_size([5, 5, 5, 5]) == pytest.approx(5.0)

    def test_unequal_clusters_fall_strictly_below_the_mean(self) -> None:
        """The direction that costs the author rather than flatters them.

        Code-style's real shape in miniature: one cluster holding most of
        the units. The arithmetic mean is 25.0 and m0 is far below it, so
        using the mean would have inflated the design effect.
        """
        sizes = [91, 3, 3, 3]
        mean = sum(sizes) / len(sizes)
        m0 = average_cluster_size(sizes)

        assert mean == pytest.approx(25.0)
        assert m0 == pytest.approx((100 - (91**2 + 27) / 100) / 3)
        assert m0 < mean

    def test_a_single_cluster_is_refused(self) -> None:
        """No between-cluster term exists, so the ratio is undefined."""
        with pytest.raises(AppError) as excinfo:
            _ = average_cluster_size([10])

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_CLUSTERS_INSUFFICIENT
        assert str(MINIMUM_CLUSTERS) in excinfo.value.message

    def test_singleton_clusters_everywhere_are_refused(self) -> None:
        """No within-cluster term exists: this is the unclustered case.

        Refused rather than answered 0.0. A returned zero publishes as "no
        clustering was detected", which is a finding about the corpus; this
        is the absence of one, and the remedy is a coarser grouping.
        """
        with pytest.raises(AppError) as excinfo:
            _ = average_cluster_size([1, 1, 1, 1])

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_CLUSTERS_INSUFFICIENT
        assert "coarser" in excinfo.value.message


class TestTheCorrelationItself:
    """Values forced by construction, and one checked against the definition."""

    def test_perfectly_nested_clusters_correlate_completely(self) -> None:
        """All variance is between clusters, so the ICC is exactly 1.

        Every unit in a cluster carries the same difference and the clusters
        differ. There is no within-cluster spread at all, so MSW is 0 and
        the ratio collapses to 1 regardless of m0.
        """
        groups = {"a": [1.0, 1.0, 1.0], "b": [-1.0, -1.0, -1.0], "c": [0.0, 0.0, 0.0]}

        assert intracluster_correlation(groups) == pytest.approx(1.0)

    def test_identical_clusters_leave_nothing_for_clustering_to_explain(self) -> None:
        """Every cluster has the same mean, so MSB is 0 and the ICC is negative.

        This is the case the design-effect floor exists for, and the raw
        value is published unfloored precisely so it is visible here.
        """
        groups = {"a": [1.0, -1.0], "b": [1.0, -1.0], "c": [1.0, -1.0]}

        assert intracluster_correlation(groups) < 0.0

    def test_a_constant_series_partitions_no_variance_at_all(self) -> None:
        """Both mean squares are zero, so the ratio is 0/0.

        Returned as 0.0 rather than raising: every unit agreeing is a real
        and common state -- an adapter that changed nothing gives exactly
        this -- and there is genuinely no clustering in it. The guard is on
        the DENOMINATOR rather than on the data, so it also covers the case
        where between and within cancel exactly.
        """
        groups = {"a": [0.0, 0.0], "b": [0.0, 0.0]}

        assert intracluster_correlation(groups) == 0.0

    def test_it_matches_the_anova_definition_computed_separately(self) -> None:
        """Two independent expressions of the same statistic, on real shape.

        Unequal clusters, mixed signs, a value repeated within a cluster --
        the arrangement that separates a correct m0 from a plain mean.
        """
        groups = [[1.0, 1.0, 0.0, -1.0], [0.0, 0.0], [1.0, 1.0, 1.0], [-1.0]]
        named = {f"c{index}": group for index, group in enumerate(groups)}

        assert intracluster_correlation(named) == pytest.approx(_icc_by_definition(groups))

    def test_empty_clusters_are_dropped_rather_than_counted(self) -> None:
        """A grouping built by iterating a corpus can carry empty keys.

        Counting one as a zero-size cluster would raise k without raising N
        and quietly deflate m0. The answer must be identical to the same
        grouping with the empty key absent.
        """
        with_empty = {"a": [1.0, 1.0], "b": [0.0, -1.0], "gone": []}
        without: dict[str, list[float]] = {"a": [1.0, 1.0], "b": [0.0, -1.0]}

        assert intracluster_correlation(with_empty) == pytest.approx(
            intracluster_correlation(without)
        )

    def test_an_ungroupable_mapping_is_refused(self) -> None:
        """The refusal propagates from the m0 computation, not around it."""
        with pytest.raises(AppError) as excinfo:
            _ = intracluster_correlation({"only": [1.0, 0.0]})

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_CLUSTERS_INSUFFICIENT


class TestTheRecord:
    """What is published, and the number that must never appear on it."""

    def test_clustering_never_manufactures_information(self) -> None:
        """THE MOTIVATING DEFECT. A negative ICC must not raise effective n.

        The first draft of this arithmetic, run as a throwaway script over
        code-style's 875 items, printed an effective n of 1718.6. Unfloored,
        this fixture does the same thing: identical clusters give a negative
        ICC, DE below 1, and an effective n above the sample. The floor is
        what stops it, and the raw correlation stays visible beside it so
        the floor is not silent.
        """
        groups = {"a": [1.0, -1.0], "b": [1.0, -1.0], "c": [1.0, -1.0]}

        record = clustered_paired_power(groups, "package")

        assert record["intracluster_correlation"] < 0.0
        assert record["design_effect"] == 1.0
        assert record["effective_sample_size"] == pytest.approx(6.0)
        assert record["effective_sample_size"] <= record["total_units"]

    def test_a_real_design_effect_deflates_the_sample(self) -> None:
        """The ordinary case, with every field checked against the inputs."""
        groups = {"a": [1.0, 1.0, 1.0], "b": [-1.0, -1.0, -1.0], "c": [0.0, 0.0, 0.0]}

        record = clustered_paired_power(groups, "top-level category")

        assert record["instrument"] == PowerInstrument.CLUSTERED_PAIRED.value
        assert record["unit"] == "top-level category"
        assert record["clusters"] == 3
        assert record["total_units"] == 9
        assert record["largest_cluster"] == 3
        assert record["average_cluster_size"] == pytest.approx(3.0)
        assert record["intracluster_correlation"] == pytest.approx(1.0)
        assert record["design_effect"] == pytest.approx(3.0)
        assert record["effective_sample_size"] == pytest.approx(3.0)

    def test_the_largest_cluster_is_reported_beside_the_average(self) -> None:
        """How a reader tells whether m0 describes the corpus or hides it."""
        groups = {"big": [1.0] * 8 + [0.0], "a": [0.0, 1.0], "b": [1.0, 0.0]}

        record = clustered_paired_power(groups, "package")

        assert record["largest_cluster"] == 9
        assert record["total_units"] == 13
        assert record["largest_cluster"] > record["average_cluster_size"]

    def test_an_unnamed_clustering_unit_is_refused(self) -> None:
        """A design effect whose grouping is unstated cannot be checked."""
        groups = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

        with pytest.raises(AppError) as excinfo:
            _ = clustered_paired_power(groups, "   ")

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID
        assert "grouping" in excinfo.value.message

    def test_an_ungroupable_corpus_is_refused_before_a_record_exists(self) -> None:
        """No partial record is emitted for a grouping that cannot be estimated."""
        with pytest.raises(AppError) as excinfo:
            _ = clustered_paired_power({"a": [1.0], "b": [0.0]}, "package")

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_CLUSTERS_INSUFFICIENT


class TestTheJSONBoundary:
    """Round trip, and the two fields a reader cannot act without."""

    def test_the_record_survives_a_round_trip(self) -> None:
        """Every field, including the unfloored correlation."""
        groups = {"a": [1.0, 1.0, 0.0], "b": [-1.0, 0.0], "c": [1.0, 1.0, 1.0, 0.0]}
        record = clustered_paired_power(groups, "containing directory")

        assert decode_clustered_paired_power(encode_clustered_paired_power(record)) == record

    def test_decode_rejects_a_mismatched_instrument(self) -> None:
        """A McNemar payload read as a design effect would invent a grouping."""
        groups = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        obj = encode_clustered_paired_power(clustered_paired_power(groups, "package"))
        obj["instrument"] = PowerInstrument.MCNEMAR.value

        with pytest.raises(AppError) as excinfo:
            _ = decode_clustered_paired_power(obj)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN

    def test_decode_rejects_a_blank_unit(self) -> None:
        """Checked on the way in as well as at construction.

        The file is what a later reader acts on, and a record hand-edited or
        written by an older version can carry a blank unit that construction
        never saw. A design effect without its grouping is uninterpretable
        whichever side of the boundary it was blanked on.
        """
        groups = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        obj = encode_clustered_paired_power(clustered_paired_power(groups, "package"))
        obj["unit"] = ""

        with pytest.raises(AppError) as excinfo:
            _ = decode_clustered_paired_power(obj)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID
        assert "grouping" in excinfo.value.message
