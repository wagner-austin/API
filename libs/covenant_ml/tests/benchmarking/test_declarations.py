"""Tests for the benchmark declarations.

A declaration is data, so these tests do not exercise behaviour -- they check
the two things a wrong declaration would silently do.

FIRST, THAT NO TWO FAMILIES CLAIM ONE EXPERIMENT. The experiment name is what
:mod:`platform_core.comparability` reads to decide whether two records may be
subtracted. Before board task ``6d5536cc`` there was a single constant, and
the obvious repair for the five families that emitted no record was to route
them all through it -- which would have made a ranking run and a regression run
claim one experiment and licensed subtracting a mean NDCG from an R-squared. A
copy-pasted name here reintroduces exactly that, and reads as correct.

SECOND, THAT A DECLARED METRIC EXISTS. ``declared_observations`` refuses a
metric its manifest does not carry, but only when a benchmark RUNS -- which for
these families means minutes of fitting. Checking each declared name against
its family's own quality TypedDict moves that failure to collection time, where
a typo costs nothing.
"""

from __future__ import annotations

from covenant_ml.benchmarking.declarations import (
    GOSS,
    MULTICLASS,
    QUANTIZED,
    RANKING,
    REGRESSION,
    VS_LIGHTGBM,
)
from covenant_ml.benchmarking.goss_quality import GossArmResult, GossQuality
from covenant_ml.benchmarking.harness import BenchmarkDeclaration
from covenant_ml.benchmarking.multiclass_quality import (
    MulticlassArmResult,
    MulticlassQuality,
)
from covenant_ml.benchmarking.quantized_quality import (
    QuantizedArmResult,
    QuantizedQuality,
)
from covenant_ml.benchmarking.ranking_quality import RankingArmResult, RankingQuality
from covenant_ml.benchmarking.regression_quality import (
    RegressionArmResult,
    RegressionQuality,
)

#: Every declared family, so a new one cannot be added without being checked.
ALL_DECLARATIONS: tuple[BenchmarkDeclaration, ...] = (
    GOSS,
    QUANTIZED,
    MULTICLASS,
    RANKING,
    REGRESSION,
    VS_LIGHTGBM,
)

#: Each quality family beside the types its manifest actually carries.
#:
#: ``VS_LIGHTGBM`` is absent deliberately: it declares no per-arm metric,
#: because its headline is a ratio BETWEEN arms rather than a mean of anything
#: a result row holds. That absence is asserted below rather than left implied.
QUALITY_FAMILIES: tuple[tuple[BenchmarkDeclaration, frozenset[str], frozenset[str]], ...] = (
    (GOSS, frozenset(GossQuality.__annotations__), frozenset(GossArmResult.__annotations__)),
    (
        QUANTIZED,
        frozenset(QuantizedQuality.__annotations__),
        frozenset(QuantizedArmResult.__annotations__),
    ),
    (
        MULTICLASS,
        frozenset(MulticlassQuality.__annotations__),
        frozenset(MulticlassArmResult.__annotations__),
    ),
    (
        RANKING,
        frozenset(RankingQuality.__annotations__),
        frozenset(RankingArmResult.__annotations__),
    ),
    (
        REGRESSION,
        frozenset(RegressionQuality.__annotations__),
        frozenset(RegressionArmResult.__annotations__),
    ),
)


class TestExperimentNames:
    """The comparability key, which a copy-paste would quietly share."""

    def test_no_two_families_claim_the_same_experiment(self) -> None:
        names = [declaration["experiment"] for declaration in ALL_DECLARATIONS]
        assert len(set(names)) == len(names)

    def test_every_family_names_an_experiment(self) -> None:
        assert all(declaration["experiment"] for declaration in ALL_DECLARATIONS)

    def test_the_timing_family_keeps_the_name_its_published_records_carry(self) -> None:
        # Records written before 2026-09-09 carry this string. Renaming it
        # would make every one of them incomparable with everything written
        # after, which is a worse outcome than the inconsistency of one family
        # not matching the others' "-quality" suffix.
        assert VS_LIGHTGBM["experiment"] == "cleargbm-vs-lightgbm-fit-time"


class TestDeclaredMetricsExist:
    """A declared metric its manifest does not carry fails only at run time."""

    def test_every_declared_quality_metric_is_a_field_of_its_quality_record(
        self,
    ) -> None:
        for declaration, quality_fields, _ in QUALITY_FAMILIES:
            for metric in declaration["quality_metrics"]:
                assert metric in quality_fields, (declaration["experiment"], metric)

    def test_every_declared_result_metric_is_a_field_of_its_result_record(self) -> None:
        for declaration, _, result_fields in QUALITY_FAMILIES:
            for metric in declaration["result_metrics"]:
                assert metric in result_fields, (declaration["experiment"], metric)

    def test_every_declared_arm_axis_is_a_field_of_its_result_record(self) -> None:
        # An axis the manifest does not carry names no arm, and the harness
        # refuses it -- after the fitting has already been paid for.
        for declaration, _, result_fields in QUALITY_FAMILIES:
            for axis in declaration["arm_axes"]:
                assert axis in result_fields, (declaration["experiment"], axis)

    def test_every_quality_family_measures_something(self) -> None:
        for declaration, _, _ in QUALITY_FAMILIES:
            assert declaration["quality_metrics"], declaration["experiment"]
            assert declaration["arm_axes"], declaration["experiment"]

    def test_the_timing_family_declares_no_per_arm_metric(self) -> None:
        # Not an oversight. Declaring metrics it does not have, to make the
        # table look uniform, would be the declaration lying.
        assert VS_LIGHTGBM["arm_axes"] == ()
        assert VS_LIGHTGBM["quality_metrics"] == ()
        assert VS_LIGHTGBM["result_metrics"] == ()


class TestArguments:
    """Flags are a family's whole command-line surface."""

    def test_no_family_declares_one_flag_twice(self) -> None:
        for declaration in ALL_DECLARATIONS:
            flags = [argument["flag"] for argument in declaration["arguments"]]
            assert len(set(flags)) == len(flags), declaration["experiment"]

    def test_no_family_redeclares_a_flag_the_harness_already_owns(self) -> None:
        # --seeds and --out come from the harness for every family; a
        # redeclaration is an argparse conflict at parse time.
        for declaration in ALL_DECLARATIONS:
            flags = {argument["flag"] for argument in declaration["arguments"]}
            assert not flags & {"--seeds", "--out"}, declaration["experiment"]

    def test_every_flag_is_long_form_and_documented(self) -> None:
        for declaration in ALL_DECLARATIONS:
            for argument in declaration["arguments"]:
                assert argument["flag"].startswith("--"), declaration["experiment"]
                assert argument["help"], argument["flag"]
