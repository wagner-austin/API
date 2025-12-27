"""Tests for platform_kaggle.matcher module."""

from __future__ import annotations

from platform_kaggle.matcher import (
    _calculate_match_score,
    _determine_recommendation,
    _infer_competition_requirements,
    _normalize_tag,
    _tags_overlap,
    match_competition,
    match_competitions,
)
from platform_kaggle.testing import make_fake_capability, make_fake_competition
from platform_kaggle.types import CodebaseProfile


class TestNormalizeTag:
    """Tests for _normalize_tag function."""

    def test_lowercase(self) -> None:
        """Test tag is lowercased."""
        assert _normalize_tag("TABULAR") == "tabular"

    def test_underscore_to_hyphen(self) -> None:
        """Test underscores are converted to hyphens."""
        assert _normalize_tag("binary_classification") == "binary-classification"

    def test_mixed(self) -> None:
        """Test mixed case and underscores."""
        assert _normalize_tag("Computer_Vision") == "computer-vision"


class TestTagsOverlap:
    """Tests for _tags_overlap function."""

    def test_overlapping_tags(self) -> None:
        """Test finding overlapping tags."""
        comp_tags = ("tabular", "classification", "finance")
        cap_tags = ("tabular", "regression")
        overlap = _tags_overlap(comp_tags, cap_tags)
        assert "tabular" in overlap

    def test_no_overlap(self) -> None:
        """Test no overlapping tags."""
        comp_tags = ("nlp", "text")
        cap_tags = ("tabular", "classification")
        overlap = _tags_overlap(comp_tags, cap_tags)
        assert overlap == ()

    def test_normalized_overlap(self) -> None:
        """Test overlap with normalization."""
        comp_tags = ("binary_classification",)
        cap_tags = ("binary-classification",)
        overlap = _tags_overlap(comp_tags, cap_tags)
        assert "binary-classification" in overlap


class TestInferCompetitionRequirements:
    """Tests for _infer_competition_requirements function."""

    def test_tabular_competition(self) -> None:
        """Test inferring requirements for tabular competition."""
        comp = make_fake_competition(tags=("tabular",))
        reqs = _infer_competition_requirements(comp)
        assert "xgboost_tabular" in reqs
        assert "lightgbm_tabular" in reqs

    def test_structured_competition(self) -> None:
        """Test inferring requirements for structured data competition."""
        comp = make_fake_competition(tags=("structured",))
        reqs = _infer_competition_requirements(comp)
        assert "xgboost_tabular" in reqs

    def test_classification_competition(self) -> None:
        """Test inferring requirements for classification competition."""
        comp = make_fake_competition(tags=("classification",))
        reqs = _infer_competition_requirements(comp)
        assert "sklearn_ml" in reqs

    def test_time_series_competition(self) -> None:
        """Test inferring requirements for time series competition."""
        comp = make_fake_competition(tags=("time-series",))
        reqs = _infer_competition_requirements(comp)
        assert "pytorch_deep_learning" in reqs

    def test_forecasting_competition(self) -> None:
        """Test inferring requirements for forecasting competition."""
        comp = make_fake_competition(tags=("forecasting",))
        reqs = _infer_competition_requirements(comp)
        assert "pytorch_deep_learning" in reqs

    def test_nlp_competition(self) -> None:
        """Test inferring requirements for NLP competition."""
        comp = make_fake_competition(tags=("nlp",))
        reqs = _infer_competition_requirements(comp)
        assert "language_identification" in reqs

    def test_text_competition(self) -> None:
        """Test inferring requirements for text competition."""
        comp = make_fake_competition(tags=("text",))
        reqs = _infer_competition_requirements(comp)
        assert "language_identification" in reqs

    def test_speech_competition(self) -> None:
        """Test inferring requirements for speech competition."""
        comp = make_fake_competition(tags=("speech",))
        reqs = _infer_competition_requirements(comp)
        assert "speech_to_text" in reqs

    def test_audio_competition(self) -> None:
        """Test inferring requirements for audio competition."""
        comp = make_fake_competition(tags=("audio",))
        reqs = _infer_competition_requirements(comp)
        assert "speech_to_text" in reqs

    def test_optimization_competition(self) -> None:
        """Test inferring requirements for optimization competition."""
        comp = make_fake_competition(tags=("optimization",))
        reqs = _infer_competition_requirements(comp)
        assert "hyperparameter_optimization" in reqs

    def test_no_requirements(self) -> None:
        """Test no requirements inferred for unknown tags."""
        comp = make_fake_competition(tags=("unique-tag",))
        reqs = _infer_competition_requirements(comp)
        assert reqs == ()


class TestCalculateMatchScore:
    """Tests for _calculate_match_score function."""

    def test_all_matched(self) -> None:
        """Test score when all requirements matched."""
        score = _calculate_match_score(("cap1", "cap2"), ())
        assert score == 1.0

    def test_none_matched(self) -> None:
        """Test score when no requirements matched."""
        score = _calculate_match_score((), ("cap1", "cap2"))
        assert score == 0.0

    def test_half_matched(self) -> None:
        """Test score when half requirements matched."""
        score = _calculate_match_score(("cap1",), ("cap2",))
        assert score == 0.5

    def test_no_requirements(self) -> None:
        """Test score when no requirements (default 0.5)."""
        score = _calculate_match_score((), ())
        assert score == 0.5


class TestDetermineRecommendation:
    """Tests for _determine_recommendation function."""

    def test_strong_fit(self) -> None:
        """Test strong_fit recommendation for high score."""
        assert _determine_recommendation(0.9) == "strong_fit"
        assert _determine_recommendation(0.8) == "strong_fit"

    def test_good_fit(self) -> None:
        """Test good_fit recommendation for medium score."""
        assert _determine_recommendation(0.7) == "good_fit"
        assert _determine_recommendation(0.5) == "good_fit"

    def test_stretch(self) -> None:
        """Test stretch recommendation for low-medium score."""
        assert _determine_recommendation(0.4) == "stretch"
        assert _determine_recommendation(0.2) == "stretch"

    def test_new_territory(self) -> None:
        """Test new_territory recommendation for very low score."""
        assert _determine_recommendation(0.1) == "new_territory"
        assert _determine_recommendation(0.0) == "new_territory"


class TestMatchCompetition:
    """Tests for match_competition function."""

    def test_match_with_capabilities(self) -> None:
        """Test matching competition against profile with capabilities."""
        comp = make_fake_competition(tags=("tabular", "classification"))
        cap = make_fake_capability(
            name="xgboost_tabular",
            tags=("tabular", "classification"),
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=("xgboost",),
            data_formats=("csv",),
            task_types=("binary_classification",),
        )

        match = match_competition(comp, profile)

        assert match.competition == comp
        assert "xgboost_tabular" in match.matched_capabilities
        assert match.match_score > 0

    def test_match_with_missing_capabilities(self) -> None:
        """Test matching competition with missing capabilities."""
        comp = make_fake_competition(tags=("tabular", "nlp"))
        profile = CodebaseProfile(
            capabilities=(),
            ml_backends=(),
            data_formats=(),
            task_types=(),
        )

        match = match_competition(comp, profile)

        assert match.competition.ref == comp.ref
        # With tabular+nlp tags and no capabilities, should have missing caps
        assert "xgboost_tabular" in match.missing_capabilities
        assert match.match_score < 1.0

    def test_match_tag_overlap_boost(self) -> None:
        """Test match score is boosted by tag overlap."""
        comp = make_fake_competition(tags=("classification",))
        cap = make_fake_capability(
            name="sklearn_ml",
            tags=("classification",),
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=(),
            data_formats=(),
            task_types=(),
        )

        match = match_competition(comp, profile)

        # Should have sklearn_ml as matched
        assert "sklearn_ml" in match.matched_capabilities


class TestMatchCompetitions:
    """Tests for match_competitions function."""

    def test_match_multiple(self) -> None:
        """Test matching multiple competitions."""
        comps = (
            make_fake_competition(ref="comp1", tags=("tabular",)),
            make_fake_competition(ref="comp2", tags=("nlp",)),
        )
        profile = CodebaseProfile(
            capabilities=(),
            ml_backends=(),
            data_formats=(),
            task_types=(),
        )

        matches = match_competitions(comps, profile)

        assert len(matches) == 2

    def test_match_with_min_score(self) -> None:
        """Test matching with minimum score filter."""
        comps = (
            make_fake_competition(ref="comp1", tags=("tabular",)),
            make_fake_competition(ref="comp2", tags=("unknown-tag",)),
        )
        cap = make_fake_capability(
            name="xgboost_tabular",
            tags=("tabular",),
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=(),
            data_formats=(),
            task_types=(),
        )

        # With high min_score, should filter out low matches
        matches = match_competitions(comps, profile, min_score=0.6)

        # The tabular comp should have higher score
        assert all(m.match_score >= 0.6 for m in matches)

    def test_match_sorted_by_score(self) -> None:
        """Test matches are sorted by score descending."""
        comps = (
            make_fake_competition(ref="comp1", tags=("unknown-tag",)),
            make_fake_competition(ref="comp2", tags=("tabular",)),
        )
        cap = make_fake_capability(
            name="xgboost_tabular",
            tags=("tabular",),
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=(),
            data_formats=(),
            task_types=(),
        )

        matches = match_competitions(comps, profile)

        # Should be sorted by score descending
        scores = [m.match_score for m in matches]
        assert scores == sorted(scores, reverse=True)
