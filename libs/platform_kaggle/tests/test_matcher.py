"""Tests for platform_kaggle.matcher module."""

from __future__ import annotations

from platform_kaggle.matcher import (
    _calculate_match_score,
    _determine_recommendation,
    _extract_requirements_from_description,
    _infer_competition_requirements,
    _normalize_tag,
    _tags_overlap,
    match_competition,
    match_competitions,
)
from platform_kaggle.testing import (
    make_fake_capability,
    make_fake_competition,
    make_fake_competition_pages,
)
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
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "xgboost_tabular" in reqs
        assert "lightgbm_tabular" in reqs
        assert unmapped == ()

    def test_structured_competition(self) -> None:
        """Test inferring requirements for structured data competition."""
        comp = make_fake_competition(tags=("structured",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "xgboost_tabular" in reqs
        assert unmapped == ()

    def test_classification_competition(self) -> None:
        """Test inferring requirements for classification competition."""
        comp = make_fake_competition(tags=("classification",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "sklearn_ml" in reqs
        assert unmapped == ()

    def test_time_series_competition(self) -> None:
        """Test inferring requirements for time series competition."""
        comp = make_fake_competition(tags=("time-series",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "pytorch_deep_learning" in reqs
        assert unmapped == ()

    def test_forecasting_competition(self) -> None:
        """Test inferring requirements for forecasting competition."""
        comp = make_fake_competition(tags=("forecasting",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "pytorch_deep_learning" in reqs
        assert unmapped == ()

    def test_nlp_competition(self) -> None:
        """Test inferring requirements for NLP competition."""
        comp = make_fake_competition(tags=("nlp",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "huggingface_transformers" in reqs
        assert unmapped == ()

    def test_text_competition(self) -> None:
        """Test inferring requirements for text competition."""
        comp = make_fake_competition(tags=("text",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "huggingface_transformers" in reqs
        assert unmapped == ()

    def test_speech_competition(self) -> None:
        """Test inferring requirements for speech competition."""
        comp = make_fake_competition(tags=("speech",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "speech_to_text" in reqs
        assert unmapped == ()

    def test_audio_competition(self) -> None:
        """Test inferring requirements for audio competition."""
        comp = make_fake_competition(tags=("audio",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "speech_to_text" in reqs
        assert unmapped == ()

    def test_optimization_competition(self) -> None:
        """Test inferring requirements for optimization competition."""
        comp = make_fake_competition(tags=("optimization",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "hyperparameter_optimization" in reqs
        assert unmapped == ()

    def test_no_requirements_returns_unmapped(self) -> None:
        """Test unknown tags are returned as unmapped."""
        comp = make_fake_competition(tags=("unique-tag",))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert reqs == ()
        assert "unique-tag" in unmapped

    def test_mixed_mapped_and_unmapped(self) -> None:
        """Test competition with both mapped and unmapped tags."""
        comp = make_fake_competition(tags=("tabular", "custom-metric", "mathematics"))
        reqs, unmapped = _infer_competition_requirements(comp)
        assert "xgboost_tabular" in reqs
        assert "custom-metric" in unmapped
        assert "mathematics" in unmapped


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

    def test_unmapped_tag_covered_by_capability_tags(self) -> None:
        """Test unmapped competition tag doesn't count as missing if covered by capability."""
        # "finance" is not in _TAG_CAPABILITY_MAP but our capability has it as a tag
        comp = make_fake_competition(tags=("tabular", "finance"))
        cap = make_fake_capability(
            name="xgboost_tabular",
            tags=("tabular", "finance"),  # Capability covers "finance"
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=(),
            data_formats=(),
            task_types=(),
        )

        match = match_competition(comp, profile)

        # "finance" is unmapped but covered by capability tags, so NOT missing
        assert "finance" not in match.missing_capabilities
        assert "xgboost_tabular" in match.matched_capabilities


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

    def test_match_with_pages_map(self) -> None:
        """Test matching with pages map for description-based matching."""
        comps = (make_fake_competition(ref="comp1", tags=("tabular",)),)
        profile = CodebaseProfile(
            capabilities=(),
            ml_backends=(),
            data_formats=(),
            task_types=(),
        )
        # Pages require mobile development (hard requirement)
        pages = make_fake_competition_pages(
            description="Build an on-device mobile app",
        )
        pages_map = {"comp1": pages}

        matches = match_competitions(comps, profile, pages_map=pages_map)

        assert len(matches) == 1
        # Missing mobile_development hard requirement should reduce score
        assert "mobile_development" in matches[0].missing_capabilities


class TestExtractRequirementsFromDescription:
    """Tests for _extract_requirements_from_description function."""

    def test_extracts_gemma_hard_requirement(self) -> None:
        """Test extracting Gemma as hard requirement."""
        pages = make_fake_competition_pages(
            description="Use Gemma 3n to build an app",
        )
        hard, _soft = _extract_requirements_from_description(pages)
        assert "gemma_model" in hard

    def test_extracts_llama_soft_requirement(self) -> None:
        """Test extracting LLaMA as soft requirement (mentioned, not mandated)."""
        pages = make_fake_competition_pages(
            description="Fine-tune LLaMA for your task",
        )
        _hard, soft = _extract_requirements_from_description(pages)
        assert "llama_model" in soft

    def test_extracts_mobile_hard_requirement(self) -> None:
        """Test extracting mobile as hard requirement."""
        pages = make_fake_competition_pages(
            description="Build an on-device mobile-first solution",
        )
        hard, _soft = _extract_requirements_from_description(pages)
        assert "mobile_development" in hard

    def test_extracts_video_submission_hard_requirement(self) -> None:
        """Test extracting video submission as hard requirement."""
        pages = make_fake_competition_pages(
            description="Submit a video demonstrating your solution",
        )
        hard, _soft = _extract_requirements_from_description(pages)
        assert "video_production" in hard

    def test_extracts_pytorch_soft_requirement(self) -> None:
        """Test extracting PyTorch as soft requirement."""
        pages = make_fake_competition_pages(
            description="Use PyTorch to train your model",
        )
        _hard, soft = _extract_requirements_from_description(pages)
        assert "pytorch_deep_learning" in soft

    def test_extracts_xgboost_soft_requirement(self) -> None:
        """Test extracting XGBoost as soft requirement."""
        pages = make_fake_competition_pages(
            description="XGBoost baseline available in the starter notebook",
        )
        _hard, soft = _extract_requirements_from_description(pages)
        assert "xgboost_tabular" in soft

    def test_extracts_from_evaluation(self) -> None:
        """Test extracting requirements from evaluation section."""
        pages = make_fake_competition_pages(
            description="Train a model",
            evaluation="Your sklearn model should achieve high accuracy",
        )
        _hard, soft = _extract_requirements_from_description(pages)
        assert "sklearn_ml" in soft

    def test_extracts_from_rules(self) -> None:
        """Test extracting requirements from rules section."""
        pages = make_fake_competition_pages(
            description="Train a model",
            rules="Models must run on-device",
        )
        hard, _soft = _extract_requirements_from_description(pages)
        assert "mobile_development" in hard

    def test_no_requirements(self) -> None:
        """Test no requirements extracted from generic description."""
        pages = make_fake_competition_pages(
            description="Predict the target variable",
        )
        hard, soft = _extract_requirements_from_description(pages)
        assert hard == ()
        assert soft == ()


class TestCalculateMatchScoreWithHardRequirements:
    """Tests for _calculate_match_score with hard requirements."""

    def test_missing_hard_caps_max_score(self) -> None:
        """Test missing hard requirements cap the max score."""
        # With 1 missing hard cap, max score is 0.3 - 0.15 = 0.15
        score = _calculate_match_score(
            ("cap1", "cap2"),
            (),
            ("hard_cap1",),
        )
        assert score <= 0.15

    def test_multiple_missing_hard_caps(self) -> None:
        """Test multiple missing hard requirements reduce score further."""
        score = _calculate_match_score(
            ("cap1",),
            (),
            ("hard1", "hard2"),
        )
        # 2 missing hard caps: max_score = 0.3 - 0.30 = 0.0
        assert score == 0.0

    def test_no_missing_hard_caps_normal_score(self) -> None:
        """Test no missing hard caps allows normal score calculation."""
        score = _calculate_match_score(
            ("cap1", "cap2"),
            (),
            (),
        )
        assert score == 1.0


class TestMatchCompetitionWithPages:
    """Tests for match_competition with pages parameter."""

    def test_match_with_pages_hard_requirement_missing(self) -> None:
        """Test matching with missing hard requirement from description."""
        comp = make_fake_competition(tags=("tabular",))
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
        # Description requires Gemma (hard requirement we don't have)
        pages = make_fake_competition_pages(
            description="Use Gemma 3n for this tabular task",
        )

        match = match_competition(comp, profile, pages)

        assert "gemma_model" in match.missing_capabilities
        # Score should be capped due to missing hard requirement
        assert match.match_score <= 0.3

    def test_match_with_pages_soft_requirement_matched(self) -> None:
        """Test matching with soft requirement from description."""
        comp = make_fake_competition(tags=("tabular",))
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
        # Description mentions XGBoost (soft requirement we have)
        pages = make_fake_competition_pages(
            description="Use XGBoost or LightGBM for best results",
        )

        match = match_competition(comp, profile, pages)

        assert "xgboost_tabular" in match.matched_capabilities

    def test_match_with_empty_pages(self) -> None:
        """Test matching with empty pages uses only tags."""
        comp = make_fake_competition(tags=("tabular",))
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
        # Empty pages (no description text)
        pages = make_fake_competition_pages(
            description="",
            evaluation="",
            rules="",
        )

        match = match_competition(comp, profile, pages)

        assert match.match_score > 0.3  # Normal scoring

    def test_match_pages_with_matched_hard_requirement(self) -> None:
        """Test matching when we have the hard requirement capability."""
        comp = make_fake_competition(tags=())
        cap = make_fake_capability(
            name="gemma_model",
            tags=("llm", "gemma"),
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=(),
            data_formats=(),
            task_types=(),
        )
        # Description requires Gemma (hard requirement we have)
        pages = make_fake_competition_pages(
            description="Build with Gemma 3n",
        )

        match = match_competition(comp, profile, pages)

        assert "gemma_model" in match.matched_capabilities
        assert "gemma_model" not in match.missing_capabilities
