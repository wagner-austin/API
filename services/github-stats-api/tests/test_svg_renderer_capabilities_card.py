"""Tests for svg renderer validation: RenderCapabilitiesCard."""

from __future__ import annotations

from github_stats_api.api.schemas.stats import (
    CapabilitiesResponse,
    Capability,
)
from github_stats_api.renderers import (
    render_capabilities_card,
)


class TestRenderCapabilitiesCardWithEffects:
    """Tests for render_capabilities_card with visual effects themes."""

    def test_render_capabilities_card_radical_has_effects(self) -> None:
        """Test that radical theme includes all visual effects."""
        cap: Capability = {
            "name": "test_cap",
            "strength": "strong",
            "tags": (),
            "description": "Test",
        }

        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (cap,),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="radical",
            hide_border=False,
            disable_animations=False,
        )

        assert "<defs>" in svg
        assert "linearGradient" in svg
        assert 'id="caps-grad"' in svg
        assert 'class="sparkles"' in svg
        assert ".glow-text" in svg


class TestRenderCapabilitiesCard:
    """Tests for render_capabilities_card function."""

    def test_render_capabilities_card_basic(self) -> None:
        """Test rendering capabilities card."""
        cap: Capability = {
            "name": "xgboost_tabular",
            "strength": "strong",
            "tags": ("ml", "tabular"),
            "description": "XGBoost gradient boosting",
        }

        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (cap,),
            "ml_backends": ("xgboost",),
            "frameworks": ("fastapi",),
            "data_formats": ("csv",),
            "task_types": ("binary_classification",),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="default",
            hide_border=False,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Codebase Capabilities" in svg
        assert "Xgboost Tabular" in svg  # Title-cased
        assert "strong" in svg
        assert "xgboost" in svg
        assert "Binary Classification" in svg

    def test_render_capabilities_card_multiple_capabilities(self) -> None:
        """Test rendering capabilities card with multiple capabilities."""
        caps: tuple[Capability, ...] = (
            {
                "name": "xgboost_tabular",
                "strength": "strong",
                "tags": ("ml",),
                "description": "XGBoost",
            },
            {
                "name": "fastapi_rest",
                "strength": "moderate",
                "tags": ("web",),
                "description": "FastAPI REST",
            },
            {
                "name": "pytorch_cv",
                "strength": "basic",
                "tags": ("cv",),
                "description": "PyTorch CV",
            },
        )

        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": caps,
            "ml_backends": ("xgboost", "pytorch"),
            "frameworks": ("fastapi",),
            "data_formats": ("csv", "parquet"),
            "task_types": ("binary_classification", "image_classification"),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="dracula",
            hide_border=True,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "Xgboost Tabular" in svg
        assert "Fastapi Rest" in svg
        assert "Pytorch Cv" in svg
        assert "strong" in svg
        assert "moderate" in svg
        assert "basic" in svg
        # Check strength classes
        assert "strength-strong" in svg
        assert "strength-moderate" in svg
        assert "strength-basic" in svg

    def test_render_capabilities_card_no_capabilities(self) -> None:
        """Test rendering capabilities card with no capabilities."""
        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="default",
            hide_border=False,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Codebase Capabilities" in svg

    def test_render_capabilities_card_with_theme(self) -> None:
        """Test that theme colors are applied."""
        cap: Capability = {
            "name": "test_cap",
            "strength": "strong",
            "tags": (),
            "description": "Test",
        }

        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (cap,),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="dracula",
            hide_border=False,
            disable_animations=False,
        )

        assert "#282a36" in svg  # dracula bg color
        assert "#ff79c6" in svg  # dracula title color

    def test_render_capabilities_card_many_task_types(self) -> None:
        """Test rendering card with more than 6 task types shows +N more."""
        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (
                "type1",
                "type2",
                "type3",
                "type4",
                "type5",
                "type6",
                "type7",
                "type8",
            ),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="default",
            hide_border=False,
            disable_animations=False,
        )

        assert "+2 more" in svg

    def test_render_capabilities_card_hide_border(self) -> None:
        """Test hiding border on capabilities card."""
        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="default",
            hide_border=True,
            disable_animations=False,
        )

        assert 'stroke-opacity="0"' in svg
