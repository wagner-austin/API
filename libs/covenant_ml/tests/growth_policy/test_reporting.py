"""Tests for rendering a growth-policy report as text."""

from __future__ import annotations

from covenant_ml.growth_policy.reporting import render_dataset_line, render_report

from .factories import make_arm_result, make_report


class TestRenderDatasetLine:
    """The header line describing the dataset."""

    def test_states_shape_and_positive_percentage(self) -> None:
        """The line should carry the name, both dimensions, and the rate."""
        report = make_report([make_arm_result()], [make_arm_result()])

        line = render_dataset_line(report)

        assert line == "synthetic: 100 x 4, positive 25.00%"


class TestRenderReport:
    """The per-arm summary table."""

    def test_includes_a_row_per_arm(self) -> None:
        """Two arms should produce two data rows beneath the header."""
        results = [
            make_arm_result(arm="arm-a", seed=42),
            make_arm_result(arm="arm-b", seed=42),
        ]
        report = make_report(results, results)

        lines = render_report(report).splitlines()

        assert lines[0] == "synthetic: 100 x 4, positive 25.00%"
        assert lines[1] == ""
        assert lines[2].split() == [
            "arm",
            "fit",
            "s",
            "AUC-ROC",
            "AUC-PR",
            "log-loss",
            "leaves",
        ]
        assert len(lines) == 5

    def test_formats_each_metric_to_its_own_precision(self) -> None:
        """Fit time takes three decimals, quality four, leaves one."""
        report = make_report([make_arm_result(arm="arm-a")], [make_arm_result(arm="arm-a")])

        row = render_report(report).splitlines()[3]

        assert row.split() == ["arm-a", "1.000", "0.5000", "0.2500", "0.1250", "4.0"]

    def test_ends_with_a_newline(self) -> None:
        """The rendered text should be ready to write without further joining."""
        report = make_report([make_arm_result()], [make_arm_result()])

        assert render_report(report).endswith("\n")
