"""Rendering a growth-policy report as text.

Pure string building over a decoded report: no clock, no I/O and no vendor, so
the layout is testable from a fixed record. The caller decides where the text
goes, which keeps this module free of the print ban and of stdout entirely.

Column widths are fixed so the arms line up as a table when read in a terminal
or pasted into a document, which is where these results end up.
"""

from __future__ import annotations

from .types import GrowthPolicyReport

_ARM_WIDTH = 24
_HEADER = (
    f"{'arm':<{_ARM_WIDTH}} {'fit s':>8} {'AUC-ROC':>8} {'AUC-PR':>8} {'log-loss':>9} {'leaves':>7}"
)


def render_dataset_line(report: GrowthPolicyReport) -> str:
    """Describe the dataset a report was measured on.

    Args:
        report: The report to describe.

    Returns:
        A single line naming the dataset, its shape, and its positive rate.
    """
    dataset = report["dataset"]
    positive_pct = dataset["positive_rate"] * 100.0
    return (
        f"{dataset['name']}: {dataset['row_count']} x {dataset['feature_count']}, "
        f"positive {positive_pct:.2f}%"
    )


def render_report(report: GrowthPolicyReport) -> str:
    """Render a report's per-arm summary table.

    Args:
        report: The report to render.

    Returns:
        The dataset line, a blank line, the column header, and one row per arm,
        newline-separated and newline-terminated.
    """
    lines = [render_dataset_line(report), "", _HEADER]
    for summary in report["summaries"]:
        lines.append(
            f"{summary['arm']:<{_ARM_WIDTH}} "
            f"{summary['fit_seconds']:>8.3f} "
            f"{summary['auc_roc']:>8.4f} "
            f"{summary['auc_pr']:>8.4f} "
            f"{summary['log_loss']:>9.4f} "
            f"{summary['mean_leaves']:>7.1f}"
        )
    return "\n".join(lines) + "\n"


__all__ = ["render_dataset_line", "render_report"]
