#!/usr/bin/env python3
"""Check that all source lines are covered (segment-based coverage).

This script checks ACTUAL source line coverage by analyzing LLVM coverage
segments, which merge coverage across all generic instantiations. This avoids
false negatives from phantom instantiations of generic code.

Usage:
    cargo llvm-cov --all-features --json --output-path coverage.json
    python tools/check_segment_coverage.py

Args:
    --json: Path to coverage JSON file (default: coverage.json)
    --threshold: Minimum coverage percentage required (default: 100.0)

Returns:
    Exit code 0 if coverage meets threshold, 1 otherwise.

Raises:
    SystemExit: If coverage JSON is missing or coverage is below threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def get_source_line(src_path: Path, line_num: int) -> str:
    """Get a single source line from a file.

    Args:
        src_path: Path to the source file.
        line_num: 1-indexed line number to retrieve.

    Returns:
        The source line content, or "<source not available>" if not found.
    """
    try:
        lines = src_path.read_text().splitlines()
        if 0 < line_num <= len(lines):
            return lines[line_num - 1].strip()
    except OSError:
        pass
    return "<source not available>"


def check_file_segment_coverage(
    file_data: dict[str, object],
) -> tuple[int, int, list[int]]:
    """Check segment coverage for a single file.

    Segments represent merged coverage across all instantiations of generic
    code. A line is considered covered if ANY instantiation executes it.

    Args:
        file_data: File coverage data from LLVM coverage JSON.

    Returns:
        Tuple of (covered_lines, total_lines, uncovered_line_numbers).
    """
    segments = file_data.get("segments", [])
    if not segments:
        return 0, 0, []

    line_max_count: dict[int, int] = {}
    for seg in segments:
        line = seg[0]
        count = seg[2]
        has_count = seg[3] if len(seg) > 3 else True
        if has_count:
            if line not in line_max_count:
                line_max_count[line] = count
            else:
                line_max_count[line] = max(line_max_count[line], count)

    covered = sum(1 for c in line_max_count.values() if c > 0)
    total = len(line_max_count)
    uncovered = sorted(ln for ln, cnt in line_max_count.items() if cnt == 0)

    return covered, total, uncovered


def main() -> int:
    """Check segment-based coverage and fail if below threshold.

    Returns:
        0 if coverage meets threshold, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Check segment-based source line coverage"
    )
    parser.add_argument(
        "--json",
        default="coverage.json",
        help="Coverage JSON file path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Minimum coverage percentage required",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        print("Run: cargo llvm-cov --all-features --json --output-path coverage.json")
        return 1

    with open(json_path) as f:
        data = json.load(f)

    total_covered = 0
    total_lines = 0
    files_with_gaps: list[tuple[str, float, list[int]]] = []

    for file_data in data["data"][0]["files"]:
        filename = file_data["filename"]

        # Skip non-source files
        if "src\\" not in filename and "src/" not in filename:
            continue

        covered, total, uncovered = check_file_segment_coverage(file_data)
        total_covered += covered
        total_lines += total

        if uncovered:
            pct = 100.0 * covered / total if total > 0 else 0.0
            files_with_gaps.append((filename, pct, uncovered))

    overall_pct = 100.0 * total_covered / total_lines if total_lines > 0 else 0.0

    print("=" * 70)
    print("SEGMENT-BASED COVERAGE CHECK")
    print("=" * 70)
    print(f"Total: {total_covered}/{total_lines} lines ({overall_pct:.2f}%)")
    print(f"Threshold: {args.threshold:.2f}%")
    print()

    if files_with_gaps:
        print(f"FILES WITH UNCOVERED LINES ({len(files_with_gaps)}):")
        for filename, pct, uncovered in files_with_gaps:
            src_path = Path(filename)
            short_name = src_path.name
            print(f"\n  {short_name} ({pct:.1f}%):")
            for line_num in uncovered:
                source = get_source_line(src_path, line_num)
                print(f"    {line_num:4d}: {source}")
    else:
        print("All source lines are covered!")

    print()
    if overall_pct >= args.threshold:
        print(f"PASS: Coverage {overall_pct:.2f}% >= {args.threshold:.2f}%")
        return 0
    print(f"FAIL: Coverage {overall_pct:.2f}% < {args.threshold:.2f}%")
    return 1


if __name__ == "__main__":
    sys.exit(main())
