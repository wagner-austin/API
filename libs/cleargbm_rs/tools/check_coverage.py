#!/usr/bin/env python3
"""Check Rust code coverage from LLVM JSON and fail if not 100%.

This script uses the segment-level coverage data which represents
actual file coverage, ignoring phantom instantiation artifacts.

Usage:
    cargo llvm-cov --all-features --json --output-path coverage.json
    python tools/check_coverage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    json_path = Path("coverage.json")
    if not json_path.exists():
        print("ERROR: coverage.json not found")
        print("Run: cargo llvm-cov --all-features --json --output-path coverage.json")
        return 1

    data = json.load(open(json_path))

    all_pass = True
    total_covered = 0
    total_lines = 0

    print("=" * 70)
    print("COVERAGE CHECK (using segment-level data)")
    print("=" * 70)

    for file_data in data["data"][0]["files"]:
        filename = file_data["filename"]

        # Only check our source files
        if "cleargbm_rs" not in filename:
            continue

        # Skip test files in coverage check
        if "tests" in filename.lower():
            continue

        segments = file_data.get("segments", [])
        if not segments:
            continue

        # Calculate actual file coverage from segments
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

        if not line_max_count:
            continue

        covered = sum(1 for c in line_max_count.values() if c > 0)
        file_lines = len(line_max_count)
        pct = 100.0 * covered / file_lines if file_lines > 0 else 100.0

        total_covered += covered
        total_lines += file_lines

        # Get short filename for display
        short_name = Path(filename).name
        if "serde_impl" in filename:
            short_name = f"serde_impl/{short_name}"
        elif "testkit" in filename:
            short_name = f"testkit/{short_name}"

        if covered < file_lines:
            print(f"FAIL: {short_name}: {covered}/{file_lines} ({pct:.1f}%)")
            uncovered = sorted([ln for ln, c in line_max_count.items() if c == 0])
            print(f"      Uncovered lines: {uncovered[:10]}{'...' if len(uncovered) > 10 else ''}")
            all_pass = False
        else:
            print(f"OK:   {short_name}: {covered}/{file_lines} (100.0%)")

    print("=" * 70)
    total_pct = 100.0 * total_covered / total_lines if total_lines > 0 else 100.0
    print(f"TOTAL: {total_covered}/{total_lines} ({total_pct:.1f}%)")
    print("=" * 70)

    if all_pass:
        print("\nAll files have 100% coverage.")
        return 0
    else:
        print("\nCoverage check FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
