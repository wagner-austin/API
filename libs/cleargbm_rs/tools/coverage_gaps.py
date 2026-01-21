#!/usr/bin/env python3
"""Find uncovered regions in Rust code from LLVM coverage JSON.

Usage:
    cargo llvm-cov --all-features --json --output-path coverage.json
    python tools/coverage_gaps.py [--file PATTERN]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def extract_crate_hash(mangled: str) -> str:
    """Extract crate hash from mangled Rust symbol."""
    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", mangled)
    if match:
        return match.group(1)
    return "unknown"


def demangle_function(mangled: str) -> str:
    """Extract readable function name from mangled symbol."""
    matches = re.findall(r"(\d+)([A-Za-z_][A-Za-z0-9_]*)", mangled)
    parts = []
    for length_str, name in matches:
        length = int(length_str)
        if len(name) >= length and length > 2:
            parts.append(name[:length])

    if parts:
        meaningful = [p for p in parts if len(p) > 3 and p not in ("impl", "serde")]
        return "::".join(meaningful[-4:])
    return mangled[:60]


def get_source_lines(src_path: Path, start: int, end: int) -> list[str]:
    """Get source lines from file."""
    try:
        lines = src_path.read_text().splitlines()
        return [lines[i] for i in range(start - 1, min(end, len(lines)))]
    except (OSError, IndexError):
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Find uncovered Rust code regions")
    parser.add_argument("--file", "-f", dest="file_filter")
    parser.add_argument("--json", default="coverage.json", help="Coverage JSON file")
    parser.add_argument("--show-phantom", action="store_true", help="Show phantom functions")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        print("Run: cargo llvm-cov --all-features --json --output-path coverage.json")
        return 1

    data = json.load(open(json_path))

    # Find the main crate hash by looking for "cleargbm_rs" in function names
    main_hash: str = ""
    for func in data["data"][0]["functions"]:
        name = func["name"]
        if "cleargbm_rs" in name:
            match = re.search(r"(Cs[A-Za-z0-9]{10,14})_\d+cleargbm_rs", name)
            if match:
                main_hash = match.group(1)
                break

    if not main_hash:
        print("ERROR: Could not determine main crate hash")
        return 1

    print(f"Main crate hash: {main_hash}")

    # Process files
    for file_data in data["data"][0]["files"]:
        filename = file_data["filename"]

        if args.file_filter and args.file_filter.lower() not in filename.lower():
            continue

        summary = file_data["summary"]
        if summary["lines"]["covered"] == summary["lines"]["count"]:
            continue  # 100% covered

        print(f"\n{'=' * 70}")
        print(f"FILE: {filename}")
        print(f"{'=' * 70}")
        print(f"  Summary (includes phantom instantiations):")
        print(f"    Lines: {summary['lines']['covered']}/{summary['lines']['count']} ({summary['lines']['percent']:.1f}%)")
        print(f"    Regions: {summary['regions']['covered']}/{summary['regions']['count']} ({summary['regions']['percent']:.1f}%)")
        print(f"    Instantiations: {summary['instantiations']['covered']}/{summary['instantiations']['count']} ({summary['instantiations']['percent']:.1f}%)")

        src_path = Path(filename)

        # Method 1: Check segments for file-level aggregated coverage
        # This is the REAL coverage - all instantiations merged
        segments = file_data.get("segments", [])
        if segments:
            # Segment format: [Line, Col, Count, HasCount, IsRegionEntry, IsGapRegion]
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
            pct = 100.0 * covered / total if total > 0 else 0.0
            print(f"\n  Actual file coverage (from segments): {covered}/{total} ({pct:.1f}%)")

            segment_uncovered = sorted([ln for ln, cnt in line_max_count.items() if cnt == 0])
            if segment_uncovered:
                print(f"\n  UNCOVERED LINES ({len(segment_uncovered)}):")
                for line in segment_uncovered:
                    source = get_source_lines(src_path, line, line)
                    if source:
                        print(f"    {line:4d}: {source[0].strip()}")
                    else:
                        print(f"    {line:4d}: <source not available>")
            else:
                print("  (All source lines are covered)")

        # Method 2: Check per-function instantiation coverage
        # Collect ALL lines mentioned in ANY instantiation (main or phantom)
        all_lines: set[int] = set()
        covered_lines: set[int] = set()
        uncovered_lines_by_func: dict[str, set[int]] = {}

        for func in data["data"][0]["functions"]:
            filenames = func.get("filenames", [])
            if filename not in filenames:
                continue

            crate_hash = extract_crate_hash(func["name"])
            is_phantom = crate_hash != main_hash

            regions = func.get("regions", [])
            func_uncovered: set[int] = set()

            for r in regions:
                line = r[0]
                all_lines.add(line)
                if r[4] > 0:  # covered
                    covered_lines.add(line)
                else:  # uncovered
                    func_uncovered.add(line)

            # Track uncovered lines per main instantiation
            if func_uncovered and not is_phantom:
                short_name = func["name"][:60]
                uncovered_lines_by_func[short_name] = func_uncovered

        # Find lines that are in some instantiation but NEVER covered
        truly_uncovered = all_lines - covered_lines

        if truly_uncovered:
            print(f"\n  TRULY UNCOVERED LINES ({len(truly_uncovered)}):")
            print("  (Not covered by ANY instantiation, main or phantom)")
            for line in sorted(truly_uncovered):
                source = get_source_lines(src_path, line, line)
                if source:
                    print(f"    {line:4d}: {source[0].strip()}")
                else:
                    print(f"    {line:4d}: <source not available>")
        else:
            print("\n  All lines covered by at least one instantiation.")

        # Show partial coverage per instantiation
        if uncovered_lines_by_func:
            print(f"\n  PARTIAL COVERAGE ({len(uncovered_lines_by_func)} main instantiations):")
            print("  These instantiations have uncovered branches:")
            for func_name, lines in sorted(uncovered_lines_by_func.items()):
                readable = demangle_function(func_name)
                print(f"\n    {readable}")
                print(f"    Uncovered lines: {sorted(lines)}")
                for line in sorted(lines)[:5]:
                    source = get_source_lines(src_path, line, line)
                    if source:
                        print(f"      {line:4d}: {source[0].strip()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
