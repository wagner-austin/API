#!/usr/bin/env python3
"""Analyze why summary differs from segments."""

import json

d = json.load(open("coverage.json"))

for f in d["data"][0]["files"]:
    if "histogram_buffer.rs" in f["filename"] and "serde_impl" in f["filename"]:
        print(f"File: {f['filename']}")
        s = f["summary"]

        print("\n=== SUMMARY ===")
        for key in ["lines", "regions", "functions", "instantiations"]:
            if key in s:
                info = s[key]
                print(f"  {key}: {info['covered']}/{info['count']} ({info['percent']:.1f}%)")

        print("\n=== SEGMENTS ===")
        segments = f.get("segments", [])
        print(f"  Total segments: {len(segments)}")

        # Count unique lines
        lines_seen: dict[int, int] = {}
        for seg in segments:
            line = seg[0]
            count = seg[2]
            has_count = seg[3] if len(seg) > 3 else True
            if has_count:
                if line not in lines_seen:
                    lines_seen[line] = count
                else:
                    lines_seen[line] = max(lines_seen[line], count)

        covered = sum(1 for c in lines_seen.values() if c > 0)
        total = len(lines_seen)
        print(f"  Unique lines in segments: {total}")
        print(f"  Covered lines: {covered}")
        print(f"  Uncovered lines: {total - covered}")

        if total - covered > 0:
            uncovered = [ln for ln, c in lines_seen.items() if c == 0]
            print(f"  Uncovered line numbers: {sorted(uncovered)}")

        # The summary says 108 lines, but segments only have 95 unique lines
        # Let's check what's missing
        print(f"\n=== DISCREPANCY ===")
        print(f"  Summary claims {s['lines']['count']} lines")
        print(f"  Segments have {total} unique lines")
        print(f"  Difference: {s['lines']['count'] - total} lines")

        break
