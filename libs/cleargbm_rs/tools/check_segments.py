#!/usr/bin/env python3
"""Check segment lines."""

import json

d = json.load(open("coverage.json"))
for f in d["data"][0]["files"]:
    if "histogram_buffer.rs" in f["filename"] and "serde_impl" in f["filename"]:
        segments = f["segments"]
        lines = set()
        for seg in segments:
            lines.add(seg[0])
        print(f"Unique lines in segments: {len(lines)}")
        print(f"Min line: {min(lines)}, Max line: {max(lines)}")

        # Check which lines are missing between min and max
        all_possible = set(range(min(lines), max(lines) + 1))
        missing = all_possible - lines
        print(f"Missing lines (not in segments): {len(missing)}")
        print(f"Missing: {sorted(missing)}")
        break
