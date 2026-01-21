#!/usr/bin/env python3
"""Find truly uncovered lines in histogram_buffer.rs."""

import json
from pathlib import Path

d = json.load(open("coverage.json"))

# Find the target file
target = None
for f in d["data"][0]["files"]:
    if "histogram_buffer.rs" in f["filename"] and "serde_impl" in f["filename"]:
        target = f
        break

if not target:
    print("File not found")
    exit(1)

print(f"File: {target['filename']}")
s = target["summary"]
print(f"Lines: {s['lines']['covered']}/{s['lines']['count']} ({s['lines']['percent']:.1f}%)")

# Parse segments to build line coverage map
# Segment format: [Line, Col, Count, HasCount, IsRegionEntry, IsGapRegion]
segments = target.get("segments", [])

# Build a map of line -> max execution count
line_counts: dict[int, int] = {}
for seg in segments:
    line = seg[0]
    count = seg[2]
    has_count = seg[3] if len(seg) > 3 else True
    if has_count:
        if line not in line_counts:
            line_counts[line] = count
        else:
            line_counts[line] = max(line_counts[line], count)

# Find lines with 0 count
uncovered_lines = [line for line, count in line_counts.items() if count == 0]
uncovered_lines.sort()

print(f"\nUncovered lines ({len(uncovered_lines)}):")
src_path = Path(target["filename"])
try:
    src_lines = src_path.read_text().splitlines()
except OSError:
    src_lines = []

for line in uncovered_lines:
    if src_lines and line <= len(src_lines):
        print(f"  {line:4d}: {src_lines[line-1].strip()}")
    else:
        print(f"  {line:4d}: <source not available>")

# Also show partially covered lines (count > 0)
covered_lines = [line for line, count in line_counts.items() if count > 0]
print(f"\nCovered lines: {len(covered_lines)}")
print(f"Total lines in segments: {len(line_counts)}")
