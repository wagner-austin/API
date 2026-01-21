#!/usr/bin/env python3
"""Check how LLVM calculates file summary."""

import json

d = json.load(open("coverage.json"))

for f in d["data"][0]["files"]:
    if "histogram_buffer.rs" in f["filename"] and "serde_impl" in f["filename"]:
        print(f"File: {f['filename']}")
        s = f["summary"]
        print(f"  Lines: {s['lines']['covered']}/{s['lines']['count']} ({s['lines']['percent']:.1f}%)")
        print(f"  Regions: {s['regions']['covered']}/{s['regions']['count']} ({s['regions']['percent']:.1f}%)")
        print(f"  Instantiations: {s['instantiations']['covered']}/{s['instantiations']['count']} ({s['instantiations']['percent']:.1f}%)")

        # Check segments (aggregated coverage)
        segments = f.get("segments", [])
        line_max: dict[int, int] = {}
        for seg in segments:
            line = seg[0]
            count = seg[2]
            has_count = seg[3] if len(seg) > 3 else True
            if has_count:
                if line not in line_max:
                    line_max[line] = count
                else:
                    line_max[line] = max(line_max[line], count)

        seg_covered = sum(1 for c in line_max.values() if c > 0)
        print(f"\n  Segments: {seg_covered}/{len(line_max)} covered")
        break
