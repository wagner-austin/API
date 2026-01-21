#!/usr/bin/env python3
"""Debug coverage JSON structure."""

import json
from pathlib import Path

d = json.load(open("coverage.json"))
files = d["data"][0]["files"]
print(f"Total files: {len(files)}")

for f in files:
    if "histogram_buffer.rs" in f["filename"] and "serde_impl" in f["filename"]:
        print(f"File: {f['filename']}")
        print(f"Keys: {list(f.keys())}")
        s = f["summary"]
        print(f"  Lines: {s['lines']['covered']}/{s['lines']['count']}")
        print(f"  Regions: {s['regions']['covered']}/{s['regions']['count']}")

        # Check segments if they exist
        if "segments" in f:
            segs = f["segments"]
            print(f"  Segments: {len(segs)}")
            # Find uncovered
            uncov = [seg for seg in segs if len(seg) > 3 and seg[3] and seg[2] == 0]
            print(f"  Uncovered segments: {len(uncov)}")
            for seg in uncov[:10]:
                print(f"    Line {seg[0]}, Col {seg[1]}: count={seg[2]}")
        break
