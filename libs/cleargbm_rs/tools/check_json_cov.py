#!/usr/bin/env python3
"""Check JSON coverage for uncovered regions."""

import json
from pathlib import Path

json_file = Path("coverage.json")
if not json_file.exists():
    print("Run: cargo llvm-cov --all-features --json --output-path coverage.json")
    exit(1)

d = json.load(open(json_file))

# Check functions with uncovered regions
print("=== FUNCTIONS WITH UNCOVERED REGIONS ===")
for func in d["data"][0]["functions"]:
    filenames = func.get("filenames", [""])
    if not any("histogram_buffer" in fn for fn in filenames):
        continue

    regions = func.get("regions", [])
    # Region format: [LineStart, ColumnStart, LineEnd, ColumnEnd, ExecutionCount, FileID, ExpandedFileID, Kind]
    uncovered = [r for r in regions if r[4] == 0]

    if uncovered:
        name = func["name"][:100]
        print(f"\nFunction: {name}")
        print(f"  Total regions: {len(regions)}, Uncovered: {len(uncovered)}")
        for r in uncovered[:5]:
            print(f"    Lines {r[0]}-{r[2]}, Cols {r[1]}-{r[3]}, count={r[4]}")

# Check file summary
print("\n=== FILE SUMMARY ===")
for f in d["data"][0]["files"]:
    if "histogram_buffer" in f["filename"]:
        s = f["summary"]
        print(f"File: {f['filename']}")
        print(f"  Lines: {s['lines']['covered']}/{s['lines']['count']} ({s['lines']['percent']:.1f}%)")
        print(f"  Regions: {s['regions']['covered']}/{s['regions']['count']} ({s['regions']['percent']:.1f}%)")
        print(f"  Functions: {s['functions']['covered']}/{s['functions']['count']} ({s['functions']['percent']:.1f}%)")
        print(f"  Instantiations: {s['instantiations']['covered']}/{s['instantiations']['count']} ({s['instantiations']['percent']:.1f}%)")
        break
