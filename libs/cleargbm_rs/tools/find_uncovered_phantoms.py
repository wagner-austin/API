#!/usr/bin/env python3
"""Find phantom instantiations with UNCOVERED regions in our source."""

import json
import re

d = json.load(open("coverage.json"))

# Find main crate hash
main_hash = ""
for func in d["data"][0]["functions"]:
    if "cleargbm_rs" in func["name"]:
        match = re.search(r"(Cs[A-Za-z0-9]{10,14})_\d+cleargbm_rs", func["name"])
        if match:
            main_hash = match.group(1)
            break

print(f"Main crate hash: {main_hash}")

# Find phantom functions with UNCOVERED regions in OUR source files
uncovered_phantoms = []
for func in d["data"][0]["functions"]:
    filenames = func.get("filenames", [])
    # Check if any filename is in our source
    if not any("cleargbm_rs\\src" in f for f in filenames):
        continue

    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", func["name"])
    if match:
        crate_hash = match.group(1)
        if crate_hash != main_hash:
            regions = func.get("regions", [])
            uncovered_regions = [r for r in regions if r[4] == 0]
            if uncovered_regions:
                uncovered_phantoms.append((crate_hash, func, uncovered_regions))

print(f"\nFound {len(uncovered_phantoms)} phantom functions with UNCOVERED regions")

for h, func, uncovered_regions in uncovered_phantoms:
    name = func["name"]
    filenames = func.get("filenames", [])
    our_files = [f for f in filenames if "cleargbm_rs" in f]

    print(f"\n=== {name[:80]} ===")
    print(f"  Crate hash: {h}")
    print(f"  Files: {our_files}")
    print(f"  Uncovered regions: {len(uncovered_regions)}")

    # Show the uncovered lines
    uncovered_lines = set()
    for r in uncovered_regions:
        # Region format: [LineStart, ColStart, LineEnd, ColEnd, ExecutionCount, ...]
        for line in range(r[0], r[2] + 1):
            uncovered_lines.add(line)

    print(f"  Uncovered lines: {sorted(uncovered_lines)[:20]}")
