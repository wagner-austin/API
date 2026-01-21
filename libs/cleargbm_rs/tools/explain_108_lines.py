#!/usr/bin/env python3
"""Explain where the 108 lines come from in histogram_buffer.rs."""

import json
import re

d = json.load(open("coverage.json"))
target = "histogram_buffer.rs"

# Get main crate hash
main_hash = ""
for func in d["data"][0]["functions"]:
    if "cleargbm_rs" in func["name"]:
        match = re.search(r"(Cs[A-Za-z0-9]{10,14})_\d+cleargbm_rs", func["name"])
        if match:
            main_hash = match.group(1)
            break
print(f"Main hash: {main_hash}")

# Count lines per instantiation
by_hash: dict[str, dict[str, set[int]]] = {}

for func in d["data"][0]["functions"]:
    filenames = func.get("filenames", [])
    if not any(target in f and "serde_impl" in f for f in filenames):
        continue

    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", func["name"])
    if not match:
        continue
    h = match.group(1)

    if h not in by_hash:
        by_hash[h] = {"all": set(), "covered": set()}

    for r in func.get("regions", []):
        line = r[0]
        by_hash[h]["all"].add(line)
        if r[4] > 0:
            by_hash[h]["covered"].add(line)

print("\n=== Lines per crate hash ===")
total_lines_all = set()
total_lines_covered = set()

for h, data in sorted(by_hash.items()):
    tag = "MAIN" if h == main_hash else "PHANTOM"
    all_l = data["all"]
    cov_l = data["covered"]
    total_lines_all.update(all_l)
    total_lines_covered.update(cov_l)
    print(f"  {tag} ({h}): {len(cov_l)}/{len(all_l)} lines")

print(f"\n=== Aggregated (union of all instantiations) ===")
print(f"  Total unique lines: {len(total_lines_all)}")
print(f"  Covered lines: {len(total_lines_covered)}")
print(f"  Uncovered: {len(total_lines_all - total_lines_covered)}")

uncovered = sorted(total_lines_all - total_lines_covered)
if uncovered:
    print(f"  Uncovered line numbers: {uncovered}")

# Check file summary
for f in d["data"][0]["files"]:
    if target in f["filename"] and "serde_impl" in f["filename"]:
        s = f["summary"]
        print(f"\n=== LLVM Summary ===")
        print(f"  Lines: {s['lines']['covered']}/{s['lines']['count']}")
        print(f"  The 'count' ({s['lines']['count']}) should equal total unique lines ({len(total_lines_all)})")
        break
