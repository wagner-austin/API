#!/usr/bin/env python3
"""Check truly uncovered lines in types/mod.rs."""

import json
import re

d = json.load(open("coverage.json"))

# Get main crate hash
main_hash = ""
for func in d["data"][0]["functions"]:
    if "cleargbm_rs" in func["name"]:
        match = re.search(r"(Cs[A-Za-z0-9]{10,14})_\d+cleargbm_rs", func["name"])
        if match:
            main_hash = match.group(1)
            break
print(f"Main hash: {main_hash}")

# For types/mod.rs, check which lines are covered by ANY instantiation
target = "types\\mod.rs"
all_lines: set[int] = set()
covered_lines: set[int] = set()

for func in d["data"][0]["functions"]:
    filenames = func.get("filenames", [])
    if not any(target in f for f in filenames):
        continue

    for r in func.get("regions", []):
        line = r[0]
        all_lines.add(line)
        if r[4] > 0:
            covered_lines.add(line)

truly_uncovered = all_lines - covered_lines
print(f"types/mod.rs: {len(covered_lines)}/{len(all_lines)} lines covered")
print(f"Truly uncovered ({len(truly_uncovered)}): {sorted(truly_uncovered)[:30]}")

# Check main vs phantom
print("\n=== By crate hash ===")
hashes: dict[str, tuple[set[int], set[int]]] = {}

for func in d["data"][0]["functions"]:
    filenames = func.get("filenames", [])
    if not any(target in f for f in filenames):
        continue

    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", func["name"])
    if not match:
        continue
    h = match.group(1)

    if h not in hashes:
        hashes[h] = (set(), set())

    for r in func.get("regions", []):
        line = r[0]
        hashes[h][0].add(line)  # all lines
        if r[4] > 0:
            hashes[h][1].add(line)  # covered lines

for h, (all_l, cov_l) in hashes.items():
    tag = "MAIN" if h == main_hash else "PHANTOM"
    print(f"  {tag} ({h}): {len(cov_l)}/{len(all_l)} covered")
