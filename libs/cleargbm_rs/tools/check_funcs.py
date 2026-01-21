#!/usr/bin/env python3
"""Check function-level coverage for histogram_buffer.rs."""

import json
import re

d = json.load(open("coverage.json"))
target_file = "histogram_buffer.rs"

# Find main crate hash
main_hash = ""
for func in d["data"][0]["functions"]:
    if "cleargbm_rs" in func["name"]:
        match = re.search(r"(Cs[A-Za-z0-9]{10,14})_\d+cleargbm_rs", func["name"])
        if match:
            main_hash = match.group(1)
            break

print(f"Main crate hash: {main_hash}")

# Check functions for this file
print("\n=== FUNCTIONS FOR histogram_buffer.rs ===")
for func in d["data"][0]["functions"]:
    filenames = func.get("filenames", [])
    if not any(target_file in fn and "serde_impl" in fn for fn in filenames):
        continue

    # Extract crate hash
    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", func["name"])
    crate_hash = match.group(1) if match else "unknown"
    is_main = crate_hash == main_hash

    regions = func.get("regions", [])
    covered_regions = [r for r in regions if r[4] > 0]
    uncovered_regions = [r for r in regions if r[4] == 0]

    # Get unique lines
    all_lines = set(r[0] for r in regions)
    covered_lines = set(r[0] for r in covered_regions)
    uncovered_only_lines = all_lines - covered_lines

    count = func.get("count", 0)

    tag = "[MAIN]" if is_main else "[PHANTOM]"

    # Shorten name for display
    name_short = func["name"][:80]

    print(f"\n{tag} count={count}")
    print(f"  Name: {name_short}...")
    print(f"  Regions: {len(covered_regions)} covered, {len(uncovered_regions)} uncovered")
    print(f"  Lines: {len(covered_lines)} covered, {len(uncovered_only_lines)} only in uncovered regions")
    if uncovered_only_lines:
        print(f"  Uncovered-only lines: {sorted(uncovered_only_lines)[:10]}")
