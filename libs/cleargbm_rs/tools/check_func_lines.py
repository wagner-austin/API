#!/usr/bin/env python3
"""Check function-level line counts."""

import json
import re

d = json.load(open("coverage.json"))
target = "histogram_buffer.rs"

# Find main hash
main_hash = ""
for func in d["data"][0]["functions"]:
    if "cleargbm_rs" in func["name"]:
        match = re.search(r"(Cs[A-Za-z0-9]{10,14})_\d+cleargbm_rs", func["name"])
        if match:
            main_hash = match.group(1)
            break

print(f"Main hash: {main_hash}")

# Count lines per function
total_lines = 0
total_covered = 0
main_lines_union = set()
main_covered_union = set()

for func in d["data"][0]["functions"]:
    filenames = func.get("filenames", [])
    if not any(target in fn and "serde_impl" in fn for fn in filenames):
        continue

    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", func["name"])
    crate_hash = match.group(1) if match else "unknown"
    is_main = crate_hash == main_hash

    regions = func.get("regions", [])
    func_lines = set()
    func_covered = set()
    for r in regions:
        line = r[0]
        func_lines.add(line)
        if r[4] > 0:
            func_covered.add(line)

    tag = "MAIN" if is_main else "PHANTOM"
    print(f"[{tag}] Lines: {len(func_lines)}, Covered: {len(func_covered)}, Name: {func['name'][:50]}...")

    if is_main:
        main_lines_union.update(func_lines)
        main_covered_union.update(func_covered)

    # Sum up for total (this is how LLVM might calculate summary)
    total_lines += len(func_lines)
    total_covered += len(func_covered)

print(f"\n=== AGGREGATED ===")
print(f"Sum of all function line counts: {total_lines}")
print(f"Sum of all function covered counts: {total_covered}")
print(f"Main instantiations union: {len(main_lines_union)} lines, {len(main_covered_union)} covered")
