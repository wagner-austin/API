#!/usr/bin/env python3
"""Analyze phantom instantiations in OUR source files."""

import json
import re
import subprocess

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

# Find phantom functions in OUR source files
our_phantoms = []
for func in d["data"][0]["functions"]:
    filenames = func.get("filenames", [])
    # Check if any filename is in our source
    if not any("cleargbm_rs\\src" in f for f in filenames):
        continue

    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", func["name"])
    if match:
        crate_hash = match.group(1)
        if crate_hash != main_hash:
            our_phantoms.append((crate_hash, func))

print(f"\nFound {len(our_phantoms)} phantom functions in OUR source files")

# Group by crate hash
by_hash: dict[str, list] = {}
for h, func in our_phantoms:
    if h not in by_hash:
        by_hash[h] = []
    by_hash[h].append(func)

print(f"\nPhantom crate hashes in our files: {list(by_hash.keys())}")

for h, funcs in by_hash.items():
    print(f"\n=== Crate hash: {h} ({len(funcs)} functions) ===")
    for func in funcs[:10]:
        name = func["name"]

        # Try to demangle
        try:
            result = subprocess.run(
                ["rustfilt", name],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                demangled = result.stdout.strip()
            else:
                demangled = name[:120]
        except Exception:
            demangled = name[:120]

        filenames = func.get("filenames", [])
        count = func.get("count", 0)
        regions = func.get("regions", [])
        uncovered_regions = len([r for r in regions if r[4] == 0])

        print(f"\n  Demangled: {demangled[:100]}")
        print(f"  Count: {count}, Regions: {len(regions)}, Uncovered: {uncovered_regions}")
        for f in filenames:
            if "cleargbm_rs" in f:
                print(f"  File: {f}")
