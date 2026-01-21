#!/usr/bin/env python3
"""Analyze phantom instantiations to understand their origin."""

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

# Find all phantom functions
phantom_funcs = []
for func in d["data"][0]["functions"]:
    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", func["name"])
    if match:
        crate_hash = match.group(1)
        if crate_hash != main_hash:
            phantom_funcs.append(func)

print(f"\nFound {len(phantom_funcs)} phantom functions")

# Group by crate hash
by_hash: dict[str, list] = {}
for func in phantom_funcs:
    match = re.search(r"(Cs[A-Za-z0-9]{10,14})_", func["name"])
    if match:
        h = match.group(1)
        if h not in by_hash:
            by_hash[h] = []
        by_hash[h].append(func)

print(f"\nPhantom crate hashes: {list(by_hash.keys())}")

for h, funcs in by_hash.items():
    print(f"\n=== Crate hash: {h} ({len(funcs)} functions) ===")
    for func in funcs[:5]:
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
                demangled = name[:100]
        except Exception:
            demangled = name[:100]

        filenames = func.get("filenames", [])
        count = func.get("count", 0)
        regions = len(func.get("regions", []))

        print(f"\n  Name: {demangled}")
        print(f"  Files: {filenames}")
        print(f"  Count: {count}, Regions: {regions}")
