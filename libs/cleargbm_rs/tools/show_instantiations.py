#!/usr/bin/env python3
"""Show instantiation coverage details."""

import json
from pathlib import Path

d = json.load(open("coverage.json"))
for f in d["data"][0]["files"]:
    if "histogram_buffer" in f["filename"] and "serde_impl" in f["filename"]:
        print(f"File: {Path(f['filename']).name}")
        inst = f["summary"]["instantiations"]
        print(f"  Instantiations: {inst['covered']}/{inst['count']} covered")
        print(f"  Functions: {f['summary']['functions']['count']}")
        print(f"  Lines: {f['summary']['lines']['covered']}/{f['summary']['lines']['count']}")
        print(f"  Regions: {f['summary']['regions']['covered']}/{f['summary']['regions']['count']}")
