#!/usr/bin/env python3
"""Check segment lines with has_count filter."""

import json

d = json.load(open("coverage.json"))
for f in d["data"][0]["files"]:
    if "histogram_buffer.rs" in f["filename"] and "serde_impl" in f["filename"]:
        segments = f["segments"]

        all_lines = set()
        has_count_lines = set()
        uncovered_lines = set()

        for seg in segments:
            line = seg[0]
            count = seg[2]
            has_count = seg[3] if len(seg) > 3 else True

            all_lines.add(line)
            if has_count:
                has_count_lines.add(line)
                if count == 0:
                    uncovered_lines.add(line)

        print(f"All lines in segments: {len(all_lines)}")
        print(f"Lines with has_count=True: {len(has_count_lines)}")
        print(f"Uncovered lines (count=0 and has_count): {len(uncovered_lines)}")

        no_count_lines = all_lines - has_count_lines
        print(f"\nLines without has_count ({len(no_count_lines)}): {sorted(no_count_lines)}")

        if uncovered_lines:
            print(f"\nUncovered lines: {sorted(uncovered_lines)}")

        # Now check: summary says 108 lines, 101 covered (7 uncovered)
        # But we see 0 uncovered from segments
        # The discrepancy must be in how LLVM calculates the summary
        print(f"\n=== Summary claims 108 lines, 7 uncovered ===")
        print(f"Segments show {len(has_count_lines)} lines, {len(uncovered_lines)} uncovered")
        break
