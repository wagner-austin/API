#!/usr/bin/env python3
"""Check HTML coverage for uncovered lines."""

import re
import sys
from pathlib import Path

html_file = Path("target/llvm-cov/html/coverage/Users/Test/PROJECTS/API/libs/cleargbm_rs/src/types/serde_impl/histogram_buffer.rs.html")

if not html_file.exists():
    print(f"File not found: {html_file}")
    sys.exit(1)

html = html_file.read_text()

# Find uncovered lines (class='uncovered-line' pattern)
# HTML format: <td class='uncovered-line'></td>
uncovered_count = html.count("uncovered-line")
print(f"'uncovered-line' occurrences: {uncovered_count}")

# Find all line numbers with uncovered-line
# Pattern: L{num}...uncovered-line
matches = re.findall(r"L(\d+).*?uncovered-line", html)
print(f"Lines with uncovered-line after: {matches[:30]}")

# Also check for uncovered-region
uncovered_region = html.count("uncovered-region")
print(f"'uncovered-region' occurrences: {uncovered_region}")

# Extract lines with count 0
zero_count = re.findall(r"L(\d+).*?<pre>0</pre>", html)
print(f"Lines with count 0: {zero_count[:30]}")
