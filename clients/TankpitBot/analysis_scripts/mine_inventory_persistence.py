"""Archive check: first inventory by rank + cross-session persistence."""
import json
from collections import Counter
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot.types import decode_capture_session
from tankpit_bot.validate.wire_timeline import extract_wire_timeline

rows = []
for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        session = decode_capture_session(
            narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))
        )
    except Exception:
        continue
    if session["magic"] is None:
        continue
    tl = extract_wire_timeline(session)
    if not tl["inventory_snapshots"]:
        continue
    first = tl["inventory_snapshots"][0]["counts"]
    last = tl["inventory_snapshots"][-1]["counts"]
    start_ms = session["messages"][0]["timestamp_ms"] if session["messages"] else 0
    rows.append((start_ms, path.name, tl["rank"], first, last))

rows.sort()
by_rank_first_radar = {}
cap_hits = Counter()
for _, name, rank, first, last in rows:
    by_rank_first_radar.setdefault(rank, Counter())[first[4]] += 1
    if rank is not None:
        cap = 20 + 5 * rank
        if any(c > cap for c in first) or any(c > cap for c in last):
            cap_hits[f"OVER-CAP rank {rank}"] += 1
        if max(first) == cap or max(last) == cap:
            cap_hits[f"at-cap rank {rank}"] += 1

print("first radar count by session rank:")
for rank, hist in sorted(by_rank_first_radar.items(), key=lambda kv: (kv[0] is None, kv[0])):
    top = dict(hist.most_common(6))
    print(f"  rank {rank}: n={sum(hist.values())} histogram(top) {top}")
print("cap checks:", dict(cap_hits))

print("\ncross-session persistence (last counts vs next session's first, consecutive sessions):")
matches = 0
comparisons = 0
for (t1, n1, r1, f1, l1), (t2, n2, r2, f2, l2) in zip(rows, rows[1:]):
    if r1 is None or r2 is None:
        continue
    comparisons += 1
    if l1 == f2:
        matches += 1
print(f"  exact carry-over {matches}/{comparisons} consecutive session pairs")
for (t1, n1, r1, f1, l1), (t2, n2, r2, f2, l2) in list(zip(rows, rows[1:]))[:6]:
    print(f"  {n1[:24]} last={l1}  ->  {n2[:24]} first={f2}")
