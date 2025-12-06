from __future__ import annotations

import os
from collections.abc import Generator, Sequence
from pathlib import Path


def list_text_files(root: str) -> list[str]:
    p = Path(root)
    if p.is_file():
        return [str(p)]
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".txt", ".text")):
                paths.append(str(Path(dirpath) / name))
    return paths


def iter_lines(files: Sequence[str]) -> Generator[str, None, None]:
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield s


def count_lines(files: Sequence[str]) -> int:
    n = 0
    for _ in iter_lines(files):
        n += 1
    return n


def sample_lines(files: Sequence[str], k: int, *, seed: int) -> list[str]:
    import random

    if k <= 0:
        return []
    rng = random.Random(seed)
    reservoir: list[str] = []
    for i, s in enumerate(iter_lines(files), start=1):
        if len(reservoir) < k:
            reservoir.append(s)
        else:
            j = rng.randint(1, i)
            if j <= k:
                reservoir[j - 1] = s
    return reservoir
