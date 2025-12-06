from __future__ import annotations

from pathlib import Path

import pytest
import scripts.guard as guard


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while not (p / "libs").is_dir():
        if p.parent == p:
            raise RuntimeError("libs directory not found for tests")
        p = p.parent
    return p


def test_guard_find_monorepo_root_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        guard._find_monorepo_root(tmp_path)


def test_guard_main_verbose_and_root() -> None:
    root = _repo_root()
    rc = guard.main(["--root", str(root), "--verbose"])
    assert isinstance(rc, int) and rc in (0, 2)


# Coverage for line 52: unrecognized argument is skipped (else branch)
def test_guard_main_unrecognized_argument() -> None:
    root = _repo_root()
    # Pass an unrecognized argument that is not --root or --verbose/-v
    # This exercises the else branch at line 52 where idx += 1
    rc = guard.main(["--root", str(root), "--unknown-flag", "some-value"])
    assert isinstance(rc, int) and rc in (0, 2)
