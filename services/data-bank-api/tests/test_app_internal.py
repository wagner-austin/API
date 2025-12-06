from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Never

from _pytest.monkeypatch import MonkeyPatch

from data_bank_api.health import _is_writable


def test__is_writable_handles_oserror(monkeypatch: MonkeyPatch) -> None:
    def _raise(*args: str, **kwargs: str) -> Never:
        raise OSError("denied")

    monkeypatch.setattr(tempfile, "mkstemp", _raise)
    ok = _is_writable(Path(tempfile.gettempdir()) / "nope")
    assert ok is False
