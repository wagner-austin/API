from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Generator
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.conftest import make_probs

import turkic_api.core.corpus_download as cd
import turkic_api.core.langid as lid
import turkic_api.core.translit as tr
from turkic_api.core.langid import LangIdModel


class ProgressStatsProtocol(Protocol):
    """Protocol for ProgressStats to enable typed testing."""

    start_time: float
    lines_kept: int
    lines_removed: int
    ipa_chars: int
    target_ipa_chars: int

    def elapsed_seconds(self) -> float: ...
    def elapsed_str(self) -> str: ...
    def lines_per_sec(self) -> float: ...
    def chars_per_sec(self) -> float: ...
    def progress_pct(self) -> float: ...
    def eta_str(self) -> str: ...
    def memory_mb(self) -> float: ...
    def reset_for_phase(self) -> None: ...


def _load_script_module(path: Path) -> ModuleType:
    spec: ModuleSpec | None = importlib.util.spec_from_file_location(
        "build_balanced_corpora", str(path)
    )
    assert spec is not None
    loader: Loader | None = spec.loader
    assert loader is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _create_stats(mod: ModuleType) -> ProgressStatsProtocol:
    """Create a ProgressStats instance with proper typing via Protocol."""
    stats_cls: type[ProgressStatsProtocol] = mod.ProgressStats
    return stats_cls()


def _stub_stream_fn(counts: dict[str, list[str]]) -> Callable[[str], Generator[str, None, None]]:
    def stream(lang: str) -> Generator[str, None, None]:
        yield from counts.get(lang, [])

    return stream


class _StubLID:
    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        return (("__label__ug",), make_probs(1.0))


def stub_load_langid_model(data_dir: str, prefer_218e: bool = True) -> LangIdModel:
    return _StubLID()


def stub_build_lang_script_filter(
    *, target_lang: str, script: str | None, threshold: float, model: LangIdModel
) -> Callable[[str], bool]:
    return lambda s: True


def stub_to_ipa(text: str, lang: str) -> str:
    return text


def test_dry_run_bottleneck_and_no_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path("scripts/build_balanced_corpora.py")
    streams = {"ug": ["aaaaa", "aaaaa"], "kk": ["a" * 50, "a" * 50]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    def stub_load_langid_model(data_dir: str, prefer_218e: bool = True) -> LangIdModel:
        return _StubLID()

    def stub_build_lang_script_filter(
        *, target_lang: str, script: str | None, threshold: float, model: LangIdModel
    ) -> Callable[[str], bool]:
        return lambda s: True

    def stub_to_ipa(text: str, lang: str) -> str:
        return text

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    mod = _load_script_module(script)
    report = tmp_path / "r.json"
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "ug,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
        "--report",
        str(report),
    ]
    main_fn: Callable[[], None] = mod.main
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    main_fn()

    text = report.read_text(encoding="utf-8")
    assert '"language": "ug"' in text
    assert not any((tmp_path / "out").glob("*.txt"))


def test_write_balanced_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path("scripts/build_balanced_corpora.py")
    streams = {"ug": ["aaaaa", "aaaaa"], "kk": ["a" * 7, "a" * 7, "a" * 7]}

    def stub_stream_oscar2(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar2
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa
    mod = _load_script_module(script)

    out_dir = tmp_path / "out"
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "ug,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(out_dir),
    ]
    main_fn: Callable[[], None] = mod.main
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    main_fn()

    ug_file = out_dir / "oscar_ug_ipa.txt"
    kk_file = out_dir / "oscar_kk_ipa.txt"
    assert ug_file.exists() and kk_file.exists()

    def count_letters(p: Path) -> int:
        return sum(1 for ch in p.read_text(encoding="utf-8") if ch.isalpha())

    assert count_letters(ug_file) == 10
    assert count_letters(kk_file) == 10


def test_full_scan_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path("scripts/build_balanced_corpora.py")
    streams = {"ug": ["aaa"], "kk": ["aa"]}

    def stub_stream_oscar3(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar3
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa
    mod = _load_script_module(script)

    report = tmp_path / "scan.json"
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "ug,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--no-uz-assume",
        "--dry-run",
        "--report",
        str(report),
    ]
    main_fn: Callable[[], None] = mod.main
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    main_fn()
    text = report.read_text(encoding="utf-8")
    assert '"language": "kk"' in text


def test_ipa_extra_char_detection() -> None:
    # Directly exercise the IPA extra token set to cover branch
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    is_tok: Callable[[str], bool] = mod._is_ipa_token_char
    assert is_tok("\u02d0")  # ː length mark is counted
    assert is_tok("a")  # letters counted
    assert not is_tok(".")  # punctuation not counted


def test_get_streamer_invalid_raises() -> None:
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    get_streamer: Callable[[str], Callable[[str], Generator[str, None, None]]] = mod._get_streamer
    with pytest.raises(ValueError):
        get_streamer("bogus")


def test_filtered_stream_filters_out(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the predicate to always return False; expect empty stream
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))

    def _false_filter(
        *, target_lang: str, script: str | None, threshold: float, model: LangIdModel
    ) -> Callable[[str], bool]:
        return lambda _s: False

    name = "build_lang_script_filter"
    setattr(mod, name, _false_filter)

    def src_stream(_lang: str) -> Generator[str, None, None]:
        yield from ["a", "b", "c"]

    filt: Callable[
        [Callable[[str], Generator[str, None, None]], str, LangIdModel, float],
        Generator[str, None, None],
    ] = mod._filtered_stream
    out = list(filt(src_stream, "kk", _StubLID(), 0.95))
    assert out == []


def test_empty_languages_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Drive main with empty --langs to hit the guard raising RuntimeError
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    streams = {"ug": ["a"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
    ]
    main_fn: Callable[[], None] = mod.main
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    with pytest.raises(RuntimeError):
        main_fn()


def test_internal_helpers_cover_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))

    # _is_ipa_token_char: empty char -> False, break branch
    is_tok: Callable[[str], bool] = mod._is_ipa_token_char
    assert not is_tok("")

    # _get_streamer: culturax branch executes
    get_streamer: Callable[[str], Callable[[str], Generator[str, None, None]]] = mod._get_streamer
    _ = get_streamer("culturax")

    # _count_until_limit: not reached path
    def src_stream(_lang: str) -> Generator[str, None, None]:
        yield from ["a", "a", "a"]

    name2 = "build_lang_script_filter"
    setattr(mod, name2, stub_build_lang_script_filter)
    count_until_limit: Callable[
        [Callable[[str], Generator[str, None, None]], LangIdModel, float, str, int],
        tuple[bool, int],
    ] = mod._count_until_limit
    out_reached, counted = count_until_limit(src_stream, _StubLID(), 0.95, "kk", 10)
    assert out_reached is False and counted == 3

    # _truncate_line_by_ipa: early break when remaining==0
    trunc_fn: Callable[[str, int], tuple[str, int]] = mod._truncate_line_by_ipa
    trunc, used = trunc_fn("aaaa", 0)
    assert trunc == "" and used == 0
    trunc2, used2 = trunc_fn(".", 1)
    assert trunc2 == "." and used2 == 0

    # _write_balanced_corpus: target==0 triggers immediate break
    def sstream(_lang: str) -> Generator[str, None, None]:
        yield from ["aaa"]

    write_fn: Callable[
        [Callable[[str], Generator[str, None, None]], LangIdModel, float, str, Path, int],
        tuple[int, int],
    ] = mod._write_balanced_corpus
    written, lines = write_fn(sstream, _StubLID(), 0.95, "kk", Path("NUL"), 0)
    assert written == 0 and lines == 0

    # used > 0 truncation branch
    def sstream2(_lang: str) -> Generator[str, None, None]:
        yield from ["aa"]

    out_file = tmp_path / "w.txt"
    written2, lines2 = write_fn(sstream2, _StubLID(), 0.95, "kk", out_file, 1)
    assert written2 == 1 and lines2 == 1


def test_assume_path_other_language_smaller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With assume_uz_bottleneck=True and uz in languages, detect kk is smaller via count_until_limit
    script = Path("scripts/build_balanced_corpora.py")
    # uz has more content, kk has less - so kk becomes the real bottleneck
    streams = {"uz": ["aaaaa", "aaaaa"], "kk": ["a"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    mod = _load_script_module(script)
    report = tmp_path / "assume.json"
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "uz,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
        "--report",
        str(report),
    ]
    main_fn: Callable[[], None] = mod.main
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    main_fn()
    text = report.read_text(encoding="utf-8")
    assert '"language": "kk"' in text


def test_assume_path_language_reaches_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With assume_uz_bottleneck=True, kk has MORE chars than uz so reaches limit
    # This covers the return True, total branch in _count_until_limit (line 152)
    script = Path("scripts/build_balanced_corpora.py")
    # kk has more content than uz, so kk reaches uz's limit and returns (True, total)
    # uz remains the bottleneck
    streams = {"uz": ["aa"], "kk": ["aaaa", "aaaa"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    mod = _load_script_module(script)
    report = tmp_path / "assume_limit.json"
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "uz,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
        "--report",
        str(report),
    ]
    main_fn: Callable[[], None] = mod.main
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    main_fn()
    text = report.read_text(encoding="utf-8")
    # uz stays bottleneck since kk reached the limit
    assert '"language": "uz"' in text


def test_full_scan_non_decreasing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Full-scan path where second language is not smaller (covers else branch)
    script = Path("scripts/build_balanced_corpora.py")
    streams = {"ug": ["aa"], "kk": ["aaa"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa
    mod = _load_script_module(script)

    report = tmp_path / "scan2.json"
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "ug,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--no-uz-assume",
        "--dry-run",
        "--report",
        str(report),
    ]
    main_fn: Callable[[], None] = mod.main
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    main_fn()
    text = report.read_text(encoding="utf-8")
    assert '"language": "ug"' in text


def test_main_guard_executes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Execute the script as __main__ to cover the guard
    import runpy

    script = Path("scripts/build_balanced_corpora.py")
    streams = {"ug": ["a"], "kk": ["aa"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    report = tmp_path / "main.json"
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "ug,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
        "--report",
        str(report),
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    runpy.run_path(str(script), run_name="__main__")
    assert report.exists()


def test_progress_disabled_emits_no_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # With --progress-every=0 (default), no progress lines should be printed
    script = Path("scripts/build_balanced_corpora.py")
    streams = {"ug": ["a", "a", "a"], "kk": ["a", "a", "a"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    mod = _load_script_module(script)
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "ug,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
        # implicit --progress-every=0
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    mod.main()
    out = capsys.readouterr().out
    # With progress disabled, no count_all/count_until/write progress lines should appear
    assert "count_all" not in out and "count_until" not in out


def test_progress_enabled_dry_run_logs_counting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # With --progress-every, expect count_all / count_until progress lines in dry-run
    script = Path("scripts/build_balanced_corpora.py")
    streams = {"ug": ["a", "a", "a", "a", "a"], "kk": ["a", "a", "a"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    mod = _load_script_module(script)
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "ug,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
        "--progress-every",
        "2",
        "--log-format",
        "text",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    mod.main()
    out = capsys.readouterr().out
    # Rich format: "count_all <lang> <lines> lines <chars> IPA chars"
    assert "count_all" in out or "count_until" in out


def test_progress_enabled_write_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Non-dry-run should emit write progress lines
    script = Path("scripts/build_balanced_corpora.py")
    streams = {"ug": ["a", "a", "a", "a"], "kk": ["a", "a", "a", "a"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    mod = _load_script_module(script)
    out_dir = tmp_path / "out"
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "ug,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(out_dir),
        "--progress-every",
        "2",
        "--log-format",
        "text",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    mod.main()
    out = capsys.readouterr().out
    # Rich format: "write <lang> <lines> lines <chars> IPA chars"
    assert "write" in out and "lines" in out


def test_progress_enabled_count_until_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Test progress logging in _count_until_limit (line 148) with assume_uz_bottleneck
    script = Path("scripts/build_balanced_corpora.py")
    # uz has many chars (limit=100), kk has lines with 1 char each
    # Progress triggers on lines % progress_every == 0 (line 2, 4, etc.)
    # kk must NOT reach limit before line 2 to trigger progress logging
    streams = {"uz": ["a" * 100], "kk": ["a", "a", "a", "a", "a"]}

    def stub_stream_oscar(lang: str) -> Generator[str, None, None]:
        yield from streams.get(lang, [])

    cd.stream_oscar = stub_stream_oscar
    lid.load_langid_model = stub_load_langid_model
    lid.build_lang_script_filter = stub_build_lang_script_filter
    tr.to_ipa = stub_to_ipa

    mod = _load_script_module(script)
    argv = [
        "prog",
        "--source",
        "oscar",
        "--threshold",
        "0.95",
        "--langs",
        "uz,kk",
        "--data-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
        "--progress-every",
        "2",
        "--log-format",
        "text",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    mod.main()
    out = capsys.readouterr().out
    # With assume_uz_bottleneck=True and uz in langs, _count_until_limit is called for kk
    # Rich format: "count_until <lang> <lines> lines <chars> IPA chars limit=<n>"
    assert "count_until" in out


def test_progress_rich_helper_noop_when_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # _log_progress_rich should do nothing when interval is disabled (default 0)
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    log_rich_fn: Callable[[str, str, str], None] = mod._log_progress_rich
    log_rich_fn("test_phase", "kk", "extra")
    out = capsys.readouterr().out
    assert out == ""


def test_handle_interrupt_exits_with_code_130() -> None:
    # _handle_interrupt should exit with code 130 (standard for SIGINT)
    from types import FrameType

    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    handler: Callable[[int, FrameType | None], None] = mod._handle_interrupt
    with pytest.raises(SystemExit) as exc_info:
        handler(2, None)  # SIGINT = 2
    assert exc_info.value.code == 130


def test_progress_stats_elapsed_str_hours_branch() -> None:
    """Test elapsed_str returns hours format when elapsed > 1 hour."""
    import time

    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    # Set start_time to 1h 2m 5s ago
    stats.start_time = time.time() - 3725
    result = stats.elapsed_str()
    assert result == "1h 2m 5s"


def test_progress_stats_elapsed_str_minutes_branch() -> None:
    """Test elapsed_str returns minutes format when 1 min < elapsed < 1 hour."""
    import time

    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    # Set start_time to 2m 5s ago
    stats.start_time = time.time() - 125
    result = stats.elapsed_str()
    assert result == "2m 5s"


def test_progress_stats_progress_pct_no_target() -> None:
    """Test progress_pct returns 0 when target_ipa_chars is 0."""
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    stats.ipa_chars = 100
    stats.target_ipa_chars = 0
    result = stats.progress_pct()
    assert result == 0.0


def test_progress_stats_eta_str_no_target() -> None:
    """Test eta_str returns N/A when target_ipa_chars is 0."""
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    stats.target_ipa_chars = 0
    stats.ipa_chars = 100
    result = stats.eta_str()
    assert result == "N/A"


def test_progress_stats_eta_str_no_chars() -> None:
    """Test eta_str returns N/A when ipa_chars is 0."""
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    stats.target_ipa_chars = 100
    stats.ipa_chars = 0
    result = stats.eta_str()
    assert result == "N/A"


def test_progress_stats_eta_str_calculating() -> None:
    """Test eta_str returns calculating when elapsed < 1s."""
    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    stats.target_ipa_chars = 100
    stats.ipa_chars = 50
    # start_time is now, so elapsed < 1s
    result = stats.eta_str()
    assert result == "calculating..."


def test_progress_stats_eta_str_completed() -> None:
    """Test eta_str returns 0s when already at target."""
    import time

    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    stats.start_time = time.time() - 10  # 10 seconds ago
    stats.target_ipa_chars = 100
    stats.ipa_chars = 100  # Already at target
    result = stats.eta_str()
    assert result == "0s"


def test_progress_stats_eta_str_seconds() -> None:
    """Test eta_str returns seconds when ETA < 1 minute."""
    import time

    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    stats.start_time = time.time() - 10  # 10 seconds elapsed
    stats.ipa_chars = 50
    stats.target_ipa_chars = 100  # 50 chars in 10s = 5 chars/s, need 50 more = 10s ETA
    result = stats.eta_str()
    assert "s" in result
    assert "m" not in result
    assert "h" not in result


def test_progress_stats_eta_str_minutes() -> None:
    """Test eta_str returns minutes format when ETA > 1 minute."""
    import time

    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    stats.start_time = time.time() - 10  # 10 seconds elapsed
    stats.ipa_chars = 10
    stats.target_ipa_chars = 1000  # 10 chars in 10s = 1 char/s, need 990 more = 990s = 16m 30s
    result = stats.eta_str()
    assert "m" in result


def test_progress_stats_eta_str_hours() -> None:
    """Test eta_str returns hours format when ETA > 1 hour."""
    import time

    mod = _load_script_module(Path("scripts/build_balanced_corpora.py"))
    stats = _create_stats(mod)
    stats.start_time = time.time() - 10  # 10 seconds elapsed
    stats.ipa_chars = 1
    stats.target_ipa_chars = 100000  # 1 char in 10s, need 99999 more = ~1M seconds = hours
    result = stats.eta_str()
    assert "h" in result
