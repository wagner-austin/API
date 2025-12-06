"""
Build balanced IPA corpora from OSCAR or CulturaX with FastText LID >= threshold.

Pipeline:
- Stream a source (OSCAR/CulturaX) for languages: kk, ky, tr, az, ug, uz
- Filter with FastText language-ID at a confidence threshold (default 0.95)
- Transliterate each sentence to IPA
- Determine the bottleneck language by IPA character count
- Write per-language corpora truncated to the same IPA character count

Design constraints:
- Strict typing everywhere; no Any/cast/ignore/shims/stubs
- No best-effort fallbacks; errors propagate
- JSON reporting uses platform_core.json_utils

Env:
- HF_TOKEN is respected via inner loaders (recommended for OSCAR-2301)

Usage examples:
  python API/services/turkic-api/scripts/build_balanced_corpora.py \
    --source oscar --threshold 0.95 --out-dir data/balanced/oscar

  python API/services/turkic-api/scripts/build_balanced_corpora.py \
    --source culturax --threshold 0.95 --out-dir data/balanced/culturax
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import unicodedata as ud
from collections.abc import Callable, Generator
from pathlib import Path
from types import FrameType
from typing import Final, TypedDict

import psutil

# SCRIPT_DIR set in main to avoid E402
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.logging import LogFormat, LogLevel, setup_logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from turkic_api.core.corpus_download import stream_culturax, stream_oscar
from turkic_api.core.langid import LangIdModel, build_lang_script_filter, load_langid_model
from turkic_api.core.translit import to_ipa

Language = str

# Rich console for colored CLI output
_console: Console = Console()


class ProgressStats:
    """Track progress metrics during corpus processing."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.lines_kept: int = 0
        self.lines_removed: int = 0
        self.ipa_chars: int = 0
        self.target_ipa_chars: int = 0  # For progress % during write phase

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        secs = int(self.elapsed_seconds())
        mins, secs = divmod(secs, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs}h {mins}m {secs}s"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def lines_per_sec(self) -> float:
        elapsed = self.elapsed_seconds()
        if elapsed < 0.001:
            return 0.0
        return self.lines_kept / elapsed

    def chars_per_sec(self) -> float:
        elapsed = self.elapsed_seconds()
        if elapsed < 0.001:
            return 0.0
        return self.ipa_chars / elapsed

    def progress_pct(self) -> float:
        if self.target_ipa_chars <= 0:
            return 0.0
        return min(100.0, (self.ipa_chars / self.target_ipa_chars) * 100)

    def eta_str(self) -> str:
        if self.target_ipa_chars <= 0 or self.ipa_chars <= 0:
            return "N/A"
        elapsed = self.elapsed_seconds()
        if elapsed < 1.0:
            return "calculating..."
        rate = self.ipa_chars / elapsed
        remaining_chars = self.target_ipa_chars - self.ipa_chars
        if remaining_chars <= 0:
            return "0s"
        eta_secs = int(remaining_chars / rate)
        mins, secs = divmod(eta_secs, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs}h {mins}m"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def memory_mb(self) -> float:
        process: psutil.Process = psutil.Process()
        mem_info = process.memory_info()
        rss_bytes: int = mem_info.rss
        return rss_bytes / (1024 * 1024)

    def reset_for_phase(self) -> None:
        """Reset counters for a new phase while keeping start time."""
        self.lines_kept = 0
        self.lines_removed = 0
        self.ipa_chars = 0
        self.target_ipa_chars = 0
        self.start_time = time.time()


# Global progress stats instance
_stats: ProgressStats = ProgressStats()


def _handle_interrupt(signum: int, frame: FrameType | None) -> None:
    """Handle Ctrl+C gracefully with a clean exit message."""
    _console.log("\n[yellow]Interrupted by user. Exiting...[/yellow]")
    sys.exit(130)  # Standard exit code for SIGINT


def _is_ipa_token_char(ch: str) -> bool:
    """Return True if character should count toward IPA token budget.

    Counts Unicode letters (category starting with 'L') and a fixed set of IPA
    modifier symbols encoded via escapes for portability.
    """
    if not ch:
        return False
    cat = ud.category(ch)
    if cat.startswith("L"):
        return True
    ipa_extra: Final[set[str]] = {
        "\u02d0",  # ː length
        "\u02d1",  # ˑ half length
        "\u02b0",  # ʰ aspiration
        "\u02b2",  # ʲ palatalization
        "\u02b7",  # ʷ labialization
        "\u02de",  # ˞ rhoticity
        "\u02bc",  # ʼ modifier apostrophe
        "\u2019",  # ’ right single quote
        "\u0294",  # ʔ glottal stop
        "\u0295",  # ʕ pharyngeal fricative
        "\u02c8",  # ˈ primary stress
        "\u02cc",  # ˌ secondary stress
    }
    return ch in ipa_extra


def _ipa_char_count(s: str) -> int:
    return sum(1 for c in s if _is_ipa_token_char(c))


# Opt-in progress interval (lines). 0 disables progress logs.
_PROGRESS_EVERY_LINES: int = 0


def _log_progress_rich(phase: str, lang: str, extra: str = "") -> None:
    """Emit a Rich-formatted progress panel when progress is enabled."""
    if _PROGRESS_EVERY_LINES <= 0:
        return

    # Build progress info table
    stats_table = Table.grid(padding=(0, 2))
    stats_table.add_column(style="dim", justify="right")
    stats_table.add_column(style="bold")

    # Core metrics
    stats_table.add_row("Phase:", f"[magenta]{phase}[/magenta]")
    stats_table.add_row("Language:", f"[cyan]{lang}[/cyan]")
    stats_table.add_row("Elapsed:", f"[white]{_stats.elapsed_str()}[/white]")
    stats_table.add_row("Memory:", f"[white]{_stats.memory_mb():.1f} MB[/white]")

    # Line stats
    total_seen = _stats.lines_kept + _stats.lines_removed
    keep_pct = (_stats.lines_kept / total_seen * 100) if total_seen > 0 else 0
    stats_table.add_row(
        "Lines:",
        f"[green]{_stats.lines_kept:,}[/green] kept / "
        f"[red]{_stats.lines_removed:,}[/red] removed "
        f"([yellow]{keep_pct:.1f}%[/yellow] kept)",
    )

    # IPA stats
    stats_table.add_row("IPA Characters:", f"[green]{_stats.ipa_chars:,}[/green]")

    # Throughput
    stats_table.add_row(
        "Throughput:",
        f"[white]{_stats.lines_per_sec():.1f}[/white] lines/s, "
        f"[white]{_stats.chars_per_sec():,.0f}[/white] chars/s",
    )

    # Progress % and ETA (only during write phase with target)
    if _stats.target_ipa_chars > 0:
        stats_table.add_row(
            "Progress:",
            f"[yellow]{_stats.progress_pct():.1f}%[/yellow] "
            f"({_stats.ipa_chars:,} / {_stats.target_ipa_chars:,})",
        )
        stats_table.add_row("ETA:", f"[cyan]{_stats.eta_str()}[/cyan]")

    # Extra info
    if extra:
        stats_table.add_row("Info:", f"[dim]{extra}[/dim]")

    panel = Panel(stats_table, title="[bold]Progress[/bold]", border_style="blue", expand=False)
    _console.log(panel)


def _get_streamer(source: str) -> Callable[[Language], Generator[str, None, None]]:
    if source == "oscar":
        return lambda lang: stream_oscar(lang)
    if source == "culturax":
        return lambda lang: stream_culturax(lang)
    raise ValueError(f"Unsupported source: {source}")


def _filtered_stream(
    source_stream: Callable[[Language], Generator[str, None, None]],
    lang: Language,
    lid: LangIdModel,
    threshold: float,
    script: str | None = None,
) -> Generator[str, None, None]:
    keep = build_lang_script_filter(target_lang=lang, script=script, threshold=threshold, model=lid)
    for s in source_stream(lang):
        if keep(s):
            _stats.lines_kept += 1
            yield s
        else:
            _stats.lines_removed += 1


def _full_ipa_char_count(
    source_stream: Callable[[Language], Generator[str, None, None]],
    lid: LangIdModel,
    threshold: float,
    lang: Language,
    script: str | None = None,
) -> int:
    _stats.reset_for_phase()
    for line in _filtered_stream(source_stream, lang, lid, threshold, script):
        ipa = to_ipa(line, lang)
        _stats.ipa_chars += _ipa_char_count(ipa)
        if _PROGRESS_EVERY_LINES > 0 and (_stats.lines_kept % _PROGRESS_EVERY_LINES == 0):
            _log_progress_rich("count_all", lang)
    return _stats.ipa_chars


def _count_until_limit(
    source_stream: Callable[[Language], Generator[str, None, None]],
    lid: LangIdModel,
    threshold: float,
    lang: Language,
    limit: int,
    script: str | None = None,
) -> tuple[bool, int]:
    _stats.reset_for_phase()
    _stats.target_ipa_chars = limit
    for line in _filtered_stream(source_stream, lang, lid, threshold, script):
        ipa = to_ipa(line, lang)
        _stats.ipa_chars += _ipa_char_count(ipa)
        if _PROGRESS_EVERY_LINES > 0 and (_stats.lines_kept % _PROGRESS_EVERY_LINES == 0):
            _log_progress_rich("count_until", lang, f"target={limit:,}")
        if _stats.ipa_chars >= limit:
            return True, _stats.ipa_chars
    return False, _stats.ipa_chars


def _truncate_line_by_ipa(ipa_line: str, remaining: int) -> tuple[str, int]:
    consumed = 0
    out_chars: list[str] = []
    for ch in ipa_line:
        if consumed >= remaining:
            break
        out_chars.append(ch)
        if _is_ipa_token_char(ch):
            consumed += 1
    return ("".join(out_chars), consumed)


def _write_balanced_corpus(
    source_stream: Callable[[Language], Generator[str, None, None]],
    lid: LangIdModel,
    threshold: float,
    lang: Language,
    out_path: Path,
    target_ipa_chars: int,
    script: str | None = None,
) -> tuple[int, int]:
    _stats.reset_for_phase()
    _stats.target_ipa_chars = target_ipa_chars
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for line in _filtered_stream(source_stream, lang, lid, threshold, script):
            if _stats.ipa_chars >= target_ipa_chars:
                break
            ipa = to_ipa(line, lang)
            line_ipa_count = _ipa_char_count(ipa)
            remaining = target_ipa_chars - _stats.ipa_chars
            if line_ipa_count <= remaining:
                fh.write(ipa + "\n")
                _stats.ipa_chars += line_ipa_count
                lines_written += 1
                if _PROGRESS_EVERY_LINES > 0 and (lines_written % _PROGRESS_EVERY_LINES == 0):
                    _log_progress_rich("write", lang)
            else:
                truncated, used = _truncate_line_by_ipa(ipa, remaining)
                fh.write(truncated + "\n")
                lines_written += 1
                _stats.ipa_chars += used
                break
    return _stats.ipa_chars, lines_written


class BuildResult(TypedDict):
    source: str
    languages: list[Language]
    threshold: float
    bottleneck_lang: Language
    bottleneck_chars: int
    actual_written: dict[Language, int]
    lines_written: dict[Language, int]


def _script_for_lang(lang_code: Language, prefer_218e: bool) -> str | None:
    # Enforce Latin for Azerbaijani when using 218e (labels include script);
    # with lid.176 labels carry no script, so return None.
    return "Latn" if prefer_218e and lang_code == "az" else None


def _determine_bottleneck(
    languages: list[Language],
    streamer: Callable[[Language], Generator[str, None, None]],
    lid: LangIdModel,
    threshold: float,
    assume_uz_bottleneck: bool,
    prefer_218e: bool,
) -> tuple[Language, int]:
    bottleneck_lang: Language | None = None
    bottleneck_chars = 0
    if assume_uz_bottleneck and "uz" in languages:
        bottleneck_lang = "uz"
        bottleneck_chars = _full_ipa_char_count(
            streamer, lid, threshold, "uz", _script_for_lang("uz", prefer_218e)
        )
        for lang in languages:
            if lang == "uz":
                continue
            reached, counted = _count_until_limit(
                streamer,
                lid,
                threshold,
                lang,
                bottleneck_chars,
                _script_for_lang(lang, prefer_218e),
            )
            if not reached and counted < bottleneck_chars:
                bottleneck_lang = lang
                bottleneck_chars = counted
    else:
        for lang in languages:
            cnt = _full_ipa_char_count(
                streamer, lid, threshold, lang, _script_for_lang(lang, prefer_218e)
            )
            if bottleneck_lang is None or cnt < bottleneck_chars:
                bottleneck_lang = lang
                bottleneck_chars = cnt
    if bottleneck_lang is None:
        raise RuntimeError("Failed to determine bottleneck language")
    return bottleneck_lang, bottleneck_chars


def build_balanced(
    *,
    source: str,
    data_dir: Path,
    out_dir: Path,
    threshold: float = 0.95,
    languages: list[Language] | tuple[Language, ...] = ("kk", "ky", "tr", "az", "ug", "uz", "fi"),
    prefer_218e: bool = True,
    assume_uz_bottleneck: bool = True,
    dry_run: bool = False,
) -> BuildResult:
    languages = list(languages)
    out_dir.mkdir(parents=True, exist_ok=True)

    lid = load_langid_model(str(data_dir), prefer_218e=prefer_218e)
    streamer = _get_streamer(source)

    bottleneck_lang, bottleneck_chars = _determine_bottleneck(
        languages, streamer, lid, threshold, assume_uz_bottleneck, prefer_218e
    )

    actual_written: dict[Language, int] = {}
    lines_written: dict[Language, int] = {}
    if not dry_run:
        for lang in languages:
            out_path = out_dir / f"{source}_{lang}_ipa.txt"
            actual, nlines = _write_balanced_corpus(
                streamer,
                lid,
                threshold,
                lang,
                out_path,
                bottleneck_chars,
                _script_for_lang(lang, prefer_218e),
            )
            actual_written[lang] = actual
            lines_written[lang] = nlines

    return {
        "source": source,
        "languages": languages,
        "threshold": threshold,
        "bottleneck_lang": bottleneck_lang,
        "bottleneck_chars": bottleneck_chars,
        "actual_written": actual_written,
        "lines_written": lines_written,
    }


def main() -> None:
    # Register graceful interrupt handler for Ctrl+C
    signal.signal(signal.SIGINT, _handle_interrupt)

    script_dir: Final[Path] = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build balanced IPA corpora with LID filtering")
    parser.add_argument("--source", choices=["oscar", "culturax"], required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument(
        "--langs",
        type=str,
        default="kk,ky,tr,az,ug,uz,fi",
        help="Comma-separated language codes",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(script_dir.parent / "data"),
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Emit progress every N lines processed (0 disables)",
    )
    parser.add_argument(
        "--no-uz-assume",
        action="store_true",
        help="Do not assume uz is bottleneck; full count all",
    )
    parser.add_argument("--prefer-218e", action="store_true", default=True)
    parser.add_argument("--prefer-176", dest="prefer_218e", action="store_false")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute bottleneck only; do not write corpora",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write JSON report (platform_core.json_utils)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Logging output format",
    )
    args = parser.parse_args()

    # Extract typed CLI values (avoid Any propagation in mypy)
    langs_arg: str = args.langs
    source: str = args.source
    data_dir_arg: str = args.data_dir
    out_dir_arg: str = args.out_dir
    threshold_val: float = args.threshold
    prefer218e_val: bool = args.prefer_218e
    no_uz_assume_val: bool = args.no_uz_assume
    dry_run_val: bool = args.dry_run
    report_arg: str | None = args.report
    progress_every_val: int = args.progress_every
    log_level_arg: str = args.log_level
    log_format_arg: str = args.log_format

    # Normalize log level and format to typed literals
    _levels: Final[dict[str, LogLevel]] = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }
    _formats: Final[dict[str, LogFormat]] = {
        "text": "text",
        "json": "json",
    }
    log_level_val: LogLevel = _levels[log_level_arg]
    log_format_val: LogFormat = _formats[log_format_arg]

    # Initialize centralized logging for CLI usage
    setup_logging(
        level=log_level_val,
        format_mode=log_format_val,
        service_name="turkic-cli",
        instance_id=None,
        extra_fields=[],
    )

    langs: list[Language] = [x.strip() for x in langs_arg.split(",") if x.strip()]
    # Configure progress interval for internal helpers (0 disables)
    global _PROGRESS_EVERY_LINES
    _PROGRESS_EVERY_LINES = progress_every_val if progress_every_val > 0 else 0

    res = build_balanced(
        source=source,
        data_dir=Path(data_dir_arg),
        out_dir=Path(out_dir_arg),
        threshold=threshold_val,
        languages=langs,
        prefer_218e=prefer218e_val,
        assume_uz_bottleneck=not no_uz_assume_val,
        dry_run=dry_run_val,
    )

    header: Final[str] = (
        "[yellow]Dry-run (no files written)[/yellow]"
        if dry_run_val
        else "[green]Balanced IPA corpora built[/green]"
    )
    _console.log(header)
    _console.log(
        f"[cyan]source[/cyan]={res['source']} "
        f"[cyan]threshold[/cyan]={res['threshold']} "
        f"[cyan]langs[/cyan]={','.join(res['languages'])}"
    )
    _console.log(
        f"[magenta]bottleneck[/magenta]={res['bottleneck_lang']} "
        f"[magenta]chars[/magenta]={res['bottleneck_chars']:,}"
    )
    if not dry_run_val:
        table = Table(title="Results")
        table.add_column("Language", style="cyan")
        table.add_column("IPA Chars", style="green", justify="right")
        table.add_column("Lines", style="yellow", justify="right")
        for lang in res["languages"]:
            actual = res["actual_written"].get(lang, 0)
            nlines = res["lines_written"].get(lang, 0)
            table.add_row(lang, f"{actual:,}", f"{nlines:,}")
        _console.log(table)

    if report_arg is not None:
        report_path = Path(report_arg)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Build structured per-language results
        per_lang_results: dict[str, JSONValue] = {}
        for lang in res["languages"]:
            lang_data: dict[str, JSONValue] = {
                "ipa_characters": res["actual_written"].get(lang, 0),
                "lines_written": res["lines_written"].get(lang, 0),
                "output_file": f"{res['source']}_{lang}_ipa.txt",
            }
            per_lang_results[lang] = lang_data

        # Build structured manifest
        payload: dict[str, JSONValue] = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version": "2.0",
                "dry_run": dry_run_val,
            },
            "config": {
                "source": res["source"],
                "threshold": res["threshold"],
                "languages": list(res["languages"]),
                "output_directory": str(Path(out_dir_arg).resolve()),
                "lid_model": "lid218e" if prefer218e_val else "lid176",
                "assume_uz_bottleneck": not no_uz_assume_val,
            },
            "bottleneck": {
                "language": res["bottleneck_lang"],
                "ipa_characters": res["bottleneck_chars"],
            },
            "results": per_lang_results,
            "summary": {
                "total_languages": len(res["languages"]),
                "total_ipa_characters": sum(res["actual_written"].values()),
                "total_lines": sum(res["lines_written"].values()),
            },
        }

        report_path.write_text(dump_json_str(payload, indent=2), encoding="utf-8")
        _console.log(f"[green]report[/green]={report_path}")

    # Reset progress interval to avoid leaking between runs/tests
    _PROGRESS_EVERY_LINES = 0


if __name__ == "__main__":
    main()
