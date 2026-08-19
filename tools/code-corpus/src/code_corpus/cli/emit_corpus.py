"""CLI: tracked source files from named repositories to one training corpus.

Arguments are parsed by hand rather than with argparse because argparse's
namespace is untyped attribute access, and this package holds every
expression to a known type. The surface is small enough that the hand parser
is the simpler artifact: repeatable ``--repo name=path`` and ``--language``
flags, single-valued ``--out``, ``--holdout-out``, ``--holdout-fraction`` and
``--seed``, and nothing else. An unknown or malformed argument raises; a
corpus emitted under a mistyped flag would be a silently different corpus.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.json_utils import dump_json_str

from code_corpus.cli import _test_hooks
from code_corpus.contracts.corpus import RepoPin, encode_code_corpus_manifest
from code_corpus.core import _test_hooks as core_hooks
from code_corpus.core.emit import build_manifest, dedup_files, jsonl_text, split_holdout, to_record
from code_corpus.core.select import SelectedFile, repo_pin, select_files

DEFAULT_HOLDOUT_FRACTION = 0.1
DEFAULT_SEED = 0
DEFAULT_LANGUAGES = ("python",)

_REPEATABLE_FLAGS = ("--repo", "--language")
_SINGLE_FLAGS = ("--out", "--holdout-out", "--holdout-fraction", "--seed")


class EmitArgs:
    """Parsed arguments describing one emission.

    Attributes:
        repos: Repository names and roots, in command-line order. The order
            matters: deduplication keeps the first occurrence of identical
            content, so earlier repositories represent shared files.
        out: Training JSONL to write.
        holdout_out: Holdout JSONL to write.
        holdout_fraction: Fraction of selected files to hold out, by file.
        seed: Seed for the holdout sample and the train-order shuffle.
        languages: Languages to select, in command-line order.
    """

    __slots__ = ("holdout_fraction", "holdout_out", "languages", "out", "repos", "seed")

    def __init__(
        self,
        *,
        repos: Sequence[tuple[str, pathlib.Path]],
        out: pathlib.Path,
        holdout_out: pathlib.Path,
        holdout_fraction: float,
        seed: int,
        languages: Sequence[str],
    ) -> None:
        """Initialise parsed arguments.

        Args:
            repos: Repository names and roots, in command-line order.
            out: Training JSONL to write.
            holdout_out: Holdout JSONL to write.
            holdout_fraction: Fraction of selected files to hold out.
            seed: Seed for the sample and the shuffle.
            languages: Languages to select.
        """
        self.repos = tuple(repos)
        self.out = out
        self.holdout_out = holdout_out
        self.holdout_fraction = holdout_fraction
        self.seed = seed
        self.languages = tuple(languages)


def _take_value(tokens: Sequence[str], index: int, flag: str) -> str:
    """Read the value following a flag.

    Args:
        tokens: The full argument list.
        index: Index of the flag.
        flag: The flag, for the error message.

    Returns:
        The value token.

    Raises:
        ValueError: If the flag is the last token.
    """
    if index + 1 >= len(tokens):
        raise ValueError(f"{flag} requires a value")
    return tokens[index + 1]


def _parse_repo(spec: str) -> tuple[str, pathlib.Path]:
    """Parse one ``--repo name=path`` specification.

    Args:
        spec: The flag's value.

    Returns:
        The repository name and root.

    Raises:
        ValueError: If the specification has no ``=``, an empty name or
            path, or a slash in the name -- names prefix document paths, so
            a slash would make them ambiguous.
    """
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"--repo expects name=path, got '{spec}'")
    name, path_text = parts
    if name == "" or path_text == "":
        raise ValueError(f"--repo expects a non-empty name and path, got '{spec}'")
    if "/" in name or "\\" in name:
        raise ValueError(f"--repo name must not contain a slash, got '{name}'")
    return name, pathlib.Path(path_text)


def _int_flag(flag: str, raw: str) -> int:
    """Parse a flag's value as an integer.

    Args:
        flag: The flag, for the error message.
        raw: The value token.

    Returns:
        The parsed integer.

    Raises:
        ValueError: If the value is not an integer, naming the flag.
    """
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{flag} expects an integer, got '{raw}'") from exc


def _float_flag(flag: str, raw: str) -> float:
    """Parse a flag's value as a float.

    Args:
        flag: The flag, for the error message.
        raw: The value token.

    Returns:
        The parsed float.

    Raises:
        ValueError: If the value is not a number, naming the flag.
    """
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{flag} expects a number, got '{raw}'") from exc


def _collect_flags(argv: Sequence[str]) -> tuple[list[str], list[str], dict[str, str]]:
    """Walk the argument list into raw flag values.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        Raw ``--repo`` values in order, raw ``--language`` values in order,
        and the single-valued flags by name.

    Raises:
        ValueError: If a flag is unknown, misses its value, or a
            single-valued flag repeats.
    """
    repo_specs: list[str] = []
    language_values: list[str] = []
    single: dict[str, str] = {}
    index = 0
    while index < len(argv):
        flag = argv[index]
        known = flag in (*_REPEATABLE_FLAGS, *_SINGLE_FLAGS)
        value = _take_value(argv, index, flag) if known else ""
        if flag == "--repo":
            repo_specs.append(value)
        elif flag == "--language":
            language_values.append(value)
        elif flag in _SINGLE_FLAGS:
            if flag in single:
                raise ValueError(f"duplicate argument '{flag}'")
            single[flag] = value
        else:
            raise ValueError(f"unknown argument '{flag}'")
        index += 2
    return repo_specs, language_values, single


def _parse_repos(repo_specs: Sequence[str]) -> list[tuple[str, pathlib.Path]]:
    """Parse and validate every repository specification.

    Args:
        repo_specs: Raw ``--repo`` values in order.

    Returns:
        Repository names and roots, in order.

    Raises:
        ValueError: If no repository was given, a specification is
            malformed, or a name repeats.
    """
    if repo_specs == []:
        raise ValueError("--repo is required (at least one name=path)")
    repos = [_parse_repo(spec) for spec in repo_specs]
    names = [name for name, _ in repos]
    if len(set(names)) != len(names):
        raise ValueError(f"--repo names must be unique, got {names}")
    return repos


def _parse_languages(language_values: Sequence[str]) -> tuple[str, ...]:
    """Validate the requested languages, defaulting when none were given.

    Args:
        language_values: Raw ``--language`` values in order.

    Returns:
        The languages to select.

    Raises:
        ValueError: If a language repeats. Whether a name is a known
            language is checked at selection, which owns the language table.
    """
    if language_values == []:
        return DEFAULT_LANGUAGES
    if len(set(language_values)) != len(language_values):
        raise ValueError(f"--language values must be unique, got {list(language_values)}")
    return tuple(language_values)


def parse_args(argv: Sequence[str]) -> EmitArgs:
    """Parse command-line arguments.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        Parsed arguments describing one emission.

    Raises:
        ValueError: If an argument is unknown, malformed, missing, or
            repeated where it must not be.
    """
    repo_specs, language_values, single = _collect_flags(argv)
    repos = _parse_repos(repo_specs)
    if "--out" not in single:
        raise ValueError("--out is required")
    out = pathlib.Path(single["--out"])
    holdout_out = (
        pathlib.Path(single["--holdout-out"])
        if "--holdout-out" in single
        else out.with_suffix(".holdout" + out.suffix)
    )
    fraction = (
        _float_flag("--holdout-fraction", single["--holdout-fraction"])
        if "--holdout-fraction" in single
        else DEFAULT_HOLDOUT_FRACTION
    )
    seed = _int_flag("--seed", single["--seed"]) if "--seed" in single else DEFAULT_SEED
    return EmitArgs(
        repos=repos,
        out=out,
        holdout_out=holdout_out,
        holdout_fraction=fraction,
        seed=seed,
        languages=_parse_languages(language_values),
    )


def _emit_summary(
    *,
    pins: Sequence[RepoPin],
    train: Sequence[SelectedFile],
    holdout: Sequence[SelectedFile],
    excluded_generated: int,
    excluded_duplicate: int,
    excluded_empty: int,
    args: EmitArgs,
    manifest_name: str,
) -> None:
    """Report what an emission did.

    Args:
        pins: State of every contributing repository.
        train: Files written to the training output.
        holdout: Files written to the holdout output.
        excluded_generated: Files refused as generator output.
        excluded_duplicate: Files dropped as duplicates.
        excluded_empty: Files refused as whitespace-only.
        args: The parsed arguments.
        manifest_name: Filename of the written manifest.
    """
    for pin in pins:
        dirty_note = " (dirty)" if pin["dirty"] else ""
        _test_hooks.emit(f"repo             {pin['name']} @ {pin['commit']}{dirty_note}")
    _test_hooks.emit(
        f"files            {len(train) + len(holdout):,} kept "
        f"({len(train):,} train, {len(holdout):,} holdout)"
    )
    _test_hooks.emit(
        f"excluded         generated {excluded_generated:,}, "
        f"duplicate {excluded_duplicate:,}, empty {excluded_empty:,}"
    )
    tokens_train = sum(file.tokens_approx for file in train)
    tokens_holdout = sum(file.tokens_approx for file in holdout)
    _test_hooks.emit(f"tokens approx    {tokens_train:,} train, {tokens_holdout:,} holdout")
    _test_hooks.emit(f"written          {args.out}")
    _test_hooks.emit(f"holdout          {args.holdout_out}")
    _test_hooks.emit(f"manifest         {manifest_name}")


def main(argv: Sequence[str]) -> int:
    """Emit one corpus and the manifest describing it.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        Process exit code, 0 on success.

    Raises:
        ValueError: If an argument is invalid, a language is unknown, a
            tracked source file is not valid UTF-8, or the selection or
            split comes up empty.
        subprocess.CalledProcessError: If a repository cannot be pinned.
    """
    args = parse_args(argv)

    pins: list[RepoPin] = []
    combined: list[SelectedFile] = []
    excluded_generated = 0
    excluded_empty = 0
    for name, root in args.repos:
        pins.append(repo_pin(name, root))
        outcome = select_files(name, root, args.languages)
        combined.extend(outcome.files)
        excluded_generated += outcome.excluded_generated
        excluded_empty += outcome.excluded_empty

    kept, excluded_duplicate = dedup_files(combined)
    train, holdout = split_holdout(kept, fraction=args.holdout_fraction, seed=args.seed)

    core_hooks.write_text(args.out, jsonl_text([to_record(file) for file in train]))
    core_hooks.write_text(args.holdout_out, jsonl_text([to_record(file) for file in holdout]))

    manifest = build_manifest(
        train_output=args.out,
        holdout_output=args.holdout_out,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        repos=pins,
        train=train,
        holdout=holdout,
        excluded_generated=excluded_generated,
        excluded_duplicate=excluded_duplicate,
        excluded_empty=excluded_empty,
    )
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    core_hooks.write_text(
        manifest_path, dump_json_str(encode_code_corpus_manifest(manifest), indent=2) + "\n"
    )

    _emit_summary(
        pins=pins,
        train=train,
        holdout=holdout,
        excluded_generated=excluded_generated,
        excluded_duplicate=excluded_duplicate,
        excluded_empty=excluded_empty,
        args=args,
        manifest_name=manifest_path.name,
    )
    return 0


def entrypoint() -> int:
    """Console-script entrypoint reading the process arguments.

    Returns:
        Process exit code.
    """
    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(entrypoint())
