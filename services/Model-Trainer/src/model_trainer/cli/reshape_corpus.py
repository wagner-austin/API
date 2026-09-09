"""Write a reshaped copy of a corpus, and report what the reshaping cost.

The benchmark already takes ``--corpus <dir>``, so the cleanest way to measure
a second representation is to MATERIALISE it as a second corpus directory and
run the unchanged benchmark against it. No arm learns that two corpora exist,
which is what keeps the two runs comparable in everything except the text.

THE OUTPUT CARRIES NO FRONTMATTER, deliberately. The corpus reader returns a
document without a fence unchanged, and the reshaped body is prose only -- so
adding YAML back would be adding text the source's body never had, which is
exactly the invention this reshaping is built to avoid.

THE DIGEST IS PRINTED, and it is the point of the run. Two corpora with two
digests are two question sets, and every record downstream carries its own
payload digest to prove which one it answered.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger, setup_logging

from model_trainer.cli import _test_hooks
from model_trainer.core.services.model.cartridge_plans import corpus_digest
from model_trainer.core.services.model.corpus_reshape import reshape_corpus

_log = get_logger(__name__)

CORPUS_FLAG = "--corpus"
OUT_FLAG = "--out"

_FLAGS = (CORPUS_FLAG, OUT_FLAG)

#: What each written document is named. Numbered by position rather than by
#: the source filename, because the reader sorts by filename and a reshaped
#: corpus has to preserve the ORDER the source was read in -- that order
#: decides which windows the stride holds out.
DOCUMENT_STEM = "doc"


def write_reshaped_corpus(*, corpus: pathlib.Path, out: pathlib.Path) -> str:
    """Reshape a corpus and write it as a second corpus directory.

    Args:
        corpus: Directory of markdown documents to read.
        out: Directory to write the reshaped documents into. Created if
            absent.

    Returns:
        The reshaped corpus's digest.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when any document fails
            the reshape gate. Refused rather than skipped: a corpus missing
            one page is a different corpus, and a run against it would report
            a number for a text nobody chose.
    """
    documents = _test_hooks.read_corpus_documents(corpus)
    report = reshape_corpus(documents)
    if report["rejected"]:
        failures = [
            f"document {index}: {', '.join(entry['failed_clauses'])}"
            for index, entry in enumerate(report["documents"])
            if entry["failed_clauses"]
        ]
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE,
            (
                f"{report['rejected']} of {len(documents)} documents did not survive "
                f"reshaping faithfully: {'; '.join(failures)}"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE),
        )

    out.mkdir(parents=True, exist_ok=True)
    reshaped = tuple(entry["reshaped"] for entry in report["documents"])
    for index, body in enumerate(reshaped):
        # Zero-padded so the reader's filename sort reproduces the read order.
        (out / f"{DOCUMENT_STEM}{index:03d}.md").write_text(body, encoding="utf-8")

    digest = corpus_digest(reshaped)
    _log.info(
        "reshaped %d documents -> %s; %d terms lost corpus-wide; digest %s",
        len(reshaped),
        out,
        len(report["terms_lost"]),
        digest[:12],
    )
    if report["terms_lost"]:
        _log.info("terms present only in the scaffolding: %s", ", ".join(report["terms_lost"]))
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    """Write a reshaped copy of one corpus.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the corpus is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent.
        AppError: Propagated from the reader or the gate.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    write_reshaped_corpus(
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        out=pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG)),
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="reshape-corpus",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "DOCUMENT_STEM",
    "entrypoint",
    "main",
    "write_reshaped_corpus",
]


# Without this, `python -m model_trainer.cli.reshape_corpus` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
