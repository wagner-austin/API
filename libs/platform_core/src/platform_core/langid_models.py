"""Which fastText language-id model file a request means, and where it comes from.

platform_stt and turkic-api both download the same two fastText LID models into
the same ``models/`` subdirectory, and each carried its own copy of the
directory name, the two file names and the two URLs. turkic-api carried the set
TWICE -- once in its domain module and once in its hook default -- so a URL that
moved upstream would have had to be corrected in three places, and nothing in
the tree would have noticed the one that was missed.

Deciding WHICH file is wanted is shared knowledge and lives here. FETCHING it is
not: each package downloads through its own injected hook, so the download stays
at that package's own test seam rather than being pulled behind a second one.
"""

from __future__ import annotations

from pathlib import Path

from typing_extensions import TypedDict

MODEL_DIRNAME = "models"
"""Subdirectory of a data directory that holds downloaded model files."""

LID_218E_FILENAME = "lid218e.bin"
"""File name of the NLLB LID-218e model."""

LID_218E_URL = "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin"
"""Upstream location of the NLLB LID-218e model."""

LID_176_FILENAME = "lid.176.bin"
"""File name of the fastText lid.176 model."""

LID_176_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
"""Upstream location of the fastText lid.176 model."""


class LangIdModelFile(TypedDict):
    """Where a language-id model belongs on disk and where it is fetched from.

    Attributes:
        path: The file's location under ``<data_dir>/models``.
        url: The upstream URL to download it from when ``path`` is absent.
    """

    path: Path
    url: str


def langid_model_file(data_dir: str, *, prefer_218e: bool) -> LangIdModelFile:
    """Name the model file a caller is asking for.

    Args:
        data_dir: Base directory the ``models`` subdirectory sits under.
        prefer_218e: True selects NLLB LID-218e, False selects fastText lid.176.

    Returns:
        The file's path and its upstream URL. Nothing is touched on disk: the
        caller downloads through its own hook when the path turns out absent.
    """
    base = Path(data_dir) / MODEL_DIRNAME
    if prefer_218e:
        return {"path": base / LID_218E_FILENAME, "url": LID_218E_URL}
    return {"path": base / LID_176_FILENAME, "url": LID_176_URL}


__all__ = [
    "LID_176_FILENAME",
    "LID_176_URL",
    "LID_218E_FILENAME",
    "LID_218E_URL",
    "MODEL_DIRNAME",
    "LangIdModelFile",
    "langid_model_file",
]
