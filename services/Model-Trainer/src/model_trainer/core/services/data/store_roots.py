"""Where a service run's corpus and artifact stores live.

A service run's stores are the filesystem stores a cluster run already used
(:mod:`model_trainer.cluster.stores`), bound to directories under the
settings' roots. These are the one place those directories are named. A
cluster run binds its own, from the flags it was given.
"""

from __future__ import annotations

from pathlib import Path

from model_trainer.core.config.settings import Settings


def corpus_root(settings: Settings) -> Path:
    """The directory staged corpora are read from, each named by its digest.

    ``corpus`` under the data root is the directory the service's compose
    file already mounts read-only from the host, so a corpus staged there is
    visible to the API and the worker alike and nothing in the container can
    alter it.

    Args:
        settings: The run's settings.

    Returns:
        ``corpus`` under the data root.
    """
    return Path(settings["app"]["data_root"]) / "corpus"


def artifact_store_root(settings: Settings) -> Path:
    """The directory stored artifact tarballs are written to and read from.

    Args:
        settings: The run's settings.

    Returns:
        ``store`` under the artifacts root.
    """
    return Path(settings["app"]["artifacts_root"]) / "store"


__all__ = ["artifact_store_root", "corpus_root"]
