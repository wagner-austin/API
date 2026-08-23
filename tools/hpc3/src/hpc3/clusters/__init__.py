"""The clusters this package has been measured against.

Adding one is deliberately a code change, not a configuration change. The
facts in a cluster module are what turn every rule in this package from advice
into a refusal, so a machine nobody has measured is a machine this package
cannot honestly enforce anything about. Committing a module is the act of
saying "these numbers were read off the real thing".

To add a cluster: measure it (``sinfo``, ``scontrol show partition``,
``sacctmgr show qos``), write a module beside :mod:`hpc3.clusters.hpc3` naming
the source and date, and register it below. The architecture test checks the
shape; only measurement can check the values.
"""

from __future__ import annotations

from collections.abc import Mapping

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.cluster import ClusterFacts

CLUSTERS: Mapping[str, ClusterFacts] = {HPC3["slug"]: HPC3}
"""Every measured cluster, keyed by the slug a workspace selects it with.

Keyed off each module's own ``slug`` rather than a string repeated here, so
the registry key and the facts cannot disagree about what a cluster is called.
"""


def require_cluster(slug: str) -> ClusterFacts:
    """Look up a cluster by the slug a workspace named.

    Args:
        slug: Cluster slug from the workspace document.

    Returns:
        That cluster's measured facts.

    Raises:
        AppError: With ``CLUSTER_UNKNOWN`` if no module has been measured for
            it. The message lists what is available, because the alternative
            -- guessing a default -- would submit to one machine using another
            machine's ceilings.
    """
    facts = CLUSTERS.get(slug)
    if facts is None:
        raise AppError(
            Hpc3ErrorCode.CLUSTER_UNKNOWN,
            f"No cluster {slug!r} has been measured; known clusters are {sorted(CLUSTERS)}. "
            "Add a module under hpc3/clusters/ with facts read off the real machine.",
        )
    return facts


__all__ = ["CLUSTERS", "require_cluster"]
