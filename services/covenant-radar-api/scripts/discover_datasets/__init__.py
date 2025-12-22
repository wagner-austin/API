"""Dataset discovery script for auto-generating DatasetConfig entries.

Scans data/external/ and creates preliminary configurations for the registry.
"""

from scripts.discover_datasets.main import main

__all__ = ["main"]
