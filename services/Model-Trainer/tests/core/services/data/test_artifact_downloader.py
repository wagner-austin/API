from __future__ import annotations

from pathlib import Path

import pytest
from platform_ml.artifact_store import ArtifactStore, ArtifactStoreError


def test_store_download_invalid_file_id(tmp_path: Path) -> None:
    s = ArtifactStore(base_url="http://x", api_key="k")
    with pytest.raises(ArtifactStoreError):
        s.download_artifact(" ", dest_dir=tmp_path, request_id="r", expected_root="root")
