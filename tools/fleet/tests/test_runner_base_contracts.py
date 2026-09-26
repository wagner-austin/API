"""The runner-host base contract: every decode refusal, and the round trip."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.runner_base import (
    decode_disk_ceiling,
    decode_host_base,
    decode_pinned_download,
    encode_host_base,
)
from tests._runner_fixtures import a_base, base_json


def _base(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw base, with fields overridden per test.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw = base_json()
    raw.update(overrides)
    return raw


def _pin(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw pinned download.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "version": "2.7.14",
        "url": "https://example.invalid/wsl.msi",
        "sha256": "ab" * 32,
    }
    raw.update(overrides)
    return raw


def _disk(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw disk ceiling.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "ceiling_gb": 150,
        "baseline_gb": 46,
        "baseline_measured": "2026-09-26",
    }
    raw.update(overrides)
    return raw


class TestHostBase:
    """decode_host_base's contract."""

    def test_the_fixture_decodes_to_the_typed_fixture(self) -> None:
        assert decode_host_base(base_json()) == a_base()

    def test_encode_then_decode_is_the_identity(self) -> None:
        assert decode_host_base(dict(encode_host_base(a_base()))) == a_base()

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_host_base([])

    @pytest.mark.parametrize("path", ["C:\\wsl\\Ubuntu", "/opt/wsl", "wsl", "C:"])
    def test_a_distro_dir_that_is_not_a_forward_slashed_drive_path_is_refused(
        self, path: str
    ) -> None:
        with pytest.raises(JSONTypeError, match="forward-slashed drive-letter"):
            decode_host_base(_base(distro_dir=path))

    def test_no_packages_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="cannot run docker"):
            decode_host_base(_base(apt_packages=[]))

    @pytest.mark.parametrize("policy", ["Restricted", "Unrestricted", "remotesigned", ""])
    def test_a_policy_outside_the_allowed_pair_is_refused(self, policy: str) -> None:
        with pytest.raises(JSONTypeError, match="execution_policy must be one of"):
            decode_host_base(_base(execution_policy=policy))

    def test_a_missing_download_block_is_refused(self) -> None:
        raw = _base()
        del raw["rootfs"]
        with pytest.raises(JSONTypeError, match="rootfs"):
            decode_host_base(raw)


class TestPinnedDownload:
    """decode_pinned_download's contract."""

    def test_a_valid_pin_decodes(self) -> None:
        assert decode_pinned_download(_pin())["sha256"] == "ab" * 32

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_pinned_download("https://example.invalid")

    @pytest.mark.parametrize("url", ["http://example.invalid/x", "ftp://x", "example.invalid"])
    def test_a_url_that_is_not_https_is_refused(self, url: str) -> None:
        with pytest.raises(JSONTypeError, match="must be https"):
            decode_pinned_download(_pin(url=url))

    @pytest.mark.parametrize("digest", ["ab", "AB" * 32, "zz" * 32])
    def test_a_malformed_digest_is_refused(self, digest: str) -> None:
        with pytest.raises(JSONTypeError, match="64 lowercase hex"):
            decode_pinned_download(_pin(sha256=digest))


class TestDiskCeiling:
    """decode_disk_ceiling's contract."""

    def test_a_valid_ceiling_decodes(self) -> None:
        disk = decode_disk_ceiling(_disk())
        assert (disk["ceiling_gb"], disk["baseline_gb"]) == (150, 46)

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_disk_ceiling(150)

    @pytest.mark.parametrize(("ceiling", "baseline"), [(0, 46), (150, 0), (-1, -2)])
    def test_a_size_that_is_not_positive_is_refused(self, ceiling: int, baseline: int) -> None:
        with pytest.raises(JSONTypeError, match="must be positive"):
            decode_disk_ceiling(_disk(ceiling_gb=ceiling, baseline_gb=baseline))

    @pytest.mark.parametrize("baseline", [150, 200])
    def test_a_ceiling_at_or_under_the_baseline_is_refused(self, baseline: int) -> None:
        with pytest.raises(JSONTypeError, match="must be below ceiling_gb"):
            decode_disk_ceiling(_disk(baseline_gb=baseline))

    @pytest.mark.parametrize("date", ["2026-9-26", "26-09-2026", "2026/09/26", "yyyy-mm-dd"])
    def test_a_date_that_is_not_iso_is_refused(self, date: str) -> None:
        with pytest.raises(JSONTypeError, match="YYYY-MM-DD"):
            decode_disk_ceiling(_disk(baseline_measured=date))
