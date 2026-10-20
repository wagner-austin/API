"""The runner-host base contract: every decode refusal, and the round trip."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.runner_base import (
    decode_disk_ceiling,
    decode_host_base,
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


def _disk(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw disk ceiling.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "ceiling_gb": 150,
        "baseline_gb": 18,
        "baseline_measured": "2026-10-20",
        "cache_path": "/home/gharunner/.cache",
        "cache_ceiling_gb": 60,
        "work_ceiling_gb": 15,
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
        with pytest.raises(JSONTypeError) as refused:
            decode_host_base([])
        assert str(refused.value) == "host base must be a JSON object, got list"

    def test_no_packages_is_refused(self) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_host_base(_base(apt_packages=[]))
        assert str(refused.value) == (
            "apt_packages must be non-empty; a fleet worker lacks the tagged tools runner jobs call"
        )

    @pytest.mark.parametrize("field", ["apt_packages", "disk"])
    def test_a_missing_field_is_refused(self, field: str) -> None:
        raw = _base()
        del raw[field]
        with pytest.raises(JSONTypeError, match=field):
            decode_host_base(raw)


class TestDiskCeiling:
    """decode_disk_ceiling's contract."""

    def test_a_valid_ceiling_decodes(self) -> None:
        disk = decode_disk_ceiling(_disk())
        assert (disk["ceiling_gb"], disk["baseline_gb"]) == (150, 18)

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_disk_ceiling(150)

    @pytest.mark.parametrize(("ceiling", "baseline"), [(0, 18), (150, 0), (-1, -2)])
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

    def test_the_cache_ceilings_decode(self) -> None:
        disk = decode_disk_ceiling(_disk())
        assert (disk["cache_path"], disk["cache_ceiling_gb"], disk["work_ceiling_gb"]) == (
            "/home/gharunner/.cache",
            60,
            15,
        )

    @pytest.mark.parametrize("path", ["home/gharunner/.cache", "~/.cache", ""])
    def test_a_cache_path_that_is_not_absolute_is_refused(self, path: str) -> None:
        with pytest.raises(JSONTypeError, match=f"cache_path must be absolute.*{path!r}"):
            decode_disk_ceiling(_disk(cache_path=path))

    @pytest.mark.parametrize(("cache", "work"), [(0, 15), (60, 0), (-1, -1)])
    def test_a_cache_ceiling_that_is_not_positive_is_refused(self, cache: int, work: int) -> None:
        with pytest.raises(
            JSONTypeError,
            match=f"cache_ceiling_gb and work_ceiling_gb must be positive, got {cache} and {work}",
        ):
            decode_disk_ceiling(_disk(cache_ceiling_gb=cache, work_ceiling_gb=work))

    @pytest.mark.parametrize("field", ["cache_path", "cache_ceiling_gb", "work_ceiling_gb"])
    def test_a_missing_cache_field_is_refused(self, field: str) -> None:
        raw = _disk()
        del raw[field]
        with pytest.raises(JSONTypeError, match=field):
            decode_disk_ceiling(raw)
