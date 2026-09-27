"""The runner-host base contract: every decode refusal, and the round trip."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.runner_base import (
    MachineVariable,
    decode_disk_ceiling,
    decode_host_base,
    decode_machine_variable,
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


def _variable(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw machine variable.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "name": "POETRY_CACHE_DIR",
        "value": "C:\\fleet\\poetry",
        "reason": "out of System32",
    }
    raw.update(overrides)
    return raw


class TestMachineVariable:
    """decode_machine_variable's contract, and the list's distinct names."""

    def test_a_valid_variable_decodes(self) -> None:
        assert decode_machine_variable(_variable()) == MachineVariable(
            name="POETRY_CACHE_DIR", value="C:\\fleet\\poetry", reason="out of System32"
        )

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_machine_variable("POETRY_CACHE_DIR")
        assert str(refused.value) == "machine variable must be a JSON object, got str"

    @pytest.mark.parametrize("name", ["", "9LIVES", "HAS SPACE", "A-B", "X=Y"])
    def test_a_name_that_is_not_a_plain_identifier_is_refused(self, name: str) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_machine_variable(_variable(name=name))
        assert str(refused.value) == (
            f"a machine variable's name must be a plain identifier, got {name!r}"
        )

    @pytest.mark.parametrize("name", ["Path", "PATH", "path"])
    def test_path_is_refused_because_it_has_its_own_field(self, name: str) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_machine_variable(_variable(name=name))
        assert str(refused.value) == (
            "the machine PATH is machine_path_entries, not a machine variable"
        )

    @pytest.mark.parametrize("field", ["value", "reason"])
    def test_an_empty_value_or_reason_is_refused(self, field: str) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_machine_variable(_variable(**{field: ""}))
        assert str(refused.value) == (
            "machine variable POETRY_CACHE_DIR needs a non-empty value and reason"
        )

    def test_a_name_declared_twice_in_any_case_is_refused(self) -> None:
        raw = _base(machine_environment=[_variable(), _variable(name="poetry_cache_dir")])
        with pytest.raises(JSONTypeError) as refused:
            decode_host_base(raw)
        assert str(refused.value) == (
            "machine_environment declares a name twice: ['poetry_cache_dir', 'poetry_cache_dir']"
        )

    def test_a_base_without_the_field_is_refused(self) -> None:
        raw = _base()
        del raw["machine_environment"]
        with pytest.raises(JSONTypeError, match="machine_environment"):
            decode_host_base(raw)

    def test_no_variables_is_a_valid_base(self) -> None:
        assert decode_host_base(_base(machine_environment=[]))["machine_environment"] == []


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
