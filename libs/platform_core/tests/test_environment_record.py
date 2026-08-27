"""Tests for the machine and library axes of a run fingerprint.

The gap these close is stated in the module under test and in
``test_comparability_cpu``: a run that pulls no torch and runs out of a
directory environment recorded nothing that identified the machine, so two
such runs on two different boxes compared as one configuration. The tests here
cover the records themselves; the tests that assert the CONSEQUENCE for a
verdict live beside the verdict, in ``test_comparability``.

Every constructor is exercised against a value it must refuse as well as one
it must accept. A constructor that accepts everything round-trips perfectly
and validates nothing.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform

import pytest

from platform_core.environment_record import (
    HostRecord,
    PackageVersion,
    UnknownCoreCountError,
    capture_host_record,
    capture_package_versions,
    decode_host_record,
    decode_package_versions,
    encode_host_record,
    encode_package_versions,
    host_record,
    installed_version,
    package_versions,
    render_host_record,
    render_package_versions,
    stdlib_host_probe,
)
from platform_core.json_utils import JSONTypeError
from platform_core.testing import FakeHostProbe, FakeVersionReader

_LINUX = "Linux-5.14.0-x86_64-with-glibc2.34"


def _unknown_core_count() -> None:
    """Report a machine that does not say how many processors it has.

    A named function rather than a lambda, because a lambda's inferred return
    type would leave the protocol match unchecked.

    Returns:
        None, always.
    """
    return


def _eight_cores() -> int:
    """Report a machine with eight logical processors.

    Returns:
        Eight.
    """
    return 8


class TestHostRecord:
    def test_it_carries_the_machine_a_run_used(self) -> None:
        record = host_record(platform=_LINUX, machine="x86_64", logical_cores=8)

        assert record == HostRecord(platform=_LINUX, machine="x86_64", logical_cores=8)

    def test_it_refuses_an_unnamed_operating_system(self) -> None:
        # Unlike an image digest there is no honest "unknown" here: every run
        # has a machine, so an empty axis is a capture failure recorded as
        # a fact.
        with pytest.raises(ValueError, match="platform must name"):
            host_record(platform="", machine="x86_64", logical_cores=8)

    def test_it_refuses_an_unnamed_architecture(self) -> None:
        with pytest.raises(ValueError, match="machine must name"):
            host_record(platform=_LINUX, machine="", logical_cores=8)

    def test_it_refuses_a_machine_with_no_processors(self) -> None:
        with pytest.raises(ValueError, match="at least one, got 0"):
            host_record(platform=_LINUX, machine="x86_64", logical_cores=0)

    def test_it_refuses_a_negative_processor_count(self) -> None:
        with pytest.raises(ValueError, match="at least one, got -1"):
            host_record(platform=_LINUX, machine="x86_64", logical_cores=-1)


class TestCapturingTheHost:
    def test_it_reads_every_field_from_the_probe(self) -> None:
        probe = FakeHostProbe(platform=_LINUX, machine="aarch64", logical_cores=24)

        assert capture_host_record(probe) == HostRecord(
            platform=_LINUX, machine="aarch64", logical_cores=24
        )

    def test_a_probe_reporting_nothing_is_refused_rather_than_recorded(self) -> None:
        probe = FakeHostProbe(platform="", machine="x86_64", logical_cores=8)

        with pytest.raises(ValueError, match="platform must name"):
            capture_host_record(probe)


class TestTheStdlibProbe:
    def test_it_reports_the_machine_this_process_is_on(self) -> None:
        # Against the real stdlib rather than a fake: the point of this probe
        # is that it reads the actual machine, and a test that replaced the
        # reads would assert nothing about it.
        probe = stdlib_host_probe(os.cpu_count)

        assert probe.platform() == platform.platform()
        assert probe.machine() == platform.machine()
        assert probe.logical_cores() == os.cpu_count()

    def test_a_machine_that_reports_no_core_count_is_refused(self) -> None:
        probe = stdlib_host_probe(_unknown_core_count)

        with pytest.raises(UnknownCoreCountError, match="does not report"):
            probe.logical_cores()

    def test_it_returns_the_injected_count(self) -> None:
        probe = stdlib_host_probe(_eight_cores)

        assert probe.logical_cores() == 8

    def test_capture_reads_a_whole_record_through_it(self) -> None:
        record = capture_host_record(stdlib_host_probe(_eight_cores))

        assert record["platform"] == platform.platform()
        assert record["machine"] == platform.machine()
        assert record["logical_cores"] == 8


class TestPackageVersions:
    def test_it_sorts_by_name_so_two_equal_axes_render_identically(self) -> None:
        versions = package_versions({"torch": "2.6.0", "numpy": "2.3.5"})

        assert versions == (
            PackageVersion(name="numpy", version="2.3.5"),
            PackageVersion(name="torch", version="2.6.0"),
        )

    def test_it_accepts_an_environment_with_no_libraries_named(self) -> None:
        # Distinct from `capture_package_versions`, which refuses an empty
        # request: this constructor is also the decode path, and a stored
        # record with no packages is a fact to be read rather than rejected.
        assert package_versions({}) == ()

    def test_it_refuses_an_unnamed_distribution(self) -> None:
        with pytest.raises(ValueError, match="must name a distribution"):
            package_versions({"": "2.3.5"})

    def test_it_refuses_a_distribution_with_no_resolved_version(self) -> None:
        with pytest.raises(ValueError, match="'numpy' must carry the version"):
            package_versions({"numpy": ""})


class TestCapturingPackageVersions:
    def test_it_reads_each_named_distribution(self) -> None:
        read = FakeVersionReader({"numpy": "2.3.5", "torch": "2.6.0", "scipy": "1.16.3"})

        assert capture_package_versions(("torch", "numpy"), read) == (
            PackageVersion(name="numpy", version="2.3.5"),
            PackageVersion(name="torch", version="2.6.0"),
        )

    def test_it_refuses_to_record_that_nothing_decides_the_numbers(self) -> None:
        read = FakeVersionReader({"numpy": "2.3.5"})

        with pytest.raises(ValueError, match="name the distributions"):
            capture_package_versions((), read)

    def test_it_refuses_a_repeated_distribution(self) -> None:
        read = FakeVersionReader({"numpy": "2.3.5"})

        with pytest.raises(ValueError, match=r"repeated: \['numpy'\]"):
            capture_package_versions(("numpy", "numpy"), read)

    def test_a_library_the_run_does_not_have_propagates(self) -> None:
        # Rather than recording an empty or an "unknown" version. A caller
        # asking to fingerprint a library the run lacks named the wrong
        # library, and a soft value would make that unreadable later.
        read = FakeVersionReader({"numpy": "2.3.5"})

        with pytest.raises(KeyError):
            capture_package_versions(("numpy", "torch"), read)


class TestTheInstalledVersionReader:
    def test_it_reads_a_distribution_this_environment_really_has(self) -> None:
        # Against the real metadata, for the reason the stdlib probe test
        # gives: this function exists to read what is installed.
        assert installed_version("pytest") == importlib.metadata.version("pytest")

    def test_a_distribution_that_is_not_installed_propagates(self) -> None:
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            installed_version("a-distribution-that-is-not-installed")


class TestTheCodecs:
    def test_a_host_record_round_trips(self) -> None:
        record = host_record(platform=_LINUX, machine="x86_64", logical_cores=24)

        assert decode_host_record(encode_host_record(record)) == record

    def test_a_host_record_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_host_record(["not", "an", "object"])

    def test_a_host_record_missing_the_core_count_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_host_record({"platform": _LINUX, "machine": "x86_64"})

    def test_a_host_record_with_a_string_core_count_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_host_record({"platform": _LINUX, "machine": "x86_64", "logical_cores": "8"})

    def test_a_decoded_host_record_is_validated_not_merely_typed(self) -> None:
        with pytest.raises(ValueError, match="at least one, got 0"):
            decode_host_record({"platform": _LINUX, "machine": "x86_64", "logical_cores": 0})

    def test_package_versions_round_trip(self) -> None:
        versions = package_versions({"torch": "2.6.0", "numpy": "2.3.5"})

        assert decode_package_versions(encode_package_versions(versions)) == versions

    def test_packages_encode_as_a_list_so_the_order_is_the_canonical_one(self) -> None:
        versions = package_versions({"torch": "2.6.0", "numpy": "2.3.5"})

        assert encode_package_versions(versions) == [
            {"name": "numpy", "version": "2.3.5"},
            {"name": "torch", "version": "2.6.0"},
        ]

    def test_an_empty_package_axis_round_trips(self) -> None:
        assert decode_package_versions(encode_package_versions(())) == ()

    def test_a_package_axis_that_is_not_a_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a list, got dict"):
            decode_package_versions({"numpy": "2.3.5"})

    def test_a_package_entry_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_package_versions(["numpy==2.3.5"])

    def test_a_package_entry_missing_its_version_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_package_versions([{"name": "numpy"}])

    def test_a_repeated_package_is_refused_rather_than_collapsed(self) -> None:
        # Keeping the last would silently decide which version the record
        # claims, and the two entries disagree about exactly the thing the
        # axis exists to state.
        with pytest.raises(JSONTypeError, match="'numpy' appears twice"):
            decode_package_versions(
                [{"name": "numpy", "version": "2.3.5"}, {"name": "numpy", "version": "2.4.0"}]
            )

    def test_a_decoded_package_entry_is_validated_not_merely_typed(self) -> None:
        with pytest.raises(ValueError, match="must carry the version"):
            decode_package_versions([{"name": "numpy", "version": ""}])


class TestRendering:
    def test_a_host_renders_as_one_stable_key(self) -> None:
        record = host_record(platform=_LINUX, machine="x86_64", logical_cores=24)

        assert render_host_record(record) == f"{_LINUX}/x86_64/24"

    def test_two_machines_differing_only_in_cores_render_differently(self) -> None:
        eight = host_record(platform=_LINUX, machine="x86_64", logical_cores=8)
        twenty_four = host_record(platform=_LINUX, machine="x86_64", logical_cores=24)

        assert render_host_record(eight) != render_host_record(twenty_four)

    def test_packages_render_in_canonical_order(self) -> None:
        versions = package_versions({"torch": "2.6.0", "numpy": "2.3.5"})

        assert render_package_versions(versions) == "numpy=2.3.5,torch=2.6.0"

    def test_an_empty_package_axis_renders_as_nothing(self) -> None:
        assert render_package_versions(()) == ""

    def test_the_same_versions_in_two_orders_render_identically(self) -> None:
        one = package_versions({"numpy": "2.3.5", "torch": "2.6.0"})
        other = package_versions({"torch": "2.6.0", "numpy": "2.3.5"})

        assert render_package_versions(one) == render_package_versions(other)
