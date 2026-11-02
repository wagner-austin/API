"""A GPU as a runner tick carries it, and every refusal of one (MCPs board task b4d5ca0f)."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.gpu_reading import (
    GpuMemory,
    GpuReading,
    GpuSource,
    decode_gpus,
    encode_gpus,
    gpu_problem,
    require_gpu,
    vendor_of,
)

#: colossus's card on 2026-11-02 while llama-server held a 17 GB model.
B60 = GpuReading(
    address="0000:08:00.0",
    vendor="intel",
    name="Battlemage G21 [Arc Pro B60]",
    driver="xe",
    memory=GpuMemory(source=GpuSource.DMEM, total_bytes=25669140480, free_bytes=8546701312),
)

#: diphtheria's card: no driver bound, so no counter holds it.
UNBOUND = GpuReading(
    address="0000:01:00.0", vendor="nvidia", name="GA106 [RTX A2000 12GB]", driver=None, memory=None
)


class TestTheVendor:
    @pytest.mark.parametrize(
        ("vendor_id", "vendor"),
        [
            ("0x10de", "nvidia"),
            ("0x8086", "intel"),
            ("0x1002", "amd"),
            ("0x102b", "pci-102b"),
            ("8086", None),
            ("0x80866", None),
            ("0xGGGG", None),
        ],
    )
    def test_a_pci_vendor_id_is_named_or_refused(self, vendor_id: str, vendor: str | None) -> None:
        assert vendor_of(vendor_id) == vendor


class TestTheShape:
    def test_a_measured_card_and_an_unread_one_are_inside_it(self) -> None:
        assert gpu_problem(B60) is None
        assert gpu_problem(UNBOUND) is None
        assert require_gpu(B60) is B60

    @pytest.mark.parametrize(
        ("gpu", "message"),
        [
            ({**B60, "address": "08:00.0"}, "a GPU's address must be a PCI address, got '08:00.0'"),
            ({**B60, "vendor": "Intel"}, "0000:08:00.0's vendor is outside the vocabulary: 'Intel'"),
            ({**B60, "name": ""}, "0000:08:00.0's name must be 1 to 200 characters, got 0"),
            ({**B60, "name": "n" * 201}, "0000:08:00.0's name must be 1 to 200 characters, got 201"),
            ({**B60, "driver": "x e"}, "0000:08:00.0's driver is not a driver's name: 'x e'"),
            (
                {**B60, "memory": GpuMemory(source=GpuSource.DMEM, total_bytes=0, free_bytes=0)},
                "got 0 free of 0",
            ),
            (
                {**B60, "memory": GpuMemory(source=GpuSource.DMEM, total_bytes=10, free_bytes=11)},
                "got 11 free of 10",
            ),
            (
                {**B60, "memory": GpuMemory(source=GpuSource.DMEM, total_bytes=10, free_bytes=-1)},
                "got -1 free of 10",
            ),
        ],
    )
    def test_a_gpu_outside_it_is_named(self, gpu: GpuReading, message: str) -> None:
        problem = gpu_problem(gpu)

        assert problem is not None
        assert message in problem
        with pytest.raises(JSONTypeError, match=message.replace("[", r"\[")):
            require_gpu(gpu)


class TestTheWire:
    def test_a_read_card_and_an_unread_one_round_trip(self) -> None:
        encoded = encode_gpus((B60, UNBOUND))

        assert encoded == [
            {
                "address": "0000:08:00.0",
                "vendor": "intel",
                "name": "Battlemage G21 [Arc Pro B60]",
                "driver": "xe",
                "memory": {"source": "dmem", "totalBytes": 25669140480, "freeBytes": 8546701312},
            },
            {
                "address": "0000:01:00.0",
                "vendor": "nvidia",
                "name": "GA106 [RTX A2000 12GB]",
                "driver": None,
                "memory": None,
            },
        ]
        assert decode_gpus(encoded) == (B60, UNBOUND)
        assert decode_gpus([]) == ()

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ({}, "gpus must be a JSON list, got dict"),
            ([1], "a GPU must be a JSON object, got int"),
            (
                [{"address": "0000:01:00.0", "vendor": "nvidia", "name": "x", "driver": None}],
                "a GPU's memory is missing; a tick carries it or null",
            ),
            (
                [{"address": "0000:01:00.0", "vendor": "nvidia", "name": "x", "driver": None, "memory": 1}],
                "a GPU's memory must be a JSON object or null, got int",
            ),
            (
                [
                    {
                        "address": "0000:01:00.0",
                        "vendor": "nvidia",
                        "name": "x",
                        "driver": None,
                        "memory": {"source": "sysfs", "totalBytes": 1, "freeBytes": 1},
                    }
                ],
                "a GPU's memory source must be nvidia-smi or dmem, got 'sysfs'",
            ),
            (
                [
                    {
                        "address": "0000:01:00.0",
                        "vendor": "nvidia",
                        "name": "x",
                        "driver": None,
                        "memory": {"source": "dmem", "totalBytes": "1", "freeBytes": 1},
                    }
                ],
                "Field 'totalBytes' must be an integer, got str",
            ),
            (
                [{"vendor": "nvidia", "name": "x", "driver": None, "memory": None}],
                "Missing required field 'address'",
            ),
            (
                [{"address": "0000:01:00.0", "vendor": "nvidia", "name": "x", "driver": 3, "memory": None}],
                "Field 'driver' must be a string, got int",
            ),
            (
                [{"address": "0000:01:00.0", "vendor": "NVIDIA", "name": "x", "driver": None, "memory": None}],
                "0000:01:00.0's vendor is outside the vocabulary: 'NVIDIA'",
            ),
        ],
    )
    def test_a_malformed_list_is_refused_by_name(self, value: JSONValue, message: str) -> None:
        with pytest.raises(JSONTypeError, match=message):
            decode_gpus(value)
