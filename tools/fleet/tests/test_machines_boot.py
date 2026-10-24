"""What the shipped roster's machines declare for booting: their boot units and boot MACs.

These pins lived in MCPs' fleet-mcp suite, read from the roster the fleet
runner staged beside each MCPs job, which is this repository's as last
rolled. A roster edit here and the MCPs test pinning it could not move in
one commit, so on 2026-10-19 API ee52f9b22 (pendragon's MAC) failed MCPs'
fleet-mcp at 58bbf5887 and 25c1f2881 with no MCPs change, and no MCPs head
could deploy (MCPs board task 9cd8a32e). A pin on the roster's own data
belongs beside the roster, where the edit that moves it moves the pin in the
same commit; the rules on its shape (a hub never carries a MAC, no two boot
units share a key) are fleet-mcp's decoder's, which refuses a roster that
breaks them on every audit pass.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_list,
    require_dict,
)

from tests._registry import shipped_document

#: lavender's wired Realtek GbE adapter, measured 2026-10-18T23:55:15Z.
LAVENDER_MAC = "1c:69:7a:fe:78:9a"

#: pendragon's wired adapter, declared at ee52f9b22 (MCPs board task 1e9ab738).
PENDRAGON_MAC = "d8:5e:d3:84:53:d4"

#: demure's Intel I225-V Ethernet, declared at 493821b9f (MCPs board task e2ffed0d)
#: when it was austinpc, renamed demure for MCPs board task e0c286f5.
DEMURE_MAC = "d8:43:ae:24:7c:04"

#: The udev ID_SERIAL of each netboot machine's install disk, read over ssh
#: with udevadm on 2026-10-22 (MCPs board task c64006b9): lavender's one SATA
#: SSD, and pendragon's NVMe, beside which it also carries a 512 GB SD card
#: only 36 MB smaller, which a largest-disk rule would one day choose.
#: demure's one NVMe, whose netboot install accepted it on 2026-10-22 (arm b54008dd).
LAVENDER_INSTALL_DISK = "Samsung_SSD_870_EVO_1TB_S75BNL0WA21603B"
PENDRAGON_INSTALL_DISK = "Micron_2400_MTFDKBK512QFM_2316400BBFD4_1"
DEMURE_INSTALL_DISK = "Samsung_SSD_990_PRO_2TB_S73WNJ0TC01638D_1"

#: diphtheria's adb server, as its roster file declares it.
ADB_LISTENER: JSONObject = {
    "kind": "listener",
    "address": "127.0.0.1",
    "port": 5037,
    "purpose": (
        "the adb server phone-mcp reaches the Pixel 8 on this host's USB through "
        "(board task 49b6588b)"
    ),
    "remedy": (
        "make phone-up on diphtheria (the operator's 2026-09-25 ruling: adb is started by hand, "
        "never by a scheduled task)"
    ),
}


def _machines() -> JSONObject:
    """The shipped roster's machines.

    Returns:
        Each machine's file, by name.
    """
    return require_dict(shipped_document(), "machines")


def _units(name: str) -> list[JSONObject]:
    """One machine's declared boot units.

    Args:
        name: The machine.

    Returns:
        Its ``boot.units``, in file order.
    """
    boot = require_dict(require_dict(_machines(), name), "boot")
    return [narrow_json_to_dict(unit) for unit in narrow_json_to_list(boot["units"])]


def _key(unit: JSONObject) -> str:
    """A unit's key, as fleet-mcp's ``bootUnitKey`` spells it for the kinds pinned here.

    Args:
        unit: The unit.

    Returns:
        ``systemd:<unit>``, ``listener:<address>:<port>`` or ``task:<path><name>``.
    """
    kind: JSONValue = unit["kind"]
    if kind == "systemd":
        return f"systemd:{unit['unit']}"
    if kind == "listener":
        return f"listener:{unit['address']}:{unit['port']}"
    return f"task:{unit['path']}{unit['name']}"


def test_measured_boot_macs_stay_pinned_and_only_linux_workers_declare_one() -> None:
    machines = {name: narrow_json_to_dict(entry) for name, entry in _machines().items()}
    declared = {name: m["bootMac"] for name, m in machines.items() if m["bootMac"] is not None}
    # Automatic admission adds a disabled worker before its first install.
    # Pin measured identities without making each admission a test-code edit.
    assert {name: declared[name] for name in ("demure", "lavender", "pendragon")} == {
        "demure": DEMURE_MAC,
        "lavender": LAVENDER_MAC,
        "pendragon": PENDRAGON_MAC,
    }
    for name in declared:
        assert machines[name]["role"] == "worker"
        assert machines[name]["platform"] == "linux"
    assert machines["loki"]["bootMac"] is None
    assert machines["diphtheria"]["bootMac"] is None


def test_pendragon_alone_declares_netboot_wifi_and_every_machine_says_whether_it_does() -> None:
    # MCPs board task f78f7e33: pendragon's one USB-C port is also its
    # charging port, so its netboot install must bring up Wi-Fi to lose the
    # dock. fleet-mcp's decoder requires the key and refuses true without a
    # bootMac.
    machines = {name: narrow_json_to_dict(entry) for name, entry in _machines().items()}
    assert {name: type(m.get("netbootWifi")) for name, m in machines.items()} == dict.fromkeys(
        machines, bool
    )
    assert sorted(name for name, m in machines.items() if m["netbootWifi"] is True) == ["pendragon"]


def test_measured_disk_serials_stay_pinned_and_only_boot_macs_have_disk_rules() -> None:
    # MCPs board task c64006b9: a netboot install wipes only the disk the
    # roster selects. Keep the measured serials pinned; newly admitted
    # workers use only-disk, which refuses zero or multiple candidate disks.
    # fleet-mcp's decoder validates both rules and refuses a bootMac without
    # a disk. Every other machine still carries null.
    machines = {name: narrow_json_to_dict(entry) for name, entry in _machines().items()}
    named = {name: m["installDisk"] for name, m in machines.items() if m["installDisk"] is not None}
    assert {name: named[name] for name in ("demure", "lavender", "pendragon")} == {
        "demure": {"rule": "serial", "serial": DEMURE_INSTALL_DISK},
        "lavender": {"rule": "serial", "serial": LAVENDER_INSTALL_DISK},
        "pendragon": {"rule": "serial", "serial": PENDRAGON_INSTALL_DISK},
    }
    assert sorted(name for name, m in machines.items() if m["installDisk"] is None) == sorted(
        name for name, m in machines.items() if m["bootMac"] is None
    )


def test_demure_declares_no_unit_past_the_linux_baseline_since_its_reimage() -> None:
    # Reinstalled with Ubuntu on 2026-10-22 as austinpc (MCPs board task
    # e2ffed0d) and renamed demure (MCPs board task e0c286f5): its Windows
    # scheduled tasks went with Windows, the four it still declared until the
    # rename among them, and the hub runner is diphtheria's user unit since
    # MCPs board task c39ee056, so no unit is held to its boot.
    assert _units("demure") == []


def test_diphtheria_s_adb_server_is_a_listener_started_by_hand() -> None:
    listeners = [unit for unit in _units("diphtheria") if unit["kind"] == "listener"]
    assert [unit for unit in listeners if unit["port"] == 5037] == [ADB_LISTENER]


def test_diphtheria_declares_docker_its_tailnet_ingress_and_the_device_listeners() -> None:
    assert [_key(unit) for unit in _units("diphtheria")] == [
        "systemd:containerd.service",
        "systemd:docker.service",
        "systemd:corvis-tailnet-ingress.service",
        "listener:127.0.0.1:5037",
        "listener:172.18.0.1:8040",
        "listener:172.18.0.1:8042",
    ]


def test_lavender_declares_no_unit_past_the_linux_baseline_since_its_reimage() -> None:
    # Reimaged to Ubuntu on 2026-10-19 (MCPs board task 1e9ab738): its Windows
    # runners, the WSL keepalive and lavender-wsl's runners went with Windows.
    assert _units("lavender") == []
