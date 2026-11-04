---
title: A node on battery takes no check, and NixOS VM checks go to whichever node measures KVM
tags: [fleet, dispatch, capacity-probe, battery, nixos, kvm, infrastructure]
related:
  - "[[fleet-runner-listener-isolation-and-dropped-jobs]]"
source_paths:
  - tools/fleet/src/fleet/core/linux_capacity_probe.py
  - tools/fleet/src/fleet/core/node_refusal.py
  - tools/fleet/src/fleet/contracts/tagged_tools.py
  - tools/fleet/src/fleet/contracts/runner_tick.py
  - tools/fleet/registry/projects/MCPs/fleet-nixos.json
  - tools/fleet/tests/test_linux_capacity_probe.py
  - tools/fleet/tests/test_capacity_battery.py
source_git_blobs:
  "tools/fleet/src/fleet/core/linux_capacity_probe.py": 8f25450b7bd3536ef3f1b1179bf40d1763ed7a87
  "tools/fleet/src/fleet/core/node_refusal.py": dbb0f0022d762d541623f72848f494420bcac4ef
  "tools/fleet/src/fleet/contracts/tagged_tools.py": 404b961a3ff567cb1bdc06aa33de128a812db152
  "tools/fleet/src/fleet/contracts/runner_tick.py": e0f87e25be83cbf421fcc8a274a99352c05fb1b3
  "tools/fleet/registry/projects/MCPs/fleet-nixos.json": b238202a7cea3ef2ac7eb35cac69fd16913bd6db
  "tools/fleet/tests/test_linux_capacity_probe.py": 99fe2bd463e46c0d6ae9085db814c77ca7edd210
  "tools/fleet/tests/test_capacity_battery.py": c818a01fdcb98d2aa64d85397adc89ff91cf69b1
fact_checked: "2026-11-04"
confidence: high
hubs: [infrastructure]
---

# A node on battery takes no check, and NixOS VM checks go to whichever node measures KVM

On 2026-11-03 loki, a laptop, ran back-to-back NixOS VM checks off its charger and its battery fell five percent in seven minutes.[^loki] Two things made it loki: MCPs `fleet-mcp/nixos`'s check was pinned to loki by name, and nothing in a node's capacity probe said whether it ran on mains.[^pin] Both are now measured readings in the existing mechanisms rather than names in the registry.

## The power reading

The linux capacity probe prints `on_battery` every tick from `/sys/class/power_supply`.[^predicate] It reads `yes` only when a system battery reads `Discharging` and no external supply reads `online` 1. An external supply is any other type, `Mains` or a USB-C port the machine draws power from.[^external] A battery's status alone is not enough: lenovold's firmware conservation mode holds its battery near full on its charger, and at 2026-11-04T03:06Z its BAT0 read `Discharging` at 93 percent while its adapter read online.[^lenovold] Until the adapter was read beside the battery, that refused a node on mains, and its first fleet check waited behind the refusal. A supply whose `scope` is `Device` is a peripheral's (a wireless mouse or keyboard) and is skipped, so a desktop with a wireless mouse is never read as on battery.[^device]

## The refusal

`machine_refusal` refuses a node whose probe read `on_battery` with its own code, `NODE_ON_BATTERY`, naming the reading. It comes after the disk refusal and before every load refusal.[^refusal] So a reader is told the one cause that no falling load clears, and the node claims again on its first tick back on mains.[^order]

## The record a program keeps

The runner sends the reading in every tick's load as `onBattery`.[^tick] MCPs migration 903 keeps each tick in `fleet_runner_tick_history` for 30 days beside its verdict and `claiming`, so what a node did across a window outlives the next tick, which `fleet_runner_ticks` replaces.[^history] A charger pull on 2026-11-03 could be read only from the runner's daily log on the hub, which no audit check reads.

## The nixos-test tag

The toolchain probe reports a `nixos-test` tool tag, asked inside a check's unit. It requires `/dev/kvm` readable and writable, and nix's `system-features` naming `kvm` and `nixos-test`.[^tag] The `MCPs/fleet-nixos` project requires `linux` and `nixos-test` and names no node.[^project] On 2026-11-04 the tag read on demure, diphtheria, lacrymal, lavender, lenovold, loki, pendragon, sedona and serendipity, and not on colossus, which runs NixOS without `/dev/kvm`.[^nodes]

[^loki]: `tools/fleet/src/fleet/core/linux_capacity_probe.py:74-75` -- "back-to-back NixOS VM checks at 23 to 32 W, its battery falling from 35 to"
[^pin]: `tools/fleet/src/fleet/contracts/tagged_tools.py:184-186` -- "fleet registry pinned the project to loki by name on 2026-10-28 (MCPs"
[^predicate]: `tools/fleet/src/fleet/core/linux_capacity_probe.py:70-71` -- "entry whose ``type`` is ``Battery`` reads ``status`` ``Discharging`` and no"
[^external]: `tools/fleet/src/fleet/core/linux_capacity_probe.py:72` -- "external supply (any other ``type``: ``Mains``, or a ``USB`` port the"
[^lenovold]: `tools/fleet/src/fleet/core/linux_capacity_probe.py:78-80` -- "and at 2026-11-04T03:06Z its BAT0 read ``Discharging`` at 93 percent while"; `tools/fleet/tests/test_linux_capacity_probe.py:489` -- "lenovold at 2026-11-04T03:06Z: its firmware's conservation mode held BAT0"
[^device]: `tools/fleet/src/fleet/core/linux_capacity_probe.py:82-83` -- "``Device`` is a peripheral's (a wireless mouse or keyboard), not the"
[^refusal]: `tools/fleet/src/fleet/core/node_refusal.py:96-104` -- "is running on its battery: its probe read a power supply"
[^order]: `tools/fleet/tests/test_capacity_battery.py:52` -- `def test_the_battery_refusal_comes_before_every_load_refusal(self) -> None:`; `:69` -- `def test_a_node_short_of_disk_is_refused_for_its_disk_first(self) -> None:`
[^tick]: `tools/fleet/src/fleet/contracts/runner_tick.py:65` -- "The queue keeps it in each tick's history, so an audit can read"
[^history]: [synthesis] -- MCPs migration 903, `packages/db/migrations/903_fleet_runner_tick_history.sql` at MCPs 20a2ae4e7, and its writer `packages/db/src/fleet-dispatch/runner-tick-history.ts`, landed for MCPs board task 3268afdf; this repository carries only the runner that sends the reading.
[^tag]: `tools/fleet/src/fleet/contracts/tagged_tools.py:124` -- `KVM_DEVICE: Final = "/dev/kvm"`; `:130` -- `NIXOS_TEST_FEATURES: Final = ("kvm", "nixos-test")`
[^project]: `tools/fleet/registry/projects/MCPs/fleet-nixos.json:5-8` -- `"required_tags": [` with `"linux"` and `"nixos-test"`, and no `node` key.
[^nodes]: [synthesis] -- `fleet_status`'s Runners section read at 2026-11-04T03:29:04Z, each runner's detected tags; colossus's lack of `/dev/kvm` is MCPs board task 3268afdf's 2026-11-03T08:59Z measurement.
