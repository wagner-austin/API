---
title: A node that cannot reach the internet takes no check
tags: [fleet, dispatch, capacity-probe, network, egress, infrastructure]
related:
  - "[[fleet-runner-battery-refusal-and-nixos-test-tag]]"
source_paths:
  - tools/fleet/src/fleet/contracts/egress_reading.py
  - tools/fleet/src/fleet/core/linux_egress_probe.py
  - tools/fleet/src/fleet/core/node_refusal.py
  - tools/fleet/tests/test_egress_gate.py
  - tools/fleet/tests/test_linux_egress_probe.py
  - tools/fleet/tests/test_node_agent_ticks.py
  - tools/fleet/tests/test_node_refusal.py
  - libs/platform_core/src/platform_core/error_codes_fleet.py
source_git_blobs:
  "tools/fleet/src/fleet/contracts/egress_reading.py": 55f78af8c04dd5759729e84edf0a2ea25b0f5d20
  "tools/fleet/src/fleet/core/linux_egress_probe.py": 35cde8f9cada8cbd32e809c950de3eaa40e5d3f7
  "tools/fleet/src/fleet/core/node_refusal.py": 79065403e6d4e0d65161bfe8dd11bdb63694e039
  "tools/fleet/tests/test_egress_gate.py": ce97629f62aea0ae3d0fa3dd40bda3ff0a73f3c1
  "tools/fleet/tests/test_linux_egress_probe.py": daec01b4c211fa9d0a9c8abd68c9307291037363
  "tools/fleet/tests/test_node_agent_ticks.py": 6d6937d0bdb8433c05e4950bd6eccd9bb8c26cbd
  "tools/fleet/tests/test_node_refusal.py": db5be586646c028f83863a205622821a550eafcb
  "libs/platform_core/src/platform_core/error_codes_fleet.py": e897b564f1bb4ff7cefb6ce8ebf0803f5bb9e7b2
fact_checked: "2026-11-25"
confidence: high
hubs: [infrastructure]
---

# A node that cannot reach the internet takes no check

On 2026-11-25 lavender lost its IPv4 lease on the LAN and kept answering ssh over the tailnet. The hub's capacity probe reached it over ssh, so the node read as healthy and went on claiming checks. Fleet job 376c6670 then failed `make setup` in packages/maketools when poetry could not reach pypi.org, and that failure was read as maketools' own red.[^lavender] Nothing in the probe asked whether the node itself could reach out. Now it asks, and the node runner's gate refuses a node that reaches neither package index.

## The reading

Every tick, the Linux capacity probe asks both indexes a check's setup downloads from, `https://pypi.org/simple/pip/` (poetry's) and `https://registry.npmjs.org/` (npm's). Each is asked from the node with `curl --head`, with 3 s to connect and 5 s in all, so a node whose packets vanish costs the probe at most ten seconds.[^targets] For each target the probe prints `egress_<n>=<curl exit> <target>`, and then `egress_count=<n>`.[^lines]

Any HTTP answer, whatever its status, is reach: the name resolved, a route existed and the server spoke. A node reaches the internet when at least one index answers. Requiring both would refuse every check on the fleet whenever one index alone was down, including checks that download only from the other.[^reach]

The reader refuses a reading it cannot interpret rather than calling it reach or its absence. That covers a count that is missing, not a whole number or zero, a counted target with no line, and a line that is not an exit and a target. Exit 127, the shell's "command not found", means the node has no curl, which says nothing about its network, so that probe reading is refused too.[^reader] Every Linux node carries curl in its system profile, `/run/current-system/sw/bin/curl`, measured over ssh on all eleven on 2026-11-25.[^curl]

## The refusal

`machine_refusal` refuses a node that reached neither index with `NODE_EGRESS_LOST`. The refusal names each target and how curl ended, in words where curl's exit code table has them (exit 6 "could not resolve its host", 7 "could not connect to it", 28 "did not answer within the probe's bound") and by number otherwise.[^refusal] It comes after the battery refusal and before every load refusal, since no load falling clears it.[^order] The code is distinct from `NODE_UNREACHABLE`, the hub failing to reach the node: here the node answers and cannot reach out.[^code]

## What the queue sees

A refused node's runner tick reaches the queue claiming nothing, with no fits, and the refusal as its verdict. The runner never calls `dispatch_claim`, so the queue hands the node no check. `plan_dispatch` reads the same `assess`, so it skips the node too.[^tick] The queue keeps each tick in its history beside its verdict, as it does for the battery refusal ([[fleet-runner-battery-refusal-and-nixos-test-tag]]).

## How it was proved

The gate case runs the probe the fleet sends inside a user and network namespace of its own (`unshare --user --map-root-user --net`), which has no route out, and asks the runner's own gate.[^gate] At API 6eee92fa9, before the reading existed, fleet job 852779e2 on serendipity failed exactly that case: the gate admitted a node with no network, while the same probe on the host's network passed. After the change the case passes. The probe's lines are also run under `sh` against loopback servers the tests start: a 404 is reach, a port nothing listens on is exit 7, a socket that never answers is exit 28 within the bound, an `.invalid` name is exit 6, and an empty `PATH` prints 127.[^sh]

## What this does not cover

The probe asks two package indexes. A check whose setup downloads from somewhere else, such as GitHub or a model hub, is not protected when only that host is unreachable. The probe runs once per tick, so a node that loses egress mid-check fails that check as before. Why lavender's DHCP lease lapsed was not determined: the router reboot restored it, and the router's log was not read.[^cause]

[^lavender]: `tools/fleet/src/fleet/contracts/egress_reading.py:5` -- "lost its IPv4 lease to the LAN's DHCP. It still answered ssh over the"
[^targets]: `tools/fleet/src/fleet/core/linux_egress_probe.py:35` -- `TARGETS: Final = ("https://pypi.org/simple/pip/", "https://registry.npmjs.org/")`; `:38` -- `CONNECT_SECONDS: Final = 3`; `:41` -- `MAX_SECONDS: Final = 5`
[^lines]: `tools/fleet/src/fleet/core/linux_egress_probe.py:7` -- "``egress_<n>=`` then curl's exit code, a space and the target,"
[^reach]: `tools/fleet/src/fleet/contracts/egress_reading.py:13` -- "answer, whatever its status, proves the name resolved, a route existed and"; `:69` -- `return any(attempt["curl_exit"] == REACHED_EXIT for attempt in attempts)`
[^reader]: `tools/fleet/src/fleet/core/linux_egress_probe.py:97` -- "which is not a positive whole number"; `tools/fleet/src/fleet/contracts/egress_reading.py:20` -- "has no curl, which says nothing about its network, so"
[^curl]: `tools/fleet/src/fleet/core/linux_egress_probe.py:21` -- "in its system profile (``/run/current-system/sw/bin/curl``, measured over ssh"
[^refusal]: `tools/fleet/src/fleet/core/node_refusal.py:115` -- "cannot reach the internet: {asked}. Every check's setup"; `tools/fleet/src/fleet/contracts/egress_reading.py:40` -- `6: "could not resolve its host",`
[^order]: `tools/fleet/tests/test_node_refusal.py:115` -- `def test_the_battery_is_named_before_the_lost_egress(self) -> None:`; `:128` -- `def test_the_lost_egress_is_named_before_every_load_refusal(self) -> None:`
[^code]: `libs/platform_core/src/platform_core/error_codes_fleet.py:61` -- `NODE_EGRESS_LOST = "NODE_EGRESS_LOST"`
[^tick]: `tools/fleet/tests/test_node_agent_ticks.py:182` -- `def test_a_node_that_reaches_no_package_index_asks_the_queue_for_nothing(`
[^gate]: `tools/fleet/tests/test_egress_gate.py:38` -- `NO_NETWORK = ("unshare", "--user", "--map-root-user", "--net")`; `:79` -- `def test_a_node_with_no_network_claims_nothing_and_names_its_lost_egress(`
[^sh]: `tools/fleet/tests/test_linux_egress_probe.py:146` -- `class TestTheLinesUnderSh:`
[^cause]: [synthesis] -- MCPs board task e70cdf05's thread: the operator's report that he rebooted the router, and the 2026-11-25T09:11Z measurement of DHCP leases on all five hosts after it.
