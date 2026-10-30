---
title: loki's tunnel stall is the hub starting cloudflared, not the tunnel
tags: [fleet, cloudflare-tunnel, cloudflared, ssh, loki, diphtheria, cpu-pressure, infrastructure]
related:
  - "[[fleet-runner-listener-isolation-and-dropped-jobs]]"
source_paths:
  - tools/fleet/src/fleet/core/ssh_options.py
  - tools/fleet/registry/machines/loki.json
source_git_blobs:
  "tools/fleet/src/fleet/core/ssh_options.py": afa311f94fa24e74a5733b34ffcca1f4b4273885
  "tools/fleet/registry/machines/loki.json": 9aa8a3bdca64b9a58f0e8fc77975f96e865fc7ca
fact_checked: "2026-10-30"
confidence: high
hubs: [infrastructure]
---

# loki's tunnel stall is the hub starting cloudflared, not the tunnel

Until loki's NixOS reinstall on 2026-10-26 the hub reached it only through loki's Cloudflare tunnel, `loki-ssh`, while loki held the UCI VPN.[^tunnel] Since then loki declares no tunnel and is reached at its tailnet address, so the stall below is a record of that route, and of any other tunnel node's.[^now] Every ssh and scp the fleet sends there is a fresh connection with `ConnectTimeout=10`.[^opts] Now and then one gave no banner within that limit. That closed or gave back the job as `NODE_UNREACHABLE: ... Connection timed out during banner exchange`: 12 of 274 loki claims in the three days after the per-call roll, 9 of them that message.[^a3] Read against the tunnel's own records, the stall is not in the tunnel. It is the hub (diphtheria) starting a fresh `cloudflared` for every call on a CPU it cannot get.

## What the tunnel recorded

`cloudflared tail`, run with the hub account's origin certificate, streams loki's connector log at debug level without restarting it. Each probe login sent its own `X-Probe-Id` header, so the connector named the second each login's websocket arrived. In the first probe window, 5 of 322 logins took over 5 s to reach the banner. In all five, loki logged the GET 5.4 to 7.3 s after the login began and established the origin connection in the same second. Neither loki's connector nor the hub's two tunnels wrote a connection-level record within 60 s of any of them, and loki's counters moved only `concurrent_requests`.[^tail] The time was lost before the request reached the edge, so it was not lost at loki.

## Not the network either

All eight home machines leave from one IPv6 /64 to Cloudflare's LAX colo. A request the edge answers by itself (Access refuses it) was slow 24 of 898 times from diphtheria, every time in the TLS step after a fast TCP connect, and 0 of 992 times from lacrymal on the same LAN, same uplink and same colo. Tailscale pings from the hub to loki's UCI address were never slow (0 of 649).[^path] The same uplink and edge served a quiet machine cleanly, so what differs is diphtheria itself. It ran at load 21 to 30 on 8 threads, with `/proc/pressure/cpu` "some" near 60%.[^path]

## Where the seconds go

The probe ran each login's `ProxyCommand` under strace. In every slow login the seconds fall between `cloudflared`'s exec and its first DNS socket: 2.1 to 3.9 s against a 0.36 s median. DNS, the TCP connect (0.01 s), the edge's answer to the ClientHello (0.02 s) and loki's sshd are all normal.[^phases] 81 to 96% of that start is 1227 reads of 32 KiB of `/usr/bin/cloudflared`, 40206336 bytes, each served from page cache in 13 to 23 microseconds.[^phases] cloudflared hashes its own executable on every start: `main` calls `cliutil.GetBuildInfo`, whose `Checksum` field runs `currentBinaryChecksum`, which `io.Copy`s `os.Executable()` into SHA-256 in 32 KiB reads.[^main][^checksum] The hash is CPU work, not disk. Sixty rounds of SHA-256 over the same file on diphtheria took 0.17 s median (0.26 s at most) in a calm minute, and when the hub is starved the same work waits for a core.[^bench] A `cloudflared` start past about 9 s therefore becomes a banner timeout under `ConnectTimeout=10`.

Tracing every read slows the very start being measured (banner median 0.69 s under strace against 0.51 s with no tracer attached). The first probe, run with no tracer, already had the stall (15 of 346 banners over 3 s). The last 370 logins were traced only at exec, connect and getsockopt (`--seccomp-bpf`), so cloudflared ran unstopped. Exec to first DNS socket read 0.31 s median, 20 over 2 s and 6.10 s at most, while DNS to TCP connect never passed 0.44 s and the TCP connect never passed 0.17 s. Of those logins' 25 banners over 3 s, the largest gap was cloudflared's start in 19, the TLS and websocket exchange up to loki's GET in 5 (the step diphtheria's edge-only requests also stalled in), and loki's sshd in 1.[^light]

## The causal test

`ab_probe.py` alternated the fleet's route, a `ProxyCommand` that starts `cloudflared access ssh` per login, with ssh to a single long-lived `cloudflared access tcp` forwarder on the hub that had started once. Both arms open a fresh ssh connection and a fresh websocket through the same tunnel to the same sshd. Over 188 logins each, the per-call arm reached the banner in 0.548 s median, with 17 over 2 s, 8 over 3 s, 4 over 5 s and 6.26 s at most. The forwarder arm took 0.205 s median, with 1 over 2 s, none over 3 s and 2.33 s at most.[^ab] The tunnel, the edge, loki and the hub's load were the same for both arms. Only the per-call cloudflared start differed, and removing it removed the stall.

Across the whole window, 08:25Z to 09:52Z, 17 of 1158 logins stalled past 5 s. In all 17 the request reached loki's connector late, and none had a connector-level tunnel record within 60 s. loki's own counters agree: tunnel registrations stayed at 85, closed connections at 146 and HA connections at 4 throughout, so the tunnel never reconnected.[^final]

The shared ssh connection tried first (`ControlMaster`) also removed this start, but it carried every call on one session, so a single stall held them all.[^opts] One long-lived cloudflared per tunnel node keeps a fresh ssh connection per call and removes the per-call start. That change is MCPs board task 4c604f4d.

[^tunnel]: [synthesis] -- API commit 31a1638ba, the last before ec95b830f deleted loki's Windows machine file: its tools/fleet/registry/machines/loki.json held, on lines 18 and 27, the tunnel hostname "loki-ssh.austinwagner.org" and the boot unit's purpose "the Cloudflare tunnel, this vpn-jump's only way in while it holds the UCI VPN". Read with `git show 31a1638ba:tools/fleet/registry/machines/loki.json`.
[^now]: `tools/fleet/registry/machines/loki.json:18` -- `"tunnel": null,`; `:10` -- `"tailnetIp": "100.120.151.28",`; `:24` -- the notes open "INSTALLED WITH NIXOS 2026-10-26 by the fleet's netboot install".
[^opts]: `tools/fleet/src/fleet/core/ssh_options.py:14-31` and `:42` -- "ONE CONNECTION PER CALL, NEVER A SHARED ONE (MCPs board task ee3d6cdc)." and `"ConnectTimeout=10",`.
[^a3]: [synthesis] -- MCPs board task 8b70921f, RECHECK note 01e77d44 (2026-10-25T08:19Z): `fleet_dispatch_events` from 2026-10-21T05:48:09Z to 2026-10-24T05:48:09Z, 274 claims by fleet-node-loki*, 11 launches given back NODE_UNREACHABLE (9 banner-exchange timeouts) and 1 job closed NODE_UNREACHABLE.
[^tail]: [synthesis] -- MCPs board task 8b70921f, notes d3569c34 (probe design) and 0873fe81 (2026-10-25T09:13Z): `correlate.py` over diphtheria:~/diag/stall-1025/loki.jsonl, hub-chat.jsonl, hub-search.jsonl, loki-metrics.tsv and login.tsv, 08:25Z to 09:02Z.
[^path]: [synthesis] -- MCPs board task 8b70921f, note 0873fe81: edge.tsv (diphtheria), edge-lacrymal.tsv (lacrymal) and tsping.tsv over the same window; `uptime` and `/proc/pressure/cpu` on diphtheria at 08:59Z.
[^phases]: [synthesis] -- MCPs board task 8b70921f, note 0873fe81: `phases.py` and `hashspan.py` over diphtheria:~/diag/stall-1025/login2.tsv and traces/ (strace -f -tt -T -s 0), and login.tsv, whose logins ran with no tracer.
[^bench]: [synthesis] -- MCPs board task 8b70921f, the closing note: `hashbench.py /usr/bin/cloudflared 60` on diphtheria at 2026-10-25T09:05Z, page-cache read median 0.010 s and SHA-256 median 0.168 s, max 0.263 s.
[^main]: https://github.com/cloudflare/cloudflared/blob/8e090e5bfa7a53e824fb2d7eb276f8ab08337b80/cmd/cloudflared/main.go#L56 (tag 2026.10.1, the hub's version) -- "bInfo := cliutil.GetBuildInfo(BuildType, Version)".
[^checksum]: https://github.com/cloudflare/cloudflared/blob/8e090e5bfa7a53e824fb2d7eb276f8ab08337b80/cmd/cloudflared/cliutil/build_info.go#L22-L82 (tag 2026.10.1) -- "Checksum:           currentBinaryChecksum()," and "currentPath, err := os.Executable()".
[^light]: [synthesis] -- MCPs board task 8b70921f, the closing note: `phases.py 3 login3.tsv traces-light` on diphtheria, 09:10Z to 09:52Z.
[^ab]: [synthesis] -- MCPs board task 8b70921f, the closing note: `ab_probe.py` over diphtheria:~/diag/stall-1025/ab.tsv, forwarder unit stall-fwd on 127.0.0.1:22122, 2026-10-25T09:12:28Z to 09:52:29Z.
[^final]: [synthesis] -- MCPs board task 8b70921f, the closing note: `correlate.py 5.0 2026-10-25 login.tsv login2.tsv login3.tsv ab.tsv` (corr-final.txt), and loki-metrics.tsv's 409 samples from 08:25:25Z to 09:52:07Z.
