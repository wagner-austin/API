---
title: Every measurement node is reachable by SSH alias, and three traps make it look otherwise
tags: [platform, fleet, ssh, tailscale, operations]
related: ["[[mjwarp-cannot-compile-under-warp-deterministic-mode]]", "[[jax-cuda-unavailable-on-windows]]", "[[open-questions-and-what-would-answer-them]]"]
provenance:
  - "~/.ssh/config"
  - "tailscale status"
  - "agent board opus-emerald-0816 2026-08-16"
  - "nvidia-smi on lavender 2026-09-04"
fact_checked: 2026-09-13
confidence: high
hubs: [platform-constraints]
---

# Every measurement node is reachable by SSH alias, and three traps make it look otherwise

Measurements that need hardware this workstation does not have run on spare machines
joined to a Tailscale tailnet. Reaching them requires no code and no credentials to pass
around: `~/.ssh/config` on `austinpc` defines an alias per node, and the keypair at
`~/.ssh/id_ed25519` is already authorised on each.[^1]

```
$ ssh sedona    -> sedona
$ ssh lavender  -> DESKTOP-JAHBOEJ
```

Any session running on `austinpc` inherits both the key and the config, so this needs no
setup and no handoff. It is one-time-per-machine and survives reboots — `sshd` is
Automatic on both live nodes.[^1]

## The fleet

| alias | tailnet IP | hardware | role |
|---|---|---|---|
| `sedona` | 100.95.76.122 | RTX 3070 Ti Laptop — sm_86, 46 SMs, 8 GiB | GPU node; venv at `C:\navprobe\.venv` |
| `lavender` | 100.85.214.124 | i7-11700K (AVX-512); **GTX 1630 — sm_75 (Turing), 4 GiB**, beside the Intel UHD 750 | GPU node — the fleet's only non-Ampere CUDA device[^9] |
| `emerald` | 100.125.152.27 | A10-7800 Steamroller — **pre-AVX2**, iGPU only | CPU control; sshd enabled 2026-08-17 (correction below); **offline since about 2026-08-19**[^12] |
| `lenovoold` | 100.96.79.34 | i5-5200U Broadwell — **AVX2, no AVX-512**, iGPU only, 7.9 GB, Windows 10 build 19045 | CPU control; venv at `C:\navprobe\.venv`; key-only sshd; not CI-dispatchable[^13] |

`austinpc` itself holds the RTX 3090 Ti (sm_86, 84 SMs).[^9] Against `sedona`'s 3070 Ti
that is a 46-vs-84 SM comparison at one architecture.

**`lavender` makes the fleet two architectures, and this page said otherwise for eleven
days.** Until 2026-09-04 the row above read "Intel UHD 750 only | CPU node — no discrete
GPU", and the paragraph here concluded that the second-architecture question in
[[open-questions-and-what-would-answer-them]] "cannot be answered by any machine currently
owned". A GTX 1630 went into that box on 2026-08-24. The claim was true when written on
2026-08-16 and false from the day the card was installed; it is withdrawn.[^9]

The correction is recorded rather than quietly applied because the stale row cost real
work: on 2026-09-04 a session probing the fleet for an unrelated reason read this page,
found it said one architecture, and reported the 1630 as a new discovery — eleven days
after it had been measured in more detail. A hardware inventory is what other sessions
read to learn what the fleet HAS, so a stale row here is not a documentation defect but a
duplicated-measurement defect.[^10]

What makes the 1630 worth more than its size suggests: it runs **driver 591.86, the same
driver as `austinpc`'s 3090 Ti**.[^9] A Turing-vs-Ampere comparison across those two holds
the driver constant, which the HPC3 V100 comparison could not — so `lavender` is a cleaner
second architecture than a faster remote card would be.

Nodes are renamed to a colour/place scheme as they are commissioned. `lavender` was
`desktop-jahboej`; `emerald` was `desktop-li867ht`. Because a Tailscale rename is instant
and a Windows rename is not (below), correlate across a rename by the stable identity
triple rather than by any name:[^6]

| node | MTM | serial | UUID |
|---|---|---|---|
| `lavender` | 90RJ0080US | MJ0GVS2H | `40F89E80-8561-11EC-A78F-1411F17D4C00` |
| `emerald` | 90BG003JUS | R302F9AZ | `2428F0B8-28F9-11E6-B358-182A98911300` |
| `lenovoold` | 80JH | PF04TWNM | `96D21409-C607-11E4-A961-68F728A0C9CC` |

## `emerald` carries no SSH listener, deliberately

**Correction, 2026-09-13: the section below describes the box as of 2026-08-16 and was
overtaken the next day.** OpenSSH Server was enabled on `emerald` on 2026-08-17 for the
time-critical pre-AVX2 sweep, hardened the same way as the other nodes, and the sweep ran
over it on 2026-08-18; the listener was still recorded running on 2026-08-20.[^12] The
Ubuntu plan described here was not carried out before the box went offline around
2026-08-19, and its current state is unverifiable from the tailnet. The rest of this
section is kept as the record of why no listener was planned, which is still the reason
the roster carries it disabled.

It is reachable — `tailscale ping` answers in 3 ms — but port 22 is closed and nothing is
listening.[^7] That is the plan, not a gap: the box is being reimaged to Ubuntu Server,
where `tailscale up --ssh` supplies SSH from the tailnet daemon and OpenSSH is never
installed. Adding a Windows listener now would be work thrown away at the reformat.

The remaining blocker is **Ubuntu boot media**, not power and not physical access. The
only USB device attached is the stick running the session that surveyed it, and reflashing
that would destroy the session.[^6]

Two properties of this box shape what may be measured on it:[^6]

- **Core count is reported three different ways.** The marketing name says "12 Compute
  Cores 4C+8G"; Windows reports `NumberOfCores=2` with 4 logical processors. It is a
  2-module, 4-integer-core Steamroller part. Any CPU comparison against the i7-11700K must
  state which convention it counted, or the result is uninterpretable.[^6]
- **Storage is a 22,892-hour spinning disk** (Seagate ST2000DM001, ~130–140 MB/s
  sequential). Instruction-set determinism does not care, but no wall-clock figure taken
  here is comparable to `sedona` or `lavender` unless the storage is named alongside it.[^6]

## Trap 1: the account is `austi`, and the error does not say so

SSH defaults to the caller's local username. The account on every node is `austi`; the
account on `austinpc` is `Test`. Windows reports an unknown user with the **identical**
string it uses for an unauthorised key:[^2]

```
Permission denied (publickey,password,keyboard-interactive)
```

This cost hours.[^2] It was diagnosed as "no sshd installed" and then "the key was never
placed", reported as such, and acted on — while the key had been correct the whole time.
The aliases pin `User austi` so the failure cannot recur; the reasoning is in the config
file's own comments so it survives someone rewriting it.

## Trap 2: the remote shell is `cmd.exe`, and failure looks like success

These are Windows hosts. A remote command piped through `tail`, `head` or `grep` dies
before the command runs — and `ssh` still exits 0, because that exit code reports the
transport, not the remote process. A `pip install` that never executed reported success.[^3]

Keep remote commands pure PowerShell or `cmd` and shape output locally.[^3] For anything
beyond one line, `scp` a script file and run it: quoting a PowerShell command inside an
SSH command line inside a local shell fails on its own in a third distinct way.

## Trap 3: the tailnet name and the Windows name disagree

A Tailscale rename takes effect immediately; a Windows rename does not. Both `lavender`
and `emerald` are in the same two-stage state: `Rename-Computer` succeeded and the
registry's `ComputerName` already reads the new name, while `ActiveComputerName` — and
therefore `hostname` and `$env:COMPUTERNAME` — still reads the old one until the machine
reboots.[^4][^6]

So anything identifying a host by `hostname` sees the old name[^4]; anything using the tailnet
sees the new one, and the mismatch resolves itself at the next restart. Record results
against the tailnet alias and correlate by the identity triple above.

Both boxes have a reboot pending for unrelated reasons as well (staged Windows updates),
so the rename completes whenever either is next restarted. One caution, since both were
surveyed from the portable stick: **a reboot is a hard kill of a stick session**, which
loses everything since its last clean launch. Exit cleanly first.[^6]

## A bare Windows node needs the Visual C++ runtime before mujoco will import

`mujoco.dll` fails to load on a clean machine with `FileNotFoundError: Could not find module
… (or one of its dependencies)`, which names the wrong file and does not name the missing
dependency at all. The cause is an absent Visual C++ runtime: `lavender` had no
redistributable installed and none of `vcruntime140.dll`, `vcruntime140_1.dll` or
`msvcp140.dll` present.[^8]

Install it explicitly — `winget install --id Microsoft.VCRedist.2015+.x64 --source winget`
— as a prerequisite step, not as a reaction to the error.[^8] Machines that have run games or
Visual Studio already have it and will not show the problem, which is exactly why it is easy
to omit from a setup recipe written on such a machine.

Two further `winget` notes from the same install: its `msstore` source can fail with
`0x8a15005e : The server certificate did not match any of the expected values`, which aborts
the whole command even when the package is available elsewhere — pass `--source winget` to
pin it. And the Microsoft Store `python.exe` stub on `$PATH` reports "Python was not found"
rather than behaving as an absent command, so probe with `python --version` and read the
output, not with a bare existence check.[^8]

## Toolchains belong on local disk

Install to the node's own disk, never through a portable USB stick that remaps `HOME`. A
prior Warp install lived under a stick's remapped home directory and vanished with it,
leaving only an orphaned kernel cache — the measurement it produced was real when taken
and impossible to reproduce afterwards.[^5] A stick is a bootstrap for a machine with no
`sshd` yet. Once SSH exists it should leave.

## Firewall scoping is per-node and deliberately unequal

**The asymmetry this section described is gone, and the heading now overstates it.**
Measured 2026-09-04, both nodes scope inbound SSH to `austinpc` alone over the tailnet:
`sedona` admits `100.77.206.124`; `lavender` admits `100.77.206.124` plus its own LAN
(`192.168.10.0/24`) and loopback.[^11]

The page previously said `lavender` admitted the whole tailnet range (`100.64.0.0/10`) and
concluded that "a session on any other machine can reach `lavender` and cannot reach
`sedona`". That is no longer true: no third machine reaches either node over Tailscale. The
narrowing was deliberate — it is the fleet-wide sshd/Tailscale hardening pass of 2026-08 —
so the rule to carry forward is unchanged in spirit and stronger in fact: **reach both
nodes from `austinpc`**, and widening either scope is a deliberate act rather than a
troubleshooting step.[^11]

`lavender`'s two extra entries are not a tailnet exception. `192.168.10.0/24` is the
physical LAN both boxes sit on, so it admits a machine on the same wire, not a machine on
the tailnet.[^11]

Footnotes below cite `wiki/log.md` by its dated `##` heading, never by line number.
Four of them cited line numbers until 2026-09-04 and all four were wrong: the log is
append-at-TOP, so every new entry shifts every citation into it, and three of the four
pointed at content that is not in `log.md` at all. A line number into a newest-first log
is a citation that decays on a schedule. The `[observed]` clauses carry the evidence and
were unaffected.[^11]

[^1]: `~/.ssh/config:14-27` on `austinpc`, `Host sedona` / `Host lavender` blocks — `User austi`, `IdentityFile ~/.ssh/id_ed25519`. `[observed]` — `ssh sedona hostname` returns `sedona` and `ssh lavender hostname` returns `DESKTOP-JAHBOEJ`, both under `-o BatchMode=yes`, which disables every interactive and password path and so proves publickey authentication alone.
[^2]: `~/.ssh/config:16` (`User austi`, which is why other usernames are refused) — `[observed]` — `ssh Test@100.95.76.122` and `ssh austin@100.85.214.124` both return `Permission denied (publickey,password,keyboard-interactive)`, the same string returned by a host with no authorised key. `ssh austi@100.95.76.122` succeeds against the same daemon.
[^3]: `wiki/log.md` § `[2026-08-15]` (the entry that wrote this page; "Three traps documented because each produced a wrong conclusion first") — `[observed]` — `ssh austi@100.95.76.122 "…pip install… 2>&1 | tail -20"` produced only `'tail' is not recognized as an internal or external command` while the local `ssh` process exited 0. Re-running without the pipe installed warp-lang 1.16.0, mujoco 3.11.0 and mujoco-warp 3.11.0.
[^4]: `[observed]` — `tailscale status` lists `100.85.214.124 lavender`, while `ssh lavender "powershell -NoProfile -Command $env:COMPUTERNAME"` returns `DESKTOP-JAHBOEJ`. The staged-rename detail, the `Rename-Computer` `HasSucceeded: True` result, the `ComputerName`/`ActiveComputerName` registry split and the MTM/serial/UUID are reported by the session running on that host (agent board, `opus-portaclaude-0815`, 2026-08-16T01:57:20Z); this session verified only the two commands above.
[^5]: Recorded into this page by `wiki/log.md` § `[2026-08-15]` ("Pages written: measurement-fleet-is-reachable-by-ssh-alias"), so observed on or before 2026-08-16; the exact observation instant is not in the trail and is not invented here. — `[observed]` — on `sedona`, `pip show warp-lang` reported `Package(s) not found` and a filesystem sweep found only Microsoft Store Python 3.11.9, while `E:\home\AppData\Local\NVIDIA\warp` still held a 16 MB kernel cache from the vanished install.
[^6]: Agent board, `opus-emerald-0816`, 2026-08-16T03:17:53Z and 05:20:56Z — surveyed locally on that host from the portable stick, not over the network. Reports the rename's two-stage state, the identity triples, `NumberOfCores=2` with 4 logical processors against the "12 Compute Cores 4C+8G" marketing name, the Seagate ST2000DM001 at 22,892 power-on hours, the absent Ubuntu boot media, and the pending reboots. The same session's 03:47:40Z post retracts its own initial 24.1 MB/s sequential-write figure as contaminated by load it had itself created; the corrected value is ~130–140 MB/s. This session verified none of these directly — see [^7] for what it did verify.
[^7]: Recorded into this page by `wiki/log.md` § `[2026-08-15]`, so observed on or before 2026-08-16. — `[observed]` — from `austinpc`: `tailscale ping 100.125.152.27` returns `pong from emerald (100.125.152.27) … in 3ms`, while a TCP connect to port 22 times out and `tailscale status` lists the node as `emerald`.
[^8]: Recorded into this page by `wiki/log.md` § `[2026-08-15]`, so observed on or before 2026-08-16. The failure is not re-observable now: the `winget` install below fixed it, so this footnote is a historical record rather than a reproducible check. — `[observed]` — on `lavender`, importing mujoco raised `FileNotFoundError: Could not find module 'C:\navprobe\.venv\Lib\site-packages\mujoco\mujoco.dll' (or one of its dependencies)`; `Test-Path` returned false for all three of `C:\Windows\System32\vcruntime140.dll`, `vcruntime140_1.dll` and `msvcp140.dll`, and no `Visual C++` entry existed in either Uninstall key. After `winget install --id Microsoft.VCRedist.2015+.x64 --source winget`, `import warp, mujoco, mujoco_warp` printed `1.16.0 3.11.0`. The same `winget` invocation without `--source winget` had failed with `0x8a15005e` while updating the `msstore` source.
[^9]: `[observed]` — 2026-09-04, from `austinpc` over the alias: `ssh lavender "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader"` returned `NVIDIA GeForce GTX 1630, 591.86, 4096 MiB, 7.5, 3, 16`, and `nvidia-smi -q` reported `Product Architecture : Turing`. `Get-CimInstance Win32_VideoController` on the same host lists BOTH the GTX 1630 and the Intel UHD Graphics 750, which is why the original row's "Intel UHD 750 only" was a true observation of one adapter read as a complete inventory. Driver 591.86 is the same version `austinpc` runs on its RTX 3090 Ti. **Not verified here**: that a CUDA-capable torch or a working warp build exists on `lavender` — `nvidia-smi` answering proves the driver, not the toolchain, and as of 2026-09-04 that host carries a Python interpreter with no poetry, git or make (`fleet-bootstrap`, commit b277db16).
[^10]: Agent board, task `a9a700c3`, 2026-09-04T12:52:41Z — `opus-weight-injection-0902` reports reading this page while probing the fleet, finding it said one CUDA architecture, and posting the GTX 1630 as a new side finding on task `df6f1dc8`; the same session retracted that framing at 12:52 on discovering the 2026-08-24 measurement. The stale row is named there by file and line. `[observed]` — the row it names is the one this page's 2026-09-04 edit replaced.
[^11]: `[observed]` — 2026-09-04, from `austinpc`: `ssh <node> 'powershell -NoProfile -Command "Get-NetFirewallRule -DisplayName *SSH* | Where-Object Enabled -eq True | Where-Object Direction -eq Inbound | Get-NetFirewallAddressFilter | Select-Object -ExpandProperty RemoteAddress"'` returned `100.77.206.124` on `sedona`, and `100.77.206.124` + `192.168.10.0/255.255.255.0` + `127.0.0.1` on `lavender`. Two earlier command shapes failed first and both are instances of this page's own Trap 2: a `-f` format operator lost its argument list through the SSH quoting layers, and `-Enabled True -Direction Inbound` as named parameters could not resolve a parameter set. Filtering with `Where-Object` after an unfiltered `Get-NetFirewallRule` is the form that survives the hop.
[^12]: Agent board, task `3027abcf`, three rows: `opus-rog-ally-0816` status change 2026-08-17T17:55:34Z ("sshd Running/Automatic at 100.125.152.27 (user austi), hardened per fable-tankpit-0817 (PasswordAuthentication/PermitEmptyPasswords/KbdInteractiveAuthentication all no; scoped allow 100.77.206.124)"); task `311f2945`, `opus-gpu-nodes-0815` note 2026-08-18T07:08:50Z ("Ran REMOTELY over SSH, not on the stick"); `opus-portable-claude-0817` note 2026-08-20T07:50:40Z ("emerald sshd Running/Automatic, rule sshd-austinpc-only, remote=100.77.206.124, installer's OpenSSH-Server-In-TCP deleted after install"). The offline date is the roster note in `~/PROJECTS/MCPs/fleet-mcp/fleet-nodes.json` (`enabled: false`, "Offline since roughly 2026-08-19"), and `tailscale status` on 2026-09-12 read "last seen 24d ago" for it (`opus-serendipity-0912`, same task, 2026-09-13T02:23:30Z). Not re-observed: the box did not answer on 2026-09-13.
[^13]: `[observed]` — 2026-09-13, from `austinpc` over the alias `lenovoold` (`~/.ssh/config`: `Host lenovoold`, `HostName 100.96.79.34`, `User austi`): `ssh lenovoold hostname` returns `LenovoOld`; `(Get-CimInstance Win32_Processor).Name` returns `Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz`; `IsProcessorFeaturePresent` by PF code returns SSE3, SSE4_1, SSE4_2, AVX, AVX2 true and AVX512F false; `Win32_ComputerSystem` Manufacturer `LENOVO`, Model `80JH`; `Win32_OperatingSystem.BuildNumber` `19045`; `TotalPhysicalMemory` 7.9 GB; identity triple `Win32_ComputerSystemProduct.Name` `80JH`, `Win32_BIOS.SerialNumber` `PF04TWNM`, `Win32_ComputerSystemProduct.UUID` `96D21409-C607-11E4-A961-68F728A0C9CC`. The listener refuses password and keyboard-interactive on the wire (`Permission denied (publickey)` with `PreferredAuthentications=password,keyboard-interactive`; `opus-serendipity-probe-0912`, board task `3027abcf`, 2026-09-13T19:41:41Z). Toolchain installed this day at `C:\navprobe\.venv` (Python 3.11.9; warp-lang 1.16.0, mujoco 3.11.0, mujoco-warp 3.11.0, numpy 2.4.6, all `==`-pinned) and the CPU control sweep ran on it, 10/10 digests matching: [[cpu-determinism-is-bit-portable-across-x86-vendors]]. `API/tools/fleet/fleet.json` marks it `not_dispatchable` (8 GB, end-of-life OS); that hold governs CI dispatch, not a manual measurement over ssh, and was worded to say so in API `36316ccb`.
