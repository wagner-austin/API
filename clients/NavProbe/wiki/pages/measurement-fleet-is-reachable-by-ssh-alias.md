---
title: Every measurement node is reachable by SSH alias, and three traps make it look otherwise
tags: [platform, fleet, ssh, tailscale, operations]
related: ["[[mjwarp-cannot-compile-under-warp-deterministic-mode]]", "[[jax-cuda-unavailable-on-windows]]", "[[open-questions-and-what-would-answer-them]]"]
provenance:
  - "~/.ssh/config"
  - "tailscale status"
  - "agent board opus-emerald-0816 2026-08-16"
fact_checked: 2026-08-16
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
| `lavender` | 100.85.214.124 | i7-11700K (AVX-512), Intel UHD 750 only | CPU node — no discrete GPU |
| `emerald` | 100.125.152.27 | A10-7800 Steamroller — **pre-AVX2**, iGPU only | CPU control; **no sshd by design** |

`austinpc` itself holds the RTX 3090 Ti (sm_86, 84 SMs). Both CUDA devices in the fleet
are therefore Ampere, which is why the second-architecture question in
[[open-questions-and-what-would-answer-them]] cannot be answered by any machine currently
owned — and why the 46-vs-84 SM comparison is what `sedona` actually supports.

Nodes are renamed to a colour/place scheme as they are commissioned. `lavender` was
`desktop-jahboej`; `emerald` was `desktop-li867ht`. Because a Tailscale rename is instant
and a Windows rename is not (below), correlate across a rename by the stable identity
triple rather than by any name:[^6]

| node | MTM | serial | UUID |
|---|---|---|---|
| `lavender` | 90RJ0080US | MJ0GVS2H | `40F89E80-8561-11EC-A78F-1411F17D4C00` |
| `emerald` | 90BG003JUS | R302F9AZ | `2428F0B8-28F9-11E6-B358-182A98911300` |

## `emerald` carries no SSH listener, deliberately

It is reachable — `tailscale ping` answers in 3 ms — but port 22 is closed and nothing is
listening.[^7] That is the plan, not a gap: the box is being reimaged to Ubuntu Server,
where `tailscale up --ssh` supplies SSH from the tailnet daemon and OpenSSH is never
installed. Adding a Windows listener now would be work thrown away at the reformat.

The remaining blocker is **Ubuntu boot media**, not power and not physical access. The
only USB device attached is the stick running the session that surveyed it, and reflashing
that would destroy the session.[^6]

Two properties of this box shape what may be measured on it:

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

This cost hours. It was diagnosed as "no sshd installed" and then "the key was never
placed", reported as such, and acted on — while the key had been correct the whole time.
The aliases pin `User austi` so the failure cannot recur; the reasoning is in the config
file's own comments so it survives someone rewriting it.

## Trap 2: the remote shell is `cmd.exe`, and failure looks like success

These are Windows hosts. A remote command piped through `tail`, `head` or `grep` dies
before the command runs — and `ssh` still exits 0, because that exit code reports the
transport, not the remote process. A `pip install` that never executed reported success.[^3]

Keep remote commands pure PowerShell or `cmd` and shape output locally. For anything
beyond one line, `scp` a script file and run it: quoting a PowerShell command inside an
SSH command line inside a local shell fails on its own in a third distinct way.

## Trap 3: the tailnet name and the Windows name disagree

A Tailscale rename takes effect immediately; a Windows rename does not. Both `lavender`
and `emerald` are in the same two-stage state: `Rename-Computer` succeeded and the
registry's `ComputerName` already reads the new name, while `ActiveComputerName` — and
therefore `hostname` and `$env:COMPUTERNAME` — still reads the old one until the machine
reboots.[^4][^6]

So anything identifying a host by `hostname` sees the old name; anything using the tailnet
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
— as a prerequisite step, not as a reaction to the error. Machines that have run games or
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

`sedona` admits only `austinpc` (`100.77.206.124/32`); `lavender` admits the tailnet range
(`100.64.0.0/10`). A session on any other machine can reach `lavender` and cannot reach
`sedona`. That asymmetry is intentional and widening it should be a deliberate act, not a
troubleshooting step.

[^1]: `~/.ssh/config` on `austinpc`, `Host sedona` / `Host lavender` blocks — `User austi`, `IdentityFile ~/.ssh/id_ed25519`. `[observed]` — `ssh sedona hostname` returns `sedona` and `ssh lavender hostname` returns `DESKTOP-JAHBOEJ`, both under `-o BatchMode=yes`, which disables every interactive and password path and so proves publickey authentication alone.
[^2]: `[observed]` — `ssh Test@100.95.76.122` and `ssh austin@100.85.214.124` both return `Permission denied (publickey,password,keyboard-interactive)`, the same string returned by a host with no authorised key. `ssh austi@100.95.76.122` succeeds against the same daemon.
[^3]: `[observed]` — `ssh austi@100.95.76.122 "…pip install… 2>&1 | tail -20"` produced only `'tail' is not recognized as an internal or external command` while the local `ssh` process exited 0. Re-running without the pipe installed warp-lang 1.16.0, mujoco 3.11.0 and mujoco-warp 3.11.0.
[^4]: `[observed]` — `tailscale status` lists `100.85.214.124 lavender`, while `ssh lavender "powershell -NoProfile -Command $env:COMPUTERNAME"` returns `DESKTOP-JAHBOEJ`. The staged-rename detail, the `Rename-Computer` `HasSucceeded: True` result, the `ComputerName`/`ActiveComputerName` registry split and the MTM/serial/UUID are reported by the session running on that host (agent board, `opus-portaclaude-0815`, 2026-08-16T01:57:20Z); this session verified only the two commands above.
[^5]: `[observed]` — on `sedona`, `pip show warp-lang` reported `Package(s) not found` and a filesystem sweep found only Microsoft Store Python 3.11.9, while `E:\home\AppData\Local\NVIDIA\warp` still held a 16 MB kernel cache from the vanished install.
[^6]: Agent board, `opus-emerald-0816`, 2026-08-16T03:17:53Z and 05:20:56Z — surveyed locally on that host from the portable stick, not over the network. Reports the rename's two-stage state, the identity triples, `NumberOfCores=2` with 4 logical processors against the "12 Compute Cores 4C+8G" marketing name, the Seagate ST2000DM001 at 22,892 power-on hours, the absent Ubuntu boot media, and the pending reboots. The same session's 03:47:40Z post retracts its own initial 24.1 MB/s sequential-write figure as contaminated by load it had itself created; the corrected value is ~130–140 MB/s. This session verified none of these directly — see [^7] for what it did verify.
[^7]: `[observed]` — from `austinpc`: `tailscale ping 100.125.152.27` returns `pong from emerald (100.125.152.27) … in 3ms`, while a TCP connect to port 22 times out and `tailscale status` lists the node as `emerald`.
[^8]: `[observed]` — on `lavender`, importing mujoco raised `FileNotFoundError: Could not find module 'C:\navprobe\.venv\Lib\site-packages\mujoco\mujoco.dll' (or one of its dependencies)`; `Test-Path` returned false for all three of `C:\Windows\System32\vcruntime140.dll`, `vcruntime140_1.dll` and `msvcp140.dll`, and no `Visual C++` entry existed in either Uninstall key. After `winget install --id Microsoft.VCRedist.2015+.x64 --source winget`, `import warp, mujoco, mujoco_warp` printed `1.16.0 3.11.0`. The same `winget` invocation without `--source winget` had failed with `0x8a15005e` while updating the `msstore` source.
