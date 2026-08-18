---
title: MuJoCo's shipped binary requires AVX, so a pre-AVX host cannot be a measurement node at all
tags: [cpu, avx, platform, constraint, finding, mujoco, hardware-selection]
related: ["[[cpu-determinism-is-bit-portable-across-x86-vendors]]", "[[measurement-fleet-is-reachable-by-ssh-alias]]", "[[python-311-caps-scientific-stack]]"]
provenance:
  - "mujoco 3.11.0"
  - "warp-lang 1.16.0"
  - "numpy 2.4.6"
fact_checked: 2026-08-17
confidence: high
measured_with:
  package: mujoco 3.11.0
  host: surfacego — Intel Pentium, AVX absent
  installed: warp-lang 1.16.0, mujoco 3.11.0, mujoco-warp 3.11.0, numpy 2.4.6
  vcredist: present (vcruntime140.dll, vcruntime140_1.dll, msvcp140.dll all on disk)
hubs: [platform-constraints]
---

# MuJoCo's shipped binary requires AVX, so a pre-AVX host cannot be a measurement node at all

There is a hardware floor under this entire project, and it is lower and harder than the
determinism questions that sit above it. A machine whose processor lacks AVX cannot run
MuJoCo, cannot be provisioned into usefulness, and is permanently ineligible as a node — not
slow, not degraded, **unable to import the library**.

## The failure, and why it is not the usual one

On a correctly provisioned host with the exact package pins this project requires, the
install succeeds and the import does not:[^1]

```
Successfully installed  warp-lang-1.16.0  mujoco-3.11.0  mujoco-warp-3.11.0
>>> import mujoco
  File "...\site-packages\mujoco\__init__.py", line 38, in <module>
    ctypes.WinDLL(os.path.join(os.path.dirname(__file__), 'mujoco.dll'))
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed
```

The obvious suspect is the missing Visual C++ runtime described in
[[measurement-fleet-is-reachable-by-ssh-alias]], and it is **ruled out**: all three runtime
DLLs were present on disk and the redistributable had been installed before pip ran.[^1]

## The import differential is what makes it conclusive

| import | result |
|---|---|
| `numpy` 2.4.6 | OK |
| `warp-lang` 1.16.0 | OK |
| `mujoco` 3.11.0 | **WinError 1114** |

The two that succeed both adapt to the host. NumPy carries runtime SIMD dispatch and selects
an SSE path when AVX is absent; Warp is Python plus a JIT that compiles for the actual
processor. **MuJoCo ships a prebuilt `mujoco.dll`**, and a static initialiser issuing an AVX
instruction inside `DllMain` on a processor without AVX faults exactly as `1114`
describes.[^1]

Stated honestly as inference rather than measurement: nobody has disassembled `mujoco.dll`
to point at the offending instruction. What is measured is the three-way differential above,
`AVX = False`, and VCRedist present. The airtight version would be a disassembly or a
from-source build with AVX disabled.[^1]

## Two consequences that change how hardware is chosen

**Screen for AVX2 present, not for "old x86".** A cheap old machine can sit *below* the floor
and buy nothing at all. This is not a theoretical worry: it retired a node that had already
been fully provisioned, tailnet-joined and toolchain-installed before anyone tried the
import. The CPU check costs one command and belongs before the provisioning, not after.

**The trap is branding, not age.** Intel disables AVX on Pentium and Celeron parts for market
segmentation, so a Pentium contemporary with an AVX2-capable Core part still fails. Neither
the model year nor the product family predicts it — only the feature flag does.

## It bounds the portability claim rather than threatening it

[[cpu-determinism-is-bit-portable-across-x86-vendors]] establishes bit-identical digests
across AVX2 and AVX-512 hosts, and names AVX-without-AVX2 as its remaining hole. This page
is what makes that hole *final*: there is nothing below AVX to test, because nothing below
AVX runs. The reachable span is exactly AVX, AVX2 and AVX-512 — so one host with AVX and
without AVX2 closes the axis completely, and no further hardware could extend it.

## Application notes

Rebuilding MuJoCo from source with AVX disabled is possible — the opt-out is compile-time —
but it would not rescue a comparison. The digests produced by a differently-compiled binary
are not comparable to those from the shipped one, so the measurement it enabled would answer
a different question than the one being asked.[^1]

[^1]: Agent board, `opus-surfacego-0817`, 2026-08-17T08:59:43Z — measured locally on host `surfacego`. Reports the successful install at the required pins, the `WinError 1114` on `import mujoco`, the numpy/warp/mujoco import differential, `Test-Path` true for all three VC++ runtime DLLs, and the from-source opt-out being compile-time only. That session states the AVX-in-`DllMain` mechanism as its own inference and flags it as such; this page preserves that distinction. This session did not reproduce the failure independently — `surfacego` has no SSH listener and is unreachable from austinpc ([[measurement-fleet-is-reachable-by-ssh-alias]]).
