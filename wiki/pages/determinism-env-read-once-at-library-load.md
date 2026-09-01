---
title: Reduction order is an environment variable read once — and setting it late fails in silence
tags: [determinism, reproducibility, cuda, cublas, blas, platform-core, infrastructure]
related:
  - "[[monorepo-discipline]]"
  - "[[covenant-radar-backend-registry]]"
source_paths:
  - libs/platform_core/src/platform_core/determinism_env.py
source_git_blobs:
  "libs/platform_core/src/platform_core/determinism_env.py": ce7115afeaf000faed9d47d84b020c2daf111424
fact_checked: "2026-09-01"
confidence: high
hubs: [libs, infrastructure, clients]
---

# Reduction order is an environment variable read once

Floating-point addition is not associative, so a reduction's **order** decides
its bits. On both GPU and CPU that order is chosen by a native library reading
an environment variable **once, when it loads**. Two consequences follow, and
`platform_core/determinism_env.py` exists to hold both.[^module]

**Setting the variable late is accepted in silence.** Not an error, not a
warning — the library already made its choice. Measured, not assumed: two
`addmm` calls with `CUBLASLT_WORKSPACE_SIZE` set between them **both still used
split-K, 2 of 2**.[^lt] The only safe moments are before the process starts, or
before it touches CUDA.

**So the string itself is shared infrastructure.** It lives in `platform_core`,
not beside the code that applies determinism, because two things in different
dependency tiers need the identical literal: `platform_ml.determinism` sets it
in-process for a trainer about to run, and a **job submitter writes it into a
batch script** — and a submitter runs on a laptop that must not import
torch.[^module] A duplicated literal would be the worst kind: the copies drift,
nothing fails, and the runs quietly stop being comparable.

## Two CUDA libraries, two handles, neither substitutes for the other

`CUBLAS_WORKSPACE_CONFIG=":4096:8"` governs cuBLAS.[^cublas]
`CUBLASLT_WORKSPACE_SIZE` governs **cuBLASLt**, a different library with a
different handle — and it is the path a fused-epilogue matmul takes, meaning
`addmm`, and therefore **every `nn.Linear` that has a bias**.[^lt] Configuring
one and assuming the other is covered is the easy mistake.

The cuBLAS pairing is safe to rely on for a reason worth copying:
`torch.use_deterministic_algorithms(True)` **raises** a `RuntimeError` naming the
variable when it is absent, so a run configured for determinism without it fails
loudly instead of producing quietly non-reproducible numbers.[^cublas]

### Why zero workspace removes split-K

Split-K partitions the summed dimension across thread blocks and recombines the
partials through a scratch workspace. It is what makes a long reduction fast,
and it is also what makes the reduction order depend on **how many partitions the
heuristic chose — which depends on the card**. A zero-size workspace leaves
nowhere to recombine partials, so the heuristic cannot choose a split, and the
reduction runs in one order on every card that offers the non-split kernel.[^lt]

Measured 2026-08-27, driver 580.82.07, torch 2.6.0+cu124: with this set, three
cards produce **bit-identical tensors on all eight probed GEMM shapes**, and an
A100 and an A30 agree on **1,017 of `xl`'s 1,018 traced tensors**. Cost is
nothing above 128 rows and up to **+85% at 64** — so a real training step, whose
row count is batch × sequence length, pays nothing.[^lt]

**What it does not buy: attention.** Not one of 72 measured SDPA digests moves,
because the memory-efficient kernel is not a cuBLASLt call. A run with this set
is comparable across cards for its matmuls and still is not for its model. That
stated limit is why this is one control among several rather than a fix.[^lt]

## The CPU is the same shape

A multi-threaded BLAS splits a reduction across threads, and the thread **count**
decides the partitioning. Measured 2026-08-25 (numpy 2.3.5 / scipy-openblas
0.3.30, 24 cores): at a fixed thread count the result is bit-identical run after
run; across 1, 8 and 24 threads it is not — **three different answers from
identical bytes**.[^blas]

The conclusion drawn there is the transferable one: *the hazard is an unrecorded
**input**, not an unpredictable library.* All four thread variables are named —
`OMP`, `OPENBLAS`, `MKL`, `NUMEXPR` — because a numpy wheel may link against
OpenBLAS or MKL and numexpr reads its own; setting only the one that matters
today leaves the record claiming a posture the next wheel will not honour.[^blas]

## Refusing to guess the posture

`determinism_requested` treats an **absent** `TRAIN_DETERMINISTIC` as True
(determinism is the platform default, and the local worker predating any launcher
must keep behaving as it did), but a present-and-unrecognised value raises rather
than falling back.[^requested] The docstring gives the reason both guesses are
wrong: guessing "on" wastes wall clock on a run the operator wanted fast, and
guessing "off" **produces a run recorded as deterministic that is not — the exact
failure the variable exists to prevent**.

The variable is also deliberately not named for any cluster: a worker running
locally in Docker reading `HPC3_DETERMINISTIC` would be plainly wrong, and would
be set wrong.[^requested]

## `SetEnvProtocol` — a write-only seam, and why it has no read side

Production passes `os.putenv`, which reaches the real process environment that a
C library's `getenv` consults — the only environment cuBLAS or OpenBLAS reads.
The monorepo bans reading config out of `os.environ`; writing a variable a native
library requires is a different act, and the Protocol keeps the two from being
confused.[^proto]

There is deliberately no read side, and the reason is a trap worth remembering:
`os.putenv` **does not update `os.environ`**, so a "did it get set?" helper built
on the Python mapping would report False on a correctly configured process.[^proto]

[^module]: `libs/platform_core/src/platform_core/determinism_env.py:1-21` —
    module docstring: "two things in different dependency tiers need the exact
    same string and they must never disagree"; "A submitter runs on a laptop and
    must not depend on torch"; "A duplicated literal here would be the worst
    kind: the two copies would drift, nothing would fail, and the runs would
    silently stop being comparable."
[^cublas]: Same file, `:27-37` — `CUBLAS_WORKSPACE_ENV_VAR`,
    `CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"`, and the `RuntimeError`
    enforcement note.
[^lt]: Same file, `:40-80` — `CUBLASLT_WORKSPACE_ENV_VAR`,
    `CUBLASLT_NO_SPLIT_K = "0"`, the split-K mechanism, the 2026-08-27
    measurement block, the attention limit ("Not one of 72 measured SDPA digests
    moves"), and the late-set measurement ("two ``addmm`` calls with the variable
    set between them both still used split-K, 2 of 2").
[^blas]: Same file, `:139-177` — `BLAS_THREAD_ENV_VARS`, the 2026-08-25
    measurement, and `SINGLE_THREAD` ("A parallel reduction can be made
    reproducible only by fixing the split as well as the count, which no portable
    interface exposes").
[^requested]: Same file, `:83-136` — `DETERMINISM_ENV_VAR` docstring
    ("Deliberately not named for any cluster") and `determinism_requested`,
    whose `Raises:` section carries the refusal reasoning.
[^proto]: Same file, `:180-204` — `SetEnvProtocol` docstring.
