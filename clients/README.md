# clients/

Standalone applications, each a Poetry package with strict typing and 100% test
coverage. Three are **system-identification instruments** that infer the rules of
undocumented systems and prove the inferred model matches reality; the fourth is a
GPU kernel package the Model-Trainer measurements rely on.

## Clients

| Client | Description |
|--------|-------------|
| [TankpitBot](./TankpitBot) | System identification against an undocumented network protocol — CDP capture, XOR codec, live probes, provenance-carrying beliefs, archive-priced validators, server twin |
| [RustedWarfareBot](./RustedWarfareBot) | System identification against an obfuscated binary — JVM bytecode injection, build-pinned engine claims, doctrine-controlled experiments, seeded reproducible matches |
| [NavProbe](./NavProbe) | Reproducibility instrument for simulated navigation — seeded rollouts, canonical digests, determinism verdicts across MJX and MuJoCo-Warp |
| [OrderedKernels](./OrderedKernels) | Fixed-order GEMM kernels: cross-GPU bit-identical matmuls at tiled-kernel speed, matching Model-Trainer's `rank1` instrument bit for bit |

DiscordBot, the one client that consumed the monorepo's services, was archived on
2026-11-01 with the services it called, together with its guide to adding a service
client (`docs/adding-clients.md`). Both live on at the tag
`archive/api-packages-2026-11-01`.

### The instruments

TankpitBot and RustedWarfareBot solve the same problem from opposite ends: one
reverses a network protocol from the outside, the other instruments a runtime from
the inside. Both target systems that publish no spec, cannot be verified directly,
and drift without notice — so both invest more in *learning and checking* the system
than in acting on it: controlled probes, beliefs carrying confidence and provenance,
conformance validation against an archive, and claims pinned to the build or capture
they were measured on.

## Design Principles

All clients follow these conventions:

- **Strict typing**: No `Any`, `cast`, `type: ignore`, or `.pyi` stubs
- **100% test coverage**: Statement and branch coverage enforced

## Development

Each client has a Makefile with standard targets:

```bash
make lint   # Run guard checks, ruff, mypy
make test   # Run pytest with coverage
make check  # Run both lint and test
```

## Requirements

- Python 3.11+
- Poetry for dependency management
