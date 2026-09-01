# OrderedKernels

Fast fixed-order GEMM kernels: cross-GPU **bit-identical** matmuls at
tiled-kernel speed. The reduction order is owned by the program (strictly
ascending k per output element, separate multiply and add, `--fmad=false`),
so the arithmetic matches Model-Trainer's `rank1` instrument bit for bit —
the record corpus spanning seven GPUs is this package's oracle — while
shared-memory tiling recovers the speed the instrument gave away.

Kernels are CUDA-C strings compiled at runtime through cupy/NVRTC. That is a
deliberate answer to the coverage problem that deferred Triton: a kernel
STRING is data, every Python line here executes for real, and the suite
holds the workspace's 100% bar honestly.

**The test suite REQUIRES a CUDA device** (NavProbe precedent). It refuses
at session start without one; it does not skip. Run `make check` on a GPU
machine, and check `nvidia-smi` for other tenants first — host suites share
the card.

CLIs (records interoperable with the abl experiment's, label arm `ordered`):

- `python -m ordered_kernels.cli.gemm_probe --device cuda --out r.json`
- `python -m ordered_kernels.cli.train_step --device cuda --rungs tiny,medium,large --out r.json`
- `python -m ordered_kernels.cli.bench --device cuda --out b.json`
