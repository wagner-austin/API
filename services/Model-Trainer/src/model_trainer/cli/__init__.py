"""Command-line entry points that run a measurement without the service.

The API, redis and the RQ worker are one way to reach the scorer, and they
are the wrong way on a cluster: a Slurm compute node has no service stack to
talk to, and starting one to score 2,627 items would be most of the work.
These entries call the same scoring code in-process.

That also removes a failure this project has already hit. The worker runs two
RQ workers, so two scoring jobs enqueued together both grab the GPU, and the
first attempt at re-scoring the gpt2 baseline died with an illegal CUDA
memory access for exactly that reason. An in-process run has nothing to race.
"""

from __future__ import annotations

__all__: list[str] = []
