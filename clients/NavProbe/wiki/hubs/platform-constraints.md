# Platform Constraints

What can and cannot be run where. Backend availability, wheel availability, and the interpreter version ceiling — the practical limits on which measurements are reachable from which machine.

These pages exist because a measurement that cannot be taken is a project constraint, not a gap to be quietly worked around. Recording the constraint with its evidence is what stops it being rediscovered.

[JAX has no CUDA backend on native Windows](../pages/jax-cuda-unavailable-on-windows.md) -- `jax-cuda12-plugin` publishes no Windows distribution; the WSL2 route was taken and works, and the instrument needed no change
[Python 3.11 caps jax, numpy and scipy below their latest releases](../pages/python-311-caps-scientific-stack.md) -- the newest releases require 3.12; the installed versions are the latest reachable
[Every measurement node is reachable by SSH alias](../pages/measurement-fleet-is-reachable-by-ssh-alias.md) -- the tailnet fleet, what hardware each node carries, why one of them deliberately has no SSH listener, and the three traps that make a working connection look broken
[MuJoCo's shipped binary requires AVX, so a pre-AVX host is ineligible](../pages/mujoco-requires-avx-so-pre-avx-hosts-are-ineligible.md) -- a hardware floor under the whole project: below AVX the library will not import at all, so screen for the feature flag before provisioning, not after
