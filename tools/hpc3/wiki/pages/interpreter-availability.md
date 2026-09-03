---
title: The interpreter every project needs is not in `module avail python`
tags: [cluster-facts, environments, onboarding]
hubs: [cluster-facts]
related: ["[[environment-pins]]", "[[facts-are-code]]", "[[image-build-flow]]", "[[invariant-placement]]"]
source_paths:
  - "src/hpc3/clusters/hpc3.py"
  - "src/hpc3/core/env_probe.py"
  - "README.md"
  - "pyproject.toml"
source_git_blobs:
  "src/hpc3/clusters/hpc3.py": "e6fedebb13c20222c9269b158f0ebed7fbf84cc9"
  "src/hpc3/core/env_probe.py": "e83c330acd07bdb53dfdcc8fe1ee8a64de3af529"
  "README.md": "f52e3c3fc49ebeadf34f228748f207253ad726c0"
  "pyproject.toml": "feba6fe164021d49f8d3e9c2cba0117ab48ee75e"
provenance:
  - "module -t avail python on hpc3 login-i15, 2026-09-03: python/2.7.17, 3.8.0, 3.10.2, 3.14.3"
  - "/usr/bin/python3 -V on hpc3 login-i15, 2026-09-03: Python 3.9.25; which python3.11 finds nothing"
  - "module -t avail | grep conda on hpc3 login-i15, 2026-09-03: anaconda/{2020.07,2021.11,2022.05,2024.06,2025.12}, miniconda3/{23.5.2,24.9.2}, mamba/{24.3.0,26.1.0}, bioconda/4.8.3"
  - "/pub/wagnera3/envs/{abl-pinned,cleargbm,tankpit}/bin/python -V, 2026-09-03: all Python 3.11.16"
  - "/pub/wagnera3/envs/tankpit/pyvenv.cfg read 2026-09-03: home = /pub/wagnera3/envs/cleargbm/bin, version 3.11.16"
  - "ls -l /pub/wagnera3/envs/tankpit/bin/python3.11, 2026-09-03: symlink to /pub/wagnera3/envs/cleargbm/bin/python3.11"
  - "sys.version on that interpreter, 2026-09-03: '3.11.16 | packaged by conda-forge | (main, Aug 21 2026, 22:44:51) [GCC 14.4.0]'; sys.base_prefix is /pub/wagnera3/envs/cleargbm"
fact_checked: 2026-09-03
confidence: high
---

# The interpreter every project needs is not in `module avail python`

`module -t avail python` answers with four interpreters and **none of them is
the one anything here runs on**:

```
python/2.7.17   python/3.8.0   python/3.10.2   python/3.14.3
```

The system interpreter is `Python 3.9.25`, and `which python3.11` finds
nothing.[^1] Meanwhile every project in this monorepo declares `python = "^3.11"`
— TankpitBot, RustedWarfareBot, Model-Trainer, `platform_core`, and
`tools/hpc3` itself.[^2]

So the interpreter this whole stack requires exists on that cluster in **no
form the obvious command will show you**: not as a `python` module, not as a
system binary. A new project's first act is therefore to discover that its
first act cannot be `module load python`.

## The answer is real, and it is filed under a different name

3.11 is available. It is not under `python`:[^3]

```
anaconda/{2020.07,2021.11,2022.05,2024.06,2025.12}
miniconda3/{23.5.2,24.9.2}
mamba/{24.3.0,26.1.0}
```

Every environment on this cluster was built through that door.
`/pub/wagnera3/envs/abl-pinned`, `/pub/wagnera3/envs/cleargbm` and
`/pub/wagnera3/envs/tankpit` all report `Python 3.11.16`, and the interpreter
names its own origin when asked:[^4]

```
3.11.16 | packaged by conda-forge | (main, Aug 21 2026, 22:44:51) [GCC 14.4.0]
```

**This is the whole cliff, and it is one sentence long**: search for `python`
and the cluster tells you 3.11 does not exist; search for `conda` and it does.
A grep of the README finds no mention of Python versions and none of
bootstrapping, so nothing closes that gap for the next reader.[^5]

It is also why *every registered project ships an image* is load-bearing
rather than stylistic. The image is where a pinned 3.11 comes from once the
project is running; conda is where it comes from before the image exists.

## The bootstrap left a dependency nobody declared

`envs/tankpit` is not a conda environment. It is a **venv whose interpreter is
a symlink into another project's environment**, which its own `pyvenv.cfg`
records:

```
home = /pub/wagnera3/envs/cleargbm/bin
executable = /dfs6b/pub/wagnera3/envs/cleargbm/bin/python3.11
command = /pub/wagnera3/envs/cleargbm/bin/python3.11 -m venv /pub/wagnera3/envs/tankpit
```

`bin/python3.11` is a symlink to `envs/cleargbm/bin/python3.11`, and
`sys.base_prefix` is `/pub/wagnera3/envs/cleargbm`. **Deleting or moving the
cleargbm environment breaks the tankpit one**, and nothing in either project's
run document says the two are related.[^6]

This is not carelessness by whoever did it — it is the only move available
when the first environment has no supported path. An improvised bootstrap
leaves an undeclared edge between projects, and that edge is invisible to
every check the package has: `check_env_path` proves the directory exists,
`verify_env_packages` proves the packages are pinned, and both keep passing
right up until the day the other project is cleaned up.

## What this page is not

**This fact is not yet code.** `src/hpc3/clusters/hpc3.py` mentions neither
`python` nor `interpreter` — measured 2026-09-03.[^7] Under [[facts-are-code]] a
cluster's interpreter inventory is exactly the kind of thing that belongs in
the cluster module rather than in prose, alongside its partitions and its QOS
ceilings, so that a rule can ask it instead of a person remembering this page.

Two things would follow from putting it there, and neither is done:

- A project could declare the interpreter it requires the way it already
  declares `pinned_packages`, and the existing probe round trip could hold the
  environment to it. `env_probe` already runs the environment's own
  interpreter; it asks that interpreter for its distributions and never for
  its version.[^8]
- The undeclared `envs/tankpit` → `envs/cleargbm` edge above would become
  visible, because a venv's `base_prefix` is readable by the same probe.

Recorded here rather than built, because the measurement is worth having
whether or not the rule ever lands, and a page that claims a rule exists when
it does not is worse than no page.

[^1]: `module -t avail python`, `/usr/bin/python3 -V` and `which python3.11`, run over SSH on hpc3 login-i15 at 2026-09-03 19:58 UTC. Output verbatim: `python/2.7.17`, `python/3.8.0`, `python/3.10.2`, `python/3.14.3`; `Python 3.9.25`; `no python3.11 in (/opt/rcic/bin:/usr/share/Modules/bin:/usr/local/bin:/usr/bin:...)`.
[^2]: The `python = "^3.11"` line in each of `clients/TankpitBot/pyproject.toml`, `clients/RustedWarfareBot/pyproject.toml`, `services/Model-Trainer/pyproject.toml`, `libs/platform_core/pyproject.toml` and `tools/hpc3/pyproject.toml`, read 2026-09-03.
[^3]: `module -t avail 2>&1 | grep -i -E "conda|mamba|miniforge|anaconda"` on hpc3 login-i15 at 2026-09-03 20:12 UTC, returning the eleven modules listed above. The same host's `module -t avail python` (footnote 1, fourteen minutes earlier) lists none of them, which is why searching by the obvious name answers "no 3.11".
[^4]: `for e in /pub/wagnera3/envs/*/; do "$e/bin/python" -V; done` on hpc3 login-i15 at 2026-09-03 20:11 UTC: `/pub/wagnera3/envs/abl-pinned/`, `/pub/wagnera3/envs/cleargbm/` and `/pub/wagnera3/envs/tankpit/` each report `Python 3.11.16`. The first two carry `/pub/wagnera3/envs/<name>/conda-meta/`; the third does not, and is a venv (see below).
[^5]: `grep -n "3\.11\|python_version\|module load\|module avail" README.md` and `grep -n -i bootstrap README.md`, both empty, 2026-09-03.
[^6]: `ls -l /pub/wagnera3/envs/tankpit/bin/python3.11` at 2026-09-03 20:12 UTC → `lrwxr-xr-x ... -> /pub/wagnera3/envs/cleargbm/bin/python3.11` (42-byte link, dated Sep 2 22:15); `sys.base_prefix` read from that same interpreter in the same call returns `/pub/wagnera3/envs/cleargbm`. The venv resolves today, so this is a latent dependency rather than a current breakage.
[^7]: `grep -n -i "python\|interpreter" src/hpc3/clusters/hpc3.py` → no matches, 2026-09-03.
[^8]: `src/hpc3/core/env_probe.py`, `_PROBE_SOURCE`: the probe iterates `importlib.metadata.distributions()` and prints name, version and wheel tag. It never reads `sys.version_info`. `verify_env_packages` also returns before making the round trip at all when `pinned == {}`, so a project with no package pins gets no interpreter evidence whatsoever.
