#!/bin/bash -l
#SBATCH -J cleargbm.mount-probe-shared-0a0e5913-v2
#SBATCH --comment project=cleargbm;gpu=cpu-only;cpus=1;env=sif:369ae415a20a;det=on;exp=request_sha256=e8ad109df2ed702fdd1d3c8439651b7b366d8a6b79cb3d1f97c9dd3dbd1c8aca,research_attempt=mount-probe-0a0e5913-s…
#SBATCH -p free
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o /data/homezvol1/wagnera3/research/mount-probe-0a0e5913-v2-20261102/cleargbm/logs/cleargbm.mount-probe-shared-0a0e5913-v2-%j.out
#SBATCH -e /data/homezvol1/wagnera3/research/mount-probe-0a0e5913-v2-20261102/cleargbm/logs/cleargbm.mount-probe-shared-0a0e5913-v2-%j.err

# Undefined variables are a bug, not a default. No `set -e`: the
# payload's exit status is what Slurm records, and this wrapper must
# not convert a failed run into a successful job.
set -u

export HPC3_JOB_NAME="cleargbm.mount-probe-shared-0a0e5913-v2"
export TRAIN_DETERMINISTIC="1"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
module load apptainer/1.4.5
export IMAGE_DIGEST="369ae415a20a1ad8a7d672d839e12c7e62ee29d1432667d95f479d333c578fde"
export GIT_COMMIT="$(module load apptainer/1.4.5 && env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u APPTAINER_MOUNT -u SINGULARITY_BIND -u SINGULARITY_BINDPATH -u SINGULARITY_MOUNT apptainer exec --no-mount bind-paths,hostfs --bind /pub/wagnera3/research/mount-probe-0a0e5913-v2-20261102:/pub/wagnera3/research/mount-probe-0a0e5913-v2-20261102 /data/homezvol1/wagnera3/research-images/protocol-v3-087f67b4c/research-crsp-v3.sif sh -c 'cat /opt/env/GIT_COMMIT' 2>/dev/null || echo '')"

echo "host      $(hostname)"
echo "job       ${SLURM_JOB_ID:-none}"
echo "restart   ${SLURM_RESTART_COUNT:-0}"
echo "commit    ${GIT_COMMIT:-<unstamped>}"
echo "gpu       cpu-only"
date -Is

env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u APPTAINER_MOUNT -u SINGULARITY_BIND -u SINGULARITY_BINDPATH -u SINGULARITY_MOUNT apptainer exec \
    --no-mount bind-paths,hostfs \
    --bind /pub/wagnera3/research/mount-probe-0a0e5913-v2-20261102:/pub/wagnera3/research/mount-probe-0a0e5913-v2-20261102 \
    /data/homezvol1/wagnera3/research-images/protocol-v3-087f67b4c/research-crsp-v3.sif \
    env PATH="/opt/env/bin:$PATH" \
    bash -c 'sha256sum -c /data/homezvol1/wagnera3/research/mount-probe-0a0e5913-v2-20261102/shared-checksums.txt && timeout --signal=TERM --kill-after=5s 120s python /data/homezvol1/wagnera3/research/mount-probe-0a0e5913-v2-20261102/probe.py /data/homezvol1/wagnera3/research/mount-probe-0a0e5913-v2-20261102/shared-request.json /data/homezvol1/wagnera3/research/mount-probe-0a0e5913-v2-20261102/shared-result.json'
rc=$?

date -Is
echo "exit      $rc"
exit $rc
