#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_PATH=${CONFIG_PATH:-"${REPO_ROOT}/evaluationVideo/Mix/mix_config.json"}
TRAJECTORY_COUNT=${1:-2}
CFV_WORKSPACES_ROOT=${CFV_WORKSPACES_ROOT:-"${REPO_ROOT}/evaluationVideo/Mix/workspaces"}
VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -r -a DEVICE_LIST <<< "${VISIBLE_DEVICES}"
GPU_COUNT=${#DEVICE_LIST[@]}
DEFAULT_PARALLEL_SAMPLES=${TRAJECTORY_COUNT}
if (( DEFAULT_PARALLEL_SAMPLES > GPU_COUNT )); then
  DEFAULT_PARALLEL_SAMPLES=${GPU_COUNT}
fi
CFV_PARALLEL_SAMPLES=${CFV_PARALLEL_SAMPLES:-${DEFAULT_PARALLEL_SAMPLES}}

if ! [[ "${TRAJECTORY_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "trajectory count must be a positive integer" >&2
  exit 2
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  CFV_REGISTRY_SNAPSHOT_ID=${CFV_REGISTRY_SNAPSHOT_ID:-snapshot_001_candidate} \
  TOOLSTAR_WORKSPACES_ROOT_OVERRIDE="${CFV_WORKSPACES_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash evaluationVideo/Mix/run_mix_batch.sh "${CONFIG_PATH}" \
    --max-samples "${TRAJECTORY_COUNT}" \
    --parallel-samples "${CFV_PARALLEL_SAMPLES}"

CFV_WORKSPACES_ROOT="${CFV_WORKSPACES_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash RL/scripts/run_evolution.sh "${TRAJECTORY_COUNT}"
