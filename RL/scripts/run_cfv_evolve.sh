#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_PATH=${CONFIG_PATH:-"${REPO_ROOT}/evaluationVideo/Mix/mix_config.json"}
TRAJECTORY_COUNT=${1:-2}

if ! [[ "${TRAJECTORY_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "trajectory count must be a positive integer" >&2
  exit 2
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  CFV_REGISTRY_SNAPSHOT_ID=${CFV_REGISTRY_SNAPSHOT_ID:-snapshot_001_candidate} \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash evaluationVideo/Mix/run_mix_batch.sh "${CONFIG_PATH}" \
    --max-samples "${TRAJECTORY_COUNT}" \
    --parallel-samples "${TRAJECTORY_COUNT}"

PYTHON_BIN="${PYTHON_BIN}" bash RL/scripts/run_evolution.sh "${TRAJECTORY_COUNT}"
