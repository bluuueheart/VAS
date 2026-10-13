#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${CODE_ROOT}/bin/python3.11"}
CONFIG_PATH=${CONFIG_PATH:-"${REPO_ROOT}/evaluationVideo/Mix/mix_config.json"}
TRAJECTORY_COUNT=${1:-2}
CFV_WORKSPACES_ROOT=${CFV_WORKSPACES_ROOT:-"${REPO_ROOT}/evaluationVideo/Mix/workspaces"}
CFV_PROGRESS_PATH=${CFV_PROGRESS_PATH:-"${CFV_WORKSPACES_ROOT}/cfv_evolve_progress.json"}
CFV_RESULTS_PATH=${CFV_RESULTS_PATH:-"${CFV_WORKSPACES_ROOT}/CFV_Evolve_result.json"}
CFV_VIDEOMME_OUTPUT_PATH=${CFV_VIDEOMME_OUTPUT_PATH:-"${CFV_WORKSPACES_ROOT}/VideoMME_official_output.json"}
CFV_SAMPLE_UIDS=${CFV_SAMPLE_UIDS:-}

# This helper is a bounded online smoke runner.  The batch scheduler owns the
# Evolver lifecycle; it must not run a second miner after the batch exits.
export ONLINE_EVOLUTION=1
export ONLINE_EVOLUTION_TRIGGER_EVERY=${ONLINE_EVOLUTION_TRIGGER_EVERY:-8}
export ONLINE_EVOLUTION_GROUP_SIZE=${ONLINE_EVOLUTION_GROUP_SIZE:-${TRAJECTORY_COUNT}}

configure_orion_cuda_runtime() {
  # Orion vGPU exposes CUDA through versioned proxy libraries.  Prefer the
  # matching proxy over wheel-bundled CUDA/cuDNN libraries so the child Python
  # process uses one coherent CUDA runtime.
  if [[ "${ORION_DEVICE_TYPE:-}" != "GPU" || -z "${CUDA_VERSION:-}" ]]; then
    return
  fi
  local major_minor
  major_minor=$(printf '%s' "${CUDA_VERSION}" | awk -F. '{print $1 "." $2}')
  local runtime_dir="/opt/orion/orion_runtime/gpu/cuda/orion-cuda-${major_minor}"
  if [[ -d "${runtime_dir}" ]]; then
    export LD_LIBRARY_PATH="${runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "Using Orion CUDA proxy runtime: ${runtime_dir}" >&2
  fi
}

configure_orion_cuda_runtime

resolve_repo_path() {
  local value=$1
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${value}"
  fi
}

CFV_WORKSPACES_ROOT=$(resolve_repo_path "${CFV_WORKSPACES_ROOT}")
CFV_PROGRESS_PATH=$(resolve_repo_path "${CFV_PROGRESS_PATH}")
CFV_RESULTS_PATH=$(resolve_repo_path "${CFV_RESULTS_PATH}")
CFV_VIDEOMME_OUTPUT_PATH=$(resolve_repo_path "${CFV_VIDEOMME_OUTPUT_PATH}")
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
MIX_ARGS=(
  --max-samples "${TRAJECTORY_COUNT}"
  --parallel-samples "${CFV_PARALLEL_SAMPLES}"
)
if [[ -n "${CFV_SAMPLE_UIDS}" ]]; then
  MIX_ARGS+=(--sample-uids "${CFV_SAMPLE_UIDS}")
fi
CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  TOOLSTAR_WORKSPACES_ROOT_OVERRIDE="${CFV_WORKSPACES_ROOT}" \
  TOOLSTAR_PROGRESS_PATH_OVERRIDE="${CFV_PROGRESS_PATH}" \
  TOOLSTAR_RESULTS_PATH_OVERRIDE="${CFV_RESULTS_PATH}" \
  TOOLSTAR_VIDEOMME_OUTPUT_PATH_OVERRIDE="${CFV_VIDEOMME_OUTPUT_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash evaluationVideo/Mix/run_mix_batch.sh "${CONFIG_PATH}" \
    "${MIX_ARGS[@]}"

echo "[Main] online_state=${CFV_WORKSPACES_ROOT}/_evolution_audit/online_state.json"
echo "[Main] workflow_status=completed_online_evolution"
