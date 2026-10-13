#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/export_packages/extracted
source /gemini/code/bin/activate
export PYTHONPATH=.:RL:evaluationVideo/Mix
export MODEL_BASE_PATH=/gemini/pretrain/model
export CUDA_VISIBLE_DEVICES=0
export TOOLSTAR_DISABLE_CUDNN=1
export TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=full
export QWEN_VL_IMAGE_MIN_PIXELS=65536
export QWEN_VL_IMAGE_MAX_PIXELS=200704
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${ORION_DEVICE_TYPE:-}" == GPU && -n "${CUDA_VERSION:-}" ]]; then
  runtime_cuda="/opt/orion/orion_runtime/gpu/cuda/orion-cuda-${CUDA_VERSION%.*}"
  if [[ -d "$runtime_cuda" ]]; then
    export LD_LIBRARY_PATH="$runtime_cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
unset CFV_REGISTRY_BASELINE_ID CFV_REGISTRY_BASELINE_SNAPSHOT_PATH CFV_ONLINE_ACTIVE_SNAPSHOT_PATH CFV_RUNTIME_SOURCE_ROOT
exec /gemini/code/bin/python3.11 runs/code_validation_v41/live_candidate_replay.py
