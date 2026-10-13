#!/usr/bin/env bash
set -euo pipefail
pilot_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$pilot_dir/../.." && pwd)
cd "$project_root"
source /gemini/code/bin/activate
export PYTHONPATH=.:RL:evaluationVideo/Mix
export MODEL_BASE_PATH=/gemini/pretrain/model
export CUDA_VISIBLE_DEVICES=0
export TOOLSTAR_DISABLE_CUDNN=1
export TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=full
export QWEN_VL_IMAGE_MIN_PIXELS=65536
export QWEN_VL_IMAGE_MAX_PIXELS=200704
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ONLINE_EVOLUTION=0
export MIX_PAIRED_ABLATIONS=0
unset CFV_REGISTRY_BASELINE_ID CFV_REGISTRY_BASELINE_SNAPSHOT_PATH CFV_ONLINE_ACTIVE_SNAPSHOT_PATH CFV_RUNTIME_SOURCE_ROOT CFV_REGISTRY_SNAPSHOT_PATH CFV_RUNTIME_RESUME_CHECKPOINT
unset TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES TOOLSTAR_ABLATION_DISABLED_TOOL_NAMES TOOLSTAR_ABLATION_SINGLE_TOOL_ONLY TOOLSTAR_ABLATION_SINGLE_ACTION_PER_STEP TOOLSTAR_ABLATION_DISABLE_META_TOOLS TOOLSTAR_PIPELINE_MODE_OVERRIDE
if [[ "${ORION_DEVICE_TYPE:-}" == GPU && -n "${CUDA_VERSION:-}" ]]; then
  pilot_cuda="/opt/orion/orion_runtime/gpu/cuda/orion-cuda-${CUDA_VERSION%.*}"
  if [[ -d "$pilot_cuda" ]]; then
    export LD_LIBRARY_PATH="$pilot_cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
exec /gemini/code/bin/python3.11 "$pilot_dir/pilot.py" "${1:-baseline}"
