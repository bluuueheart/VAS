#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

if [[ -f "${CODE_ROOT}/bin/activate" ]]; then
  # Use the conda-pack prefix extracted at /gemini/code.
  # shellcheck disable=SC1091
  source "${CODE_ROOT}/bin/activate"
fi

PYTHON_BIN=${PYTHON_BIN:-"${CODE_ROOT}/bin/python3.11"}
# Keep standalone worker launches consistent with the checked-in batch plan.
# The local Qwen checkpoints live outside the repository; callers can still
# override this explicitly for another mounted model root.
if [[ -z "${MODEL_BASE_PATH:-}" ]]; then
  export MODEL_BASE_PATH=/gemini/pretrain/model
fi
CONFIG_PATH="${SCRIPT_DIR}/mix_config.json"

if [[ "${ORION_DEVICE_TYPE:-}" == "GPU" && -n "${CUDA_VERSION:-}" ]]; then
  ORION_CUDA_VERSION=$(printf '%s' "${CUDA_VERSION}" | awk -F. '{print $1 "." $2}')
  ORION_CUDA_DIR="/opt/orion/orion_runtime/gpu/cuda/orion-cuda-${ORION_CUDA_VERSION}"
  if [[ -d "${ORION_CUDA_DIR}" && ":${LD_LIBRARY_PATH:-}:" != *":${ORION_CUDA_DIR}:"* ]]; then
    export LD_LIBRARY_PATH="${ORION_CUDA_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "Using Orion CUDA proxy runtime: ${ORION_CUDA_DIR}" >&2
  fi
fi

# The Orion CUDA proxy's cuDNN runtime-compiled engine path can abort during
# Qwen3.5 visual generation.  The caption runtime applies this only to the
# batch evaluation process; callers can explicitly override it with 0/false.
if [[ -z "${TOOLSTAR_DISABLE_CUDNN:-}" ]]; then
  export TOOLSTAR_DISABLE_CUDNN=1
fi

# Qwen3.5's FLA/causal-conv/fused-RMSNorm kernels can trigger an uncaught
# LLVM MemoryBuffer abort through the Orion CUDA proxy.  The caption runtime
# falls back to the Transformers Torch implementation when this is enabled.
# Callers on a fully compatible CUDA stack can explicitly set this to 0/false.
if [[ -z "${TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS:-}" ]]; then
  export TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=1
fi

# Keep the first local Qwen3.5 visual generation below the single-GPU memory
# cliff observed on the Orion proxy.  These are batch defaults only; callers
# can raise them explicitly when a different GPU/runtime is available.
if [[ -z "${QWEN_VL_IMAGE_MIN_PIXELS:-}" ]]; then
  export QWEN_VL_IMAGE_MIN_PIXELS=65536
fi
if [[ -z "${QWEN_VL_IMAGE_MAX_PIXELS:-}" ]]; then
  export QWEN_VL_IMAGE_MAX_PIXELS=200704
fi
if [[ -z "${PYTORCH_ALLOC_CONF:-}" && -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_ALLOC_CONF=expandable_segments:True
fi

if [[ -z "${TOOLSTAR_REUSE_SEGMENT_METADATA:-}" ]]; then
  export TOOLSTAR_REUSE_SEGMENT_METADATA=1
fi

# Caption generation settings are resolved from pipeline_options by
# run_mix_batch.py and injected into every isolated worker.  Do not set a
# shell default here: an inherited value must not silently replace the
# experiment configuration.

# Keep online Video_Clip_QA evidence calls within the large-GPU latency budget.
# The full-resolution defaults can turn one 40-frame clip into a ~30k-token
# multimodal prompt on the Orion runtime.  This preserves timestamped dynamic
# evidence while using a compact, deterministic request; callers may override
# all three values for a quality/latency probe.
if [[ -z "${VIDEO_CLIP_QA_MAX_FRAMES:-}" ]]; then
  export VIDEO_CLIP_QA_MAX_FRAMES=32
fi
if [[ -z "${VIDEO_CLIP_QA_MAX_PIXELS:-}" ]]; then
  export VIDEO_CLIP_QA_MAX_PIXELS=200704
fi
# Use the shared settings default for the four-field JSON response.  The old
# 64-token shell override truncated real evidence payloads.  An explicit
# VIDEO_CLIP_QA_MAX_NEW_TOKENS remains available for measured probes.


# The large-memory reusable worker can retain Whisper between new videos,
# avoiding a repeated ASR cold load.  Small-card callers can explicitly set
# TOOLSTAR_PERSIST_ASR_RUNTIME=0 to restore the release-after-ASR behavior.
if [[ -z "${TOOLSTAR_PERSIST_ASR_RUNTIME:-}" ]]; then
  export TOOLSTAR_PERSIST_ASR_RUNTIME=1
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi

# Evidence/ReAct runs must retain the original Semantic_Retrieve contract and
# therefore build a real dense vector index.  The direct-answer baseline may
# skip this optional preprocessing cost.  Callers can still override the
# switch explicitly for controlled ablations.
if [[ -z "${TOOLSTAR_DISABLE_LOCAL_VECTOR_INDEX:-}" ]]; then
  case "${CONFIG_PATH}" in
    *evidence*)
      export TOOLSTAR_DISABLE_LOCAL_VECTOR_INDEX=0
      # Captions/transcripts are the dense retrieval keys; frame paths remain
      # attached as auditable evidence.  Avoid one expensive Qwen vision
      # forward per segment in the full batch.  Segments without text still
      # use the generator's one-frame last resort.
      if [[ -z "${TOOLSTAR_SEMANTIC_INDEX_VIDEO:-}" ]]; then
        export TOOLSTAR_SEMANTIC_INDEX_VIDEO=0
      fi
      ;;
    *) export TOOLSTAR_DISABLE_LOCAL_VECTOR_INDEX=1 ;;
  esac
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" evaluationVideo/Mix/run_mix_batch.py --config "${CONFIG_PATH}" "$@"
