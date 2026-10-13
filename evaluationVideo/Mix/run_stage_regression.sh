#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REGRESSION_ROOT=${MIX_WORKSPACE_ROOT:-"${SCRIPT_DIR}/workspaces_evidence_regression_20260914_v35"}
export TOOLSTAR_WORKSPACES_ROOT_OVERRIDE="${REGRESSION_ROOT}"
export TOOLSTAR_PROGRESS_PATH_OVERRIDE="${REGRESSION_ROOT}/mix_progress.json"
export TOOLSTAR_RESULTS_PATH_OVERRIDE="${REGRESSION_ROOT}/CFV_MultiDataset_Evidence_result.json"
export TOOLSTAR_VIDEOMME_OUTPUT_PATH_OVERRIDE="${REGRESSION_ROOT}/VideoMME_multi_Evidence_official_output.json"
export TOOLSTAR_SHARED_PREPROCESS_ROOT_OVERRIDE=${MIX_SHARED_PREPROCESS_ROOT:-"${SCRIPT_DIR}/workspaces_evidence_formal_20260826_full_v26/_shared_cache"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=full
# Paired historical failure/correct controls across all four datasets.  This
# diagnostic run does not count as full-baseline or Evolver acceptance.
REGRESSION_UIDS='egoschema:00006:026a2f15-c454-4c28-80e0-24c85d7f4ecf,lvbench:entity-recognition:16:83,videomme:acrobatics:211-1,mlvu:anomaly-reco:0:surveil-20,egoschema:00000:0074f737-11cb-497d-8d07-77c3a8127391,lvbench:entity-recognition:39:208,videomme:acrobatics:211-2,mlvu:ego:1:ego-36'
mkdir -p "${REGRESSION_ROOT}"
bash "${SCRIPT_DIR}/run_mix_batch.sh" "${SCRIPT_DIR}/mix_config_evidence_multi.json" --sample-uids "${REGRESSION_UIDS}" --reuse-worker-samples 8 "$@" 2>&1 | tee "${REGRESSION_ROOT}/regression.log"
