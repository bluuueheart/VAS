#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${CODE_ROOT}/bin/python3.11"}
CONFIG_PATH=${1:-"${SCRIPT_DIR}/mix_config_evidence_multi.json"}
if [[ $# -gt 0 ]]; then
  shift
fi

# The online controller lives inside run_mix_batch.py. It receives each
# terminal trajectory at a worker boundary, starts a bounded Evolver cycle,
# and atomically publishes only replay-gated snapshots. This wrapper does not
# defer evolution until the batch has completed.
WORKSPACE_ROOT=${MIX_WORKSPACE_ROOT:-"${SCRIPT_DIR}/workspaces_evidence_formal_20260918_full_v39"}
if [[ -n "${MIX_SHARED_PREPROCESS_ROOT:-}" ]]; then
  SHARED_PREPROCESS_ROOT=${MIX_SHARED_PREPROCESS_ROOT}
elif [[ -n "${TOOLSTAR_SHARED_PREPROCESS_ROOT_OVERRIDE:-}" ]]; then
  SHARED_PREPROCESS_ROOT=${TOOLSTAR_SHARED_PREPROCESS_ROOT_OVERRIDE}
else
  SHARED_PREPROCESS_ROOT="${WORKSPACE_ROOT}/_shared_cache"
fi
EVOLVER_AUDIT_OUTPUT=${EVOLVER_AUDIT_OUTPUT:-"${WORKSPACE_ROOT}/_audit/mix_audit_with_evolver.json"}

mkdir -p "${WORKSPACE_ROOT}"
export TOOLSTAR_WORKSPACES_ROOT_OVERRIDE="${WORKSPACE_ROOT}"
export TOOLSTAR_PROGRESS_PATH_OVERRIDE="${WORKSPACE_ROOT}/mix_progress.json"
export TOOLSTAR_RESULTS_PATH_OVERRIDE="${WORKSPACE_ROOT}/CFV_MultiDataset_Evidence_result.json"
export TOOLSTAR_VIDEOMME_OUTPUT_PATH_OVERRIDE="${WORKSPACE_ROOT}/VideoMME_multi_Evidence_official_output.json"
export TOOLSTAR_SHARED_PREPROCESS_ROOT_OVERRIDE="${SHARED_PREPROCESS_ROOT}"
export ONLINE_EVOLUTION=1
export ONLINE_EVOLUTION_TRIGGER_EVERY=${ONLINE_EVOLUTION_TRIGGER_EVERY:-2}
# Reuse loaded models until the next eligible evolution boundary.
export ONLINE_EVOLUTION_GROUP_SIZE=${ONLINE_EVOLUTION_GROUP_SIZE:-${ONLINE_EVOLUTION_TRIGGER_EVERY}}
export MIX_PAIRED_ABLATIONS=${MIX_PAIRED_ABLATIONS:-1}
if [[ "${MIX_PAIRED_ABLATIONS}" != 0 && "${MIX_PAIRED_ABLATIONS}" != 1 ]]; then
  echo "MIX_PAIRED_ABLATIONS must be 0 or 1" >&2
  exit 2
fi
export TOOLSTAR_DISABLE_LOCAL_VECTOR_INDEX=0
export TOOLSTAR_SEMANTIC_INDEX_VIDEO=1
export MIX_PAIRED_ABLATION_TOOLS=${MIX_PAIRED_ABLATION_TOOLS:-all}
export PYTHONPATH="${SCRIPT_DIR}/../../RL:${SCRIPT_DIR}/../..${PYTHONPATH:+:${PYTHONPATH}}"

echo "[Main] workspace_root=${WORKSPACE_ROOT}"
echo "[Main] shared_preprocess_root=${SHARED_PREPROCESS_ROOT}"
echo "[Main] config=${CONFIG_PATH}"
echo "[Main] phase=batch_with_online_evolver"
bash "${SCRIPT_DIR}/run_mix_batch.sh" "${CONFIG_PATH}" "$@" 2>&1 | tee "${WORKSPACE_ROOT}/batch_online_tmux.log"

# Main-only runs still report actual evidence and Evolver outcomes. Missing
# ablations remain unproven in the report; this is not full delivery acceptance.
if [[ "${MIX_PAIRED_ABLATIONS}" == 0 ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_multi_run.py" \
    --config "${CONFIG_PATH}" --output "${EVOLVER_AUDIT_OUTPUT}"
  echo "[Main] workflow_status=completed_main_only; paired effectiveness not evaluated"
  echo "[Main] online_state=${WORKSPACE_ROOT}/_evolution_audit/online_state.json"
  echo "[Main] audit_output=${EVOLVER_AUDIT_OUTPUT}"
  exit 0
fi

# The audit consumes the online controller's immutable cycle records and
# active snapshot. It is a reporting/acceptance check, never an Evolver
# trigger, so a finished baseline cannot silently become a prerequisite for
# adaptation.
echo "[Main] phase=final_online_evolver_audit"
mkdir -p "$(dirname "${EVOLVER_AUDIT_OUTPUT}")"
for role in observer reflector; do
  "${PYTHON_BIN}" -m evolution.agent_value_audit \
    --workspaces-root "${WORKSPACE_ROOT}" \
    --counterfactual-root "${WORKSPACE_ROOT}/_ablations/${role}" \
    --disabled-role "${role}" \
    --output "${WORKSPACE_ROOT}/_evolution_audit/counterfactual_${role}.json"
done

audit_status=0
found_tool=0
for tool_root in "${WORKSPACE_ROOT}"/_ablations/tool_*; do
  [[ -d "${tool_root}" ]] || continue
  found_tool=1
  tool_name=${tool_root##*/tool_}
  audit_output="${WORKSPACE_ROOT}/_audit/tool_${tool_name}.json"
  if [[ "${tool_name}" == "Video_Clip_QA" ]]; then
    audit_output="${EVOLVER_AUDIT_OUTPUT}"
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_multi_run.py" \
    --config "${CONFIG_PATH}" --output "${audit_output}" \
    --tool-ablation-root "${tool_root}" --tool-ablation-disabled-tool "${tool_name}" \
    --require-delivery 2>&1 | tee "${WORKSPACE_ROOT}/audit_${tool_name}.log" || audit_status=2
done
if [[ "${found_tool}" == 0 || "${audit_status}" != 0 ]]; then
  echo "[Main] workflow_status=validation_failed; see actual paired results and audits"
  exit 2
fi

echo "[Main] workflow_status=completed_online_evolution"
echo "[Main] online_state=${WORKSPACE_ROOT}/_evolution_audit/online_state.json"
echo "[Main] audit_output=${EVOLVER_AUDIT_OUTPUT}"
