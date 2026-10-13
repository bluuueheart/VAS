#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/export_packages/extracted
export PYTHONPATH=.:RL:evaluationVideo/Mix
export CUDA_VISIBLE_DEVICES=0
export TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=full
export TOOLSTAR_WORKSPACES_ROOT_OVERRIDE="$PWD/evaluationVideo/Mix/workspaces_evidence_regression_20260914_v36"
export TOOLSTAR_SHARED_PREPROCESS_ROOT_OVERRIDE="$PWD/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache"
export CFV_REGISTRY_BASELINE_SNAPSHOT_PATH="$TOOLSTAR_WORKSPACES_ROOT_OVERRIDE/_audit/tool_library_baseline.json"
export ONLINE_EVOLUTION=0
mkdir -p runs/code_validation_v37/live_suffix
set +e
/gemini/code/bin/python3.11 -m evolution.harness_replay \
 --run-result "$TOOLSTAR_WORKSPACES_ROOT_OVERRIDE/egoschema/00006-026a2f15-c454-4c28-80e0-24c85d7f4ecf/results/run_result.json" \
 --snapshot "$TOOLSTAR_WORKSPACES_ROOT_OVERRIDE/_evolution_audit/evolution_online_cycle_00003/evolution/registry/snapshot_f4529b93d694ad74_candidate/snapshot.json" \
 --config evaluationVideo/Mix/mix_config_evidence_multi.json \
 --output runs/code_validation_v37/live_suffix/harness_replay.json \
 --device-id 0 --timeout-seconds 7200 \
 > runs/code_validation_v37/live_suffix/driver.log 2>&1
status=$?
printf '%s\n' "$status" > runs/code_validation_v37/live_suffix/process_exit_code.txt
exit "$status"
