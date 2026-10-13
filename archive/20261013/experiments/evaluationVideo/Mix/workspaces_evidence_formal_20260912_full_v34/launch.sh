#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/export_packages/extracted
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=/gemini/code/bin/python3.11
export PYTHONDONTWRITEBYTECODE=1
export MODEL_BASE_PATH=/gemini/pretrain/model
export MIX_WORKSPACE_ROOT=/gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34
export MIX_SHARED_PREPROCESS_ROOT=/gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache
trap 'status=$?; printf "%s\n" "$status" > "$MIX_WORKSPACE_ROOT/launcher_exit_code.txt"' EXIT
printf 'phase=final_code_regression\n'
PYTHONPATH=RL "$PYTHON_BIN" -B -m unittest evolution.test_verifier_boundaries evolution.test_replay evolution.test_selective_replay
printf 'phase=full_experiment\n'
bash evaluationVideo/Mix/run_mix_with_evolver.sh /gemini/code/export_packages/extracted/evaluationVideo/Mix/mix_config_evidence_multi.json --force-rerun
