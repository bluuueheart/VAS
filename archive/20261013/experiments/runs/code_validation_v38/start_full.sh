#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/export_packages/extracted
/gemini/code/bin/python3.11 - <<'PY'
import json
from pathlib import Path
root=Path('runs/code_validation_v38')
assert json.loads((root/'dense_smoke_result.json').read_text())['status']=='passed'
for name in ('regression_tests.log','online_final.log','tool_tests_final.log','dense_contract_final.log','delivery_launch_final.log'):
    text=(root/name).read_text()
    assert '\nOK\n' in text and 'FAILED (' not in text, name
PY
export MIX_WORKSPACE_ROOT="$PWD/evaluationVideo/Mix/workspaces_evidence_formal_20260917_full_v38"
export MIX_SHARED_PREPROCESS_ROOT="$PWD/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache"
export CUDA_VISIBLE_DEVICES=0
export TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=full
export ONLINE_EVOLUTION_GROUP_SIZE=1
export ONLINE_EVOLUTION_TRIGGER_EVERY=2
export MIX_PAIRED_ABLATION_TOOLS=all
mkdir -p "$MIX_WORKSPACE_ROOT"
set +e
bash evaluationVideo/Mix/run_mix_with_evolver.sh evaluationVideo/Mix/mix_config_evidence_multi.json \
  > "$MIX_WORKSPACE_ROOT/workflow.log" 2>&1
status=$?
printf '%s\n' "$status" > "$MIX_WORKSPACE_ROOT/workflow_exit_code.txt"
exit "$status"
