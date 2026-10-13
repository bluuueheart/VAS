#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/export_packages/extracted
/usr/bin/python3 - <<'CHECK'
import json, hashlib
from pathlib import Path
root = Path('runs/code_validation_v40')
for name in ('regression_tests.log', 'final_role_tests.log'):
    log = (root / name).read_text()
    assert '\nOK\n' in log and 'FAILED (' not in log, name
assert json.loads((root / 'live_validation.json').read_text())['status'] == 'passed', 'live validation has not passed'
for name, expected in json.loads((root / 'source_manifest.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
CHECK
export MIX_WORKSPACE_ROOT="$PWD/evaluationVideo/Mix/workspaces_evidence_formal_20260920_main_v40"
export MIX_SHARED_PREPROCESS_ROOT="$PWD/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache"
export CUDA_VISIBLE_DEVICES=0
export TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=full
export ONLINE_EVOLUTION_GROUP_SIZE=2
export ONLINE_EVOLUTION_TRIGGER_EVERY=2
export MIX_PAIRED_ABLATIONS=0
unset TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES TOOLSTAR_ABLATION_DISABLED_TOOL_NAMES CFV_ONLINE_EVOLUTION_REPLAY CFV_REGISTRY_SNAPSHOT_PATH CFV_ONLINE_ACTIVE_SNAPSHOT_PATH
mkdir -p "$MIX_WORKSPACE_ROOT"
if [[ -e "$MIX_WORKSPACE_ROOT/workflow.log" ]]; then
  mv "$MIX_WORKSPACE_ROOT/workflow.log" "$MIX_WORKSPACE_ROOT/workflow.$(date +%Y%m%d_%H%M%S).log"
fi
set +e
bash evaluationVideo/Mix/run_mix_with_evolver.sh evaluationVideo/Mix/mix_config_evidence_multi.json \
  > "$MIX_WORKSPACE_ROOT/workflow.log" 2>&1
status=$?
printf '%s\n' "$status" > "$MIX_WORKSPACE_ROOT/workflow_exit_code.txt"
exit "$status"
