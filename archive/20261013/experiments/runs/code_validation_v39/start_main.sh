#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/export_packages/extracted
/gemini/code/bin/python3.11 - <<'CHECK'
import json, hashlib
from pathlib import Path
root=Path('runs/code_validation_v39')
for name in ('regression_tests.log','evidence_final_tests.log','main_only_tests.log'):
    text=(root/name).read_text()
    assert '\nOK\n' in text and 'FAILED (' not in text, name
for name, expected in json.loads((root/'source_manifest.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
CHECK
export MIX_WORKSPACE_ROOT="$PWD/evaluationVideo/Mix/workspaces_evidence_formal_20260918_main_v39"
export MIX_SHARED_PREPROCESS_ROOT="$PWD/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache"
export CUDA_VISIBLE_DEVICES=0
export TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=full
export ONLINE_EVOLUTION_GROUP_SIZE=2
export ONLINE_EVOLUTION_TRIGGER_EVERY=2
export MIX_PAIRED_ABLATIONS=0
unset TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES TOOLSTAR_ABLATION_DISABLED_TOOL_NAMES CFV_ONLINE_EVOLUTION_REPLAY CFV_REGISTRY_SNAPSHOT_PATH CFV_ONLINE_ACTIVE_SNAPSHOT_PATH
mkdir -p "$MIX_WORKSPACE_ROOT"
set +e
bash evaluationVideo/Mix/run_mix_with_evolver.sh evaluationVideo/Mix/mix_config_evidence_multi.json \
  > "$MIX_WORKSPACE_ROOT/workflow.log" 2>&1
status=$?
printf '%s\n' "$status" > "$MIX_WORKSPACE_ROOT/workflow_exit_code.txt"
exit "$status"
