#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${RL_ROOT}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
LIMIT=${1:-2}
CFV_WORKSPACES_ROOT=${CFV_WORKSPACES_ROOT:-"${REPO_ROOT}/evaluationVideo/Mix/workspaces"}
BASELINE_EVOLUTION_REPORT=${BASELINE_EVOLUTION_REPORT:-}

EXTRA_ARGS=()
if [[ -n "${BASELINE_EVOLUTION_REPORT}" ]]; then
  EXTRA_ARGS+=(--baseline-report "${BASELINE_EVOLUTION_REPORT}")
fi

cd "${RL_ROOT}"
"${PYTHON_BIN}" -m evolution.run \
  --workspaces-root "${CFV_WORKSPACES_ROOT}" \
  --dataset mlvu \
  --limit "${LIMIT}" \
  --output-root "${REPO_ROOT}/runs" \
  --config "${RL_ROOT}/config/evolution.yaml" \
  "${EXTRA_ARGS[@]}"
