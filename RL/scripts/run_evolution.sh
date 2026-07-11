#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${RL_ROOT}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
LIMIT=${1:-2}

cd "${RL_ROOT}"
"${PYTHON_BIN}" -m evolution.run \
  --workspaces-root "${REPO_ROOT}/evaluationVideo/Mix/workspaces" \
  --dataset mlvu \
  --limit "${LIMIT}" \
  --output-root "${REPO_ROOT}/runs" \
  --config "${RL_ROOT}/config/evolution.yaml"
