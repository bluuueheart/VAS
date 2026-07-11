#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_PATH="${SCRIPT_DIR}/mix_config.json"

if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" evaluationVideo/Mix/run_mix_batch.py --config "${CONFIG_PATH}" "$@"
