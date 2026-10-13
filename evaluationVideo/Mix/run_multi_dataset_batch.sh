#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

if [[ -f "${CODE_ROOT}/bin/activate" ]]; then
  # conda-pack was unpacked at /gemini/code; activate that prefix so helper
  # tools and CONDA_PREFIX do not resolve to the unrelated base environment.
  # shellcheck disable=SC1091
  source "${CODE_ROOT}/bin/activate"
fi

PYTHON_BIN=${PYTHON_BIN:-"${CODE_ROOT}/bin/python3.11"}
CONFIG_PATH=${CONFIG_PATH:-"${SCRIPT_DIR}/mix_config_multi.json"}

if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi

cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN}" exec bash "${SCRIPT_DIR}/run_mix_batch.sh" "${CONFIG_PATH}" "$@"
