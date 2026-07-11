#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 && "${1}" != --* ]]; then
  if [[ "${1}" == *.json ]]; then
    CONFIG_PATH="$1"
    shift
    python "$SCRIPT_DIR/plot_tool_inventory.py" --config "$CONFIG_PATH" "$@"
    exit 0
  fi

  OUTPUT_DIR="$1"
  shift
  python "$SCRIPT_DIR/plot_tool_inventory.py" --output-dir "$OUTPUT_DIR" "$@"
  exit 0
fi

CONFIG_PATH="${SCRIPT_DIR}/tool_inventory_plot_config.json"
python "$SCRIPT_DIR/plot_tool_inventory.py" --config "$CONFIG_PATH" "$@"
