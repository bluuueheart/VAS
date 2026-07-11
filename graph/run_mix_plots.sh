#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-"$SCRIPT_DIR/mix_plot_config.json"}"

python "$SCRIPT_DIR/plot_mix_metrics.py" --config "$CONFIG_PATH"
