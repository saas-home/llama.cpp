#!/bin/bash
set -euo pipefail

# ====================== USAGE ==============================
# ./save-baseline.sh [config]             → run bench and save as golden baseline
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_FILE="$SCRIPT_DIR/baselines.json"
CONFIG_FILE="${1:-}"

if [[ -z "$CONFIG_FILE" ]]; then
    echo "❌ Usage: ./save-baseline.sh <config_file.conf>"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# 1. Source the config to get model info
source "$CONFIG_FILE"
CONFIG_NAME=$(basename "$CONFIG_FILE")

echo "🚀 Benchmarking to save Golden Baseline for: $CONFIG_NAME"

# 2. Run benchmark (3 rounds, 3 parallel slots by default for MoE)
# We capture stdout to parse the results
OUTPUT=$(python3 "$SCRIPT_DIR/bench-llama.py" 3 -p 3)

echo "$OUTPUT"

# 3. Parse results using grep/sed
# Generation  : avg   63.1 | min   62.5 | max   64.2 tok/s
# Throughput  :   84.3 tok/s (aggregate)

AVG_GEN=$(echo "$OUTPUT" | grep "Generation" | tail -n 1 | awk '{print $4}')
AGG_THR=$(echo "$OUTPUT" | grep "Throughput" | tail -n 1 | awk '{print $3}')

if [[ -z "$AVG_GEN" || -z "$AGG_THR" ]]; then
    echo "❌ Failed to parse benchmark results."
    exit 1
fi

echo "📊 Parsed Results:"
echo "   - Avg Gen: $AVG_GEN tok/s"
echo "   - Agg Thr: $AGG_THR tok/s"

# 4. Update baselines.json (using python for easy JSON manipulation)
python3 - << EOF
import json
import os

file_path = "$BASELINES_FILE"
data = {}
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

data["$CONFIG_NAME"] = {
    "avg_gen": float("$AVG_GEN"),
    "agg_thr": float("$AGG_THR"),
    "timestamp": "$(date -Iseconds)"
}

with open(file_path, "w") as f:
    json.dump(data, f, indent=4)
EOF

echo "✅ Golden Baseline saved to $BASELINES_FILE"
