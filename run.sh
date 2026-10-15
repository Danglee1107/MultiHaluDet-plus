#!/usr/bin/env bash
# Run the pipeline. All arguments go to run_pipeline.py.
# Usage: ./run.sh                                  (halueval, mistral-7b, en, all stages)
#        ./run.sh --dataset triviaqa --model llama2-7b --stage extract
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
if [ -d /workspace ]; then
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
fi
export MPLBACKEND=Agg        # no display on the instance
export PYTHONUNBUFFERED=1    # progress reaches the log at once

# The environment and the two models of scripts/build_data.py (from setup_data.sh) take about
# 40 GB. Delete them to make room for the pipeline environment and the pipeline models: "uv run"
# below installs the environment, and run_pipeline.py downloads the model that it needs.
# KEEP_BUILD_DATA=1 keeps them. ./setup_data.sh brings them back.
hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
# Both caches: build_data.py without HF_HOME downloads to the home directory.
for path in .venv-data .venv-vllm \
    {"$hub","$HOME/.cache/huggingface/hub"}/models--google--{translategemma-12b-it,gemma-2-2b-it}; do
    if [ ! -e "$path" ] || [ "${KEEP_BUILD_DATA:-0}" = 1 ]; then
        continue
    fi
    if pgrep -f "scripts/build_data.py" >/dev/null; then
        echo "ERROR: scripts/build_data.py still runs and needs $path. Wait until it is done." >&2
        exit 1
    fi
    echo "Deleting $path (only scripts/build_data.py needs it) to free disk space"
    rm -rf "$path"
done

mkdir -p logs
log="logs/$(date +%Y%m%d_%H%M%S).log"
echo "Log: $log"

uv run python run_pipeline.py "$@" 2>&1 | tee "$log"
