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

mkdir -p logs
log="logs/$(date +%Y%m%d_%H%M%S).log"
echo "Log: $log"

uv run python run_pipeline.py "$@" 2>&1 | tee "$log"
