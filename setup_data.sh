#!/usr/bin/env bash
# One-time setup for scripts/build_data.py only: its own environment, datasets, models.
# The project environment pins transformers 4.52.4; TranslateGemma has not been tried with it.
# Usage: HF_TOKEN=hf_xxx ./setup_data.sh [--no-models]
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv-data
DOWNLOAD_MODELS=1
[ "${1:-}" = "--no-models" ] && DOWNLOAD_MODELS=0

# Keep the Hugging Face cache on the instance volume, not the small root disk.
if [ -d /workspace ]; then
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
    mkdir -p "$HF_HOME"
fi

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found. This instance has no GPU driver." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== SGLang =="
# The vast.ai SGLang template starts a server that holds the GPU memory the models need, and its
# model takes disk space. Stop the server and delete its model from the Hugging Face cache.
# KEEP_SGLANG_MODEL=1 keeps the model files. The server starts again when the instance restarts.
sglang_server='sglang\.launch_server|sglang serve'
sglang_model="${SGLANG_MODEL:-$(pgrep -af "$sglang_server" | grep -oP -- '--model(-path)?[ =]\K\S+' | head -1 || true)}"
if pgrep -f "$sglang_server" >/dev/null; then
    if command -v supervisorctl >/dev/null; then
        # Stop it in supervisor first, or supervisor starts a killed server again.
        supervisorctl status 2>/dev/null | awk 'tolower($1) ~ /sglang/ {print $1}' \
            | xargs -r supervisorctl stop || true
    fi
    pkill -f "$sglang_server" || true
    for _ in $(seq 30); do pgrep -f "$sglang_server" >/dev/null || break; sleep 1; done
    echo "stopped the SGLang server${sglang_model:+ ($sglang_model)}"
else
    echo "no SGLang server is running"
fi
if [ -n "$sglang_model" ] && [ "${KEEP_SGLANG_MODEL:-0}" != 1 ]; then
    # Only a model directory inside a Hugging Face cache is deleted, never another path.
    for hub in "${HF_HOME:-$HOME/.cache/huggingface}/hub" /workspace/.hf_home/hub "$HOME/.cache/huggingface/hub"; do
        dir="$hub/models--${sglang_model//\//--}"
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            echo "deleted $dir"
        fi
    done
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "== Pipeline files =="
# The pipeline environment and its two models take about 35 GB, and this setup needs about 40 GB.
# Delete them to make room: ./run.sh installs the environment again, and run_pipeline.py downloads
# the model that it needs. KEEP_PIPELINE=1 keeps them. data/ and results/ are not touched.
hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
for path in .venv \
    {"$hub","$HOME/.cache/huggingface/hub"}/models--{mistralai--Mistral-7B-Instruct-v0.2,meta-llama--Llama-2-7b-hf}; do
    if [ ! -e "$path" ] || [ "${KEEP_PIPELINE:-0}" = 1 ]; then
        continue
    fi
    if pgrep -f "run_pipeline.py" >/dev/null; then
        echo "ERROR: run_pipeline.py still runs and needs $path. Wait until it is done." >&2
        exit 1
    fi
    echo "Deleting $path (only the pipeline needs it) to free disk space"
    rm -rf "$path"
done

echo "== Disk =="
free_gb=$(df -BG --output=avail "${HF_HOME:-$HOME}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "${free_gb} GB free"
if [ "$DOWNLOAD_MODELS" = 1 ] && [ "${free_gb:-0}" -lt 45 ]; then
    echo "WARNING: the models and the environment need about 40 GB. Less than 45 GB is free."
fi

echo "== HF_TOKEN =="
# .env is gitignored, so a fresh clone does not have it.
if [ ! -f .env ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: no .env file and HF_TOKEN is not set." >&2
        echo "Run: HF_TOKEN=hf_xxx ./setup_data.sh" >&2
        exit 1
    fi
    (umask 077; echo "HF_TOKEN=${HF_TOKEN}" > .env)
    echo "wrote .env"
fi
set -a; . ./.env; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "ERROR: HF_TOKEN is empty in .env" >&2; exit 1; }

echo "== uv =="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -d "$VENV" ] || uv venv "$VENV"
# accelerate and bitsandbytes are for the model loads. scikit-learn is for the leak command.
uv pip install -p "$VENV" torch transformers datasets tqdm python-dotenv accelerate bitsandbytes scikit-learn
PY="$VENV/bin/python"

echo "== CUDA =="
"$PY" -c "
import torch, transformers
assert torch.cuda.is_available(), (
    'torch ' + torch.__version__ + ' cannot see the GPU. '
    'The host NVIDIA driver is probably too old for this CUDA build. '
    'Rent an instance with a newer driver.')
print('torch', torch.__version__, '| transformers', transformers.__version__)
print(torch.cuda.device_count(), 'GPU:', torch.cuda.get_device_name(0))
"

echo "== Datasets =="
# Same names as scripts/build_data.py. This fills the cache that the script reads.
"$PY" -c "
from datasets import load_dataset
print('HaluEval:', load_dataset('pminervini/HaluEval', 'qa_samples'))
print('TriviaQA:', load_dataset('lucadiliello/triviaqa'))
"

if [ "$DOWNLOAD_MODELS" = 1 ]; then
    echo "== Models =="
    # Both models are gated: a failed download warns but does not stop setup.
    "$PY" -c "
import sys
sys.argv = ['build_data.py']
sys.path.insert(0, 'scripts')
from huggingface_hub import snapshot_download
from build_data import NEGATIVES_MODEL, TRANSLATOR_MODEL
for repo in (NEGATIVES_MODEL, TRANSLATOR_MODEL):
    try:
        snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
        print('ok     ', repo)
    except Exception as e:
        print('FAILED ', repo, '-', type(e).__name__, str(e)[:200])
        print('        If the model is gated, accept its license at https://huggingface.co/' + repo)
"
fi

run="$PY scripts/build_data.py"
[ -n "${HF_HOME:-}" ] && run="HF_HOME=$HF_HOME $run"
echo
echo "Setup done. Run, in this order:"
echo "  $run triviaqa-en"
echo "  $run leak      --dataset triviaqa --lang en    # must be near 0.5 before translating"
echo "  $run check     --dataset triviaqa --lang vi"
echo "  $run translate --dataset triviaqa --lang vi"
