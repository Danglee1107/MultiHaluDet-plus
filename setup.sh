#!/usr/bin/env bash
# One-time setup on a vast.ai instance: dependencies, datasets, models.
# Usage: HF_TOKEN=hf_xxx ./setup.sh [--no-models]
set -euo pipefail
cd "$(dirname "$0")"

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

echo "== Disk =="
free_gb=$(df -BG --output=avail "${HF_HOME:-$HOME}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "${free_gb} GB free"
if [ "$DOWNLOAD_MODELS" = 1 ] && [ "${free_gb:-0}" -lt 40 ]; then
    echo "WARNING: both models need about 30 GB. Less than 40 GB is free."
fi

echo "== HF_TOKEN =="
# .env is gitignored, so a fresh clone does not have it.
if [ ! -f .env ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: no .env file and HF_TOKEN is not set." >&2
        echo "Run: HF_TOKEN=hf_xxx ./setup.sh" >&2
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
uv sync --frozen

echo "== CUDA =="
# The lock file pins a CUDA 13 torch build; an old host driver fails here.
uv run python -c "
import torch
assert torch.cuda.is_available(), (
    'torch ' + torch.__version__ + ' cannot see the GPU. '
    'The host NVIDIA driver is probably too old for this CUDA build. '
    'Rent an instance with a newer driver.')
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))
"

echo "== Datasets =="
# Same names as src/data/loader.py. This fills the cache that the loader reads.
uv run python -c "
from datasets import load_dataset
print('HaluEval:', load_dataset('pminervini/HaluEval', 'qa_samples'))
print('TriviaQA:', load_dataset('lucadiliello/triviaqa'))
"

if [ "$DOWNLOAD_MODELS" = 1 ]; then
    echo "== Models =="
    # A failed model (for example the gated Llama-2) warns but does not stop setup.
    uv run python -c "
from huggingface_hub import snapshot_download
from src.config import MODEL_REGISTRY
for name, repo in MODEL_REGISTRY.items():
    try:
        snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])
        print('ok     ', name, repo)
    except Exception as e:
        print('FAILED ', name, repo, '-', type(e).__name__, str(e)[:200])
        print('        If the model is gated, accept its license at https://huggingface.co/' + repo)
"
fi

echo
echo "Setup done. Run: ./run.sh"
