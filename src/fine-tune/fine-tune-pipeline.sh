#!/usr/bin/env bash
# pipeline.sh 
# Usage: ./pipeline.sh [all|deps|data|model|convert_ckpt|precompile|train|clean]
set -euo pipefail

###############################################################################
# 1. Bring in environment variables from ../../.env (if it exists)
###############################################################################
ENV_FILE="$(dirname "$0")/../../.env"
if [[ -f "$ENV_FILE" ]]; then
  # Export every variable defined in the .env file
  set -a
  
  source "$ENV_FILE"
  set +a
fi

###############################################################################
# 2. Helper for neuron venv
###############################################################################
check_neuron_venv() {
  if [[ -z "${VIRTUAL_ENV:-}" || "$VIRTUAL_ENV" != *"neuronx"* ]]; then
    echo "Not inside a Neuron virtual environment."
    echo "    Run:  source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate"
    exit 1
  fi
  echo "Using Neuron virtual environment: $VIRTUAL_ENV"
}

###############################################################################
# 3. Pipeline steps (one Bash function per step)
###############################################################################
deps()        { check_neuron_venv; echo "==> Installing/validating deps…"; bash scripts/bootstrap.sh; }
data()        { check_neuron_venv; echo "==> Downloading dataset…";       bash scripts/download_data.sh; }
model()       { check_neuron_venv; echo "==> Fetching model…";            HF_TOKEN="${HF_TOKEN:-}" MODEL_ID="${MODEL_ID:-}" bash scripts/download_model.sh; }
convert_ckpt(){ check_neuron_venv; echo "==> Converting ckpt…";           bash scripts/convert_checkpoints.sh; }
precompile()  { check_neuron_venv; echo "==> Pre-compiling graphs…";       bash scripts/precompile.sh; }
train()       { check_neuron_venv; echo "==> Running fine-tune…";          bash scripts/run_training.sh; }
clean()       { check_neuron_venv; rm -rf dataset llama3_tokenizer llama3-8B_hf_weights pretrained_ckpt nemo_experiments; echo "🧹  Cleaned generated files."; }

# run as unit test
all() { deps; data; model; convert_ckpt; precompile; train; }

###############################################################################
# 4. Argument parsing
###############################################################################
main() {
  local cmd="${1:-all}"
  case "$cmd" in
    all|deps|data|model|convert_ckpt|precompile|train|clean) "$cmd" ;;
    *)  echo "Usage: $0 {all|deps|data|model|convert_ckpt|precompile|train|clean}" >&2; exit 1 ;;
  esac
}

main "$@"


