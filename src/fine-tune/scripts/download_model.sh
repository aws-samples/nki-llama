#!/usr/bin/env bash
set -e

# Go to your fine-tune workspace
cd ~/nki-llama/src/fine-tune

# Paths
export TOKENIZER_DIR=~/nki-llama/src/fine-tune/model_assets/llama_tokenizer
export MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/llama_3-1_8b
export BIN_MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/llama3-8B_hf_weights_bin
export CONSOLIDATED_BIN_MODEL_DIR=~nki-llama/src/fine-tune/model_assets/pckpt/
# Create the output directories if needed
mkdir -p "${TOKENIZER_DIR}" "${MODEL_DIR}" "${BIN_MODEL_DIR}" "${CONSOLIDATED_BIN_MODEL_DIR}"

# 1. Download from Hugging Face via Python (so we can save in HF style)
python - <<'PYCODE'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer_dir = os.path.expanduser(os.environ["TOKENIZER_DIR"])
model_dir     = os.path.expanduser(os.environ["MODEL_DIR"])

# Download & save
AutoTokenizer.from_pretrained(model_id).save_pretrained(tokenizer_dir)
AutoModelForCausalLM.from_pretrained(model_id).save_pretrained(model_dir)
PYCODE

# 2. Convert any .safetensors shards to classic .bin
python ~/nki-llama/src/fine-tune/scripts/convert_safetensors.py \
    --input_dir  "${MODEL_DIR}" \
    --output_dir "${BIN_MODEL_DIR}"

# 3. needs to be one .bin file
python ~/nki-llama/src/fine-tune/scripts/merge_checkpoints.py

