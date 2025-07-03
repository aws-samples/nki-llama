#!/usr/bin/env bash
set -e

echo "==== Starting Llama model download and conversion script ===="

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set!"
    echo "Please make sure HF_TOKEN is defined in your .env file"
    exit 1
fi

# Check if MODEL_ID is set
if [ -z "$MODEL_ID" ]; then
    echo "Error: MODEL_ID environment variable is not set!"
    echo "Please make sure MODEL_ID is defined in your .env file"
    exit 1
fi

echo "==== Changing to fine-tune workspace ===="
# Go to your fine-tune workspace
cd ~/nki-llama/src/fine-tune
echo "Current directory: $(pwd)"

echo "==== Setting path variables ===="
# Paths
export TOKENIZER_DIR=~/nki-llama/src/fine-tune/model_assets/llama_tokenizer
export MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/llama_3-1_8b
export BIN_MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/llama3-8B_hf_weights_bin
export CONSOLIDATED_BIN_MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/pckpt/

echo "Tokenizer directory: $TOKENIZER_DIR"
echo "Model directory: $MODEL_DIR"
echo "Binary model directory: $BIN_MODEL_DIR"
echo "Consolidated binary model directory: $CONSOLIDATED_BIN_MODEL_DIR"

echo "==== Creating output directories ===="
# Create the output directories if needed
mkdir -p "${TOKENIZER_DIR}" "${MODEL_DIR}" "${BIN_MODEL_DIR}" "${CONSOLIDATED_BIN_MODEL_DIR}"
echo "Output directories created"

echo "==== Step 1: Downloading model and tokenizer from Hugging Face ===="

# Check if tokenizer already exists
if [ -f "${TOKENIZER_DIR}/tokenizer_config.json" ] && [ -f "${TOKENIZER_DIR}/tokenizer.json" ]; then
    echo "Tokenizer already exists in ${TOKENIZER_DIR}, skipping download..."
else
    echo "Tokenizer not found, downloading..."
fi

# Check if model already exists (checking for common model files)
if [ -f "${MODEL_DIR}/config.json" ] && [ -n "$(ls -A ${MODEL_DIR}/*.safetensors 2>/dev/null || ls -A ${MODEL_DIR}/*.bin 2>/dev/null)" ]; then
    echo "Model already exists in ${MODEL_DIR}, skipping download..."
else
    echo "Model not found, downloading..."
fi

# 1. Download from Hugging Face via Python (so we can save in HF style)
python - <<'PYCODE'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

model_id = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B")
tokenizer_dir = os.path.expanduser(os.environ["TOKENIZER_DIR"])
model_dir     = os.path.expanduser(os.environ["MODEL_DIR"])
hf_token      = os.environ.get("HF_TOKEN")

# Check if tokenizer exists
tokenizer_path = Path(tokenizer_dir)
tokenizer_exists = (tokenizer_path / "tokenizer_config.json").exists() and \
                   (tokenizer_path / "tokenizer.json").exists()

if not tokenizer_exists:
    print(f"Downloading tokenizer from {model_id}...")
    AutoTokenizer.from_pretrained(model_id, token=hf_token).save_pretrained(tokenizer_dir)
    print(f"Tokenizer saved to {tokenizer_dir}")
else:
    print(f"Tokenizer already exists at {tokenizer_dir}, skipping download")

# Check if model exists
model_path = Path(model_dir)
config_exists = (model_path / "config.json").exists()
weights_exist = any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin"))

if not (config_exists and weights_exist):
    print(f"Downloading model from {model_id}...")
    AutoModelForCausalLM.from_pretrained(model_id, token=hf_token).save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")
else:
    print(f"Model already exists at {model_dir}, skipping download")
PYCODE

echo "==== Step 2: Converting .safetensors to .bin format ===="

# Check if .bin files already exist in the output directory
if [ -n "$(ls -A ${BIN_MODEL_DIR}/*.bin 2>/dev/null)" ]; then
    echo "Binary model files already exist in ${BIN_MODEL_DIR}, skipping conversion..."
else
    echo "Converting to .bin format..."
    # 2. Convert any .safetensors shards to classic .bin
    python ~/nki-llama/src/fine-tune/scripts/convert_safetensors.py \
        --input_dir  "${MODEL_DIR}" \
        --output_dir "${BIN_MODEL_DIR}"
    echo "Conversion to .bin format complete"
fi

echo "==== Step 3: Merging checkpoints into a single .bin file ===="

# Check if consolidated model already exists
# Assuming the merged file would be named something like "consolidated.bin" or "pytorch_model.bin"
# Adjust the filename pattern based on what your merge_checkpoints.py creates
if [ -f "${CONSOLIDATED_BIN_MODEL_DIR}/consolidated.bin" ] || \
   [ -f "${CONSOLIDATED_BIN_MODEL_DIR}/pytorch_model.bin" ] || \
   [ -f "${CONSOLIDATED_BIN_MODEL_DIR}/model.bin" ]; then
    echo "Consolidated model file already exists in ${CONSOLIDATED_BIN_MODEL_DIR}, skipping merge..."
else
    echo "Merging checkpoints..."
    # 3. needs to be one .bin file
    python ~/nki-llama/src/fine-tune/scripts/merge_checkpoints.py
    echo "Checkpoint merging complete"
fi

echo "==== Model download and conversion process complete! ===="

# Optional: Print summary of what was done
echo ""
echo "==== Summary ===="
echo "Assets location:"
echo "  Tokenizer: ${TOKENIZER_DIR}"
echo "  Model: ${MODEL_DIR}"
echo "  Binary model: ${BIN_MODEL_DIR}"
echo "  Consolidated model: ${CONSOLIDATED_BIN_MODEL_DIR}"
echo ""
echo "To force re-download, remove the respective directories and run the script again."