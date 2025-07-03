#!/bin/bash
set -e

# Define paths
CHECKPOINT_PATH="/home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/nemo_experiments/hf_llama/2025-06-11_15-11-22/checkpoints/hf_llama3_8B_SFT--step=5000-consumed_samples=319424.0.ckpt/"
BASE_MODEL_PATH="/home/ubuntu/nki-llama/src/fine-tune/model_assets/llama_3-1_8b/"
OUTPUT_PATH="/home/ubuntu/nki-llama/merged_model/"
MERGE_SCRIPT="/home/ubuntu/nki-llama/src/fine-tune/merge_lora_checkpoint.py"

# Ensure output directory exists
mkdir -p "${OUTPUT_PATH}"

echo "=== LoRA Model Merging Process ==="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Base Model: ${BASE_MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo ""

# Activate the training environment (needed for XLA utilities)
echo "Activating NeuronX training environment..."
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Download packages
echo "=== Downloading Libararies! ==="
pip install transformers
pip install accelerate
echo "=== Download Complete! ==="

# Run the LoRA merge script
echo "Merging LoRA weights with base model..."
python3 ${MERGE_SCRIPT} \
    --checkpoint_dir "${CHECKPOINT_PATH}/model" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --output_dir "${OUTPUT_PATH}" \
    --tp_size 32

echo ""
echo "=== LoRA Model Merge Complete! ==="
echo "Your merged model is ready at: ${OUTPUT_PATH}"
echo ""
echo "You can now use this model directly with transformers:"
echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "  model = AutoModelForCausalLM.from_pretrained('${OUTPUT_PATH}')"
echo "  tokenizer = AutoTokenizer.from_pretrained('${OUTPUT_PATH}')"

# Deactivate environment
deactivate