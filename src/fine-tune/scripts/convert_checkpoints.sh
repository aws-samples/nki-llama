#!/usr/bin/env bash
set -e

echo "==== Starting NxDT checkpoint conversion script ===="

echo "==== Changing to fine-tune directory ===="
cd ~/nki-llama/src/fine-tune/

if [ -d "/home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training" ]; then
    echo "==== Removing existing NxDT repository to ensure latest version ===="
    rm -rf neuronx-distributed-training
fi

echo "==== Cloning neuronx-distributed-training repository ===="
git clone https://github.com/aws-neuron/neuronx-distributed-training.git

echo "==== Changing to checkpoint converter scripts directory ===="
cd ~/nki-llama/src/fine-tune/neuronx-distributed-training/examples/checkpoint_converter_scripts

echo "==== Setting environment variables for conversion ===="
# Use NxDT's checkpoint converter to produce sharded NxDT checkpoint
export NXDT_MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/converted_hf_style_hf_to_nxdt_tp8pp4/
export MODEL_CONFIG=~/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json
export CONSOLIDATED_BIN_MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/pckpt/pytorch_model.bin

echo "==== Environment variables set: ===="
echo "NXDT_MODEL_DIR: $NXDT_MODEL_DIR"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "CONSOLIDATED_BIN_MODEL_DIR: $CONSOLIDATED_BIN_MODEL_DIR"

echo "==== Running checkpoint converter ===="
echo "==== Converting from HF to NxDT format with TP=32, PP=1 ===="
python3 checkpoint_converter.py \
    --model_style             hf \
    --hw_backend              trn1 \
    --input_dir               $CONSOLIDATED_BIN_MODEL_DIR \
    --output_dir              $NXDT_MODEL_DIR \
    --save_xser               True \
    --config                  $MODEL_CONFIG \
    --tp_size                 32 \
    --pp_size                 1 \
    --n_layers                32 \
    --kv_size_multiplier      4 \
    --qkv_linear              True \
    --convert_from_full_state

echo "==== Checkpoint conversion complete! ====="