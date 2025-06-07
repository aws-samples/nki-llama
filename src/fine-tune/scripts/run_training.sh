#!/usr/bin/env bash
set -e

echo "==== Starting NxDT model training script ===="

echo "==== Navigating to NxDT examples directory ===="
cd ~/nki-llama/src/fine-tune/neuronx-distributed-training/examples
echo "Current directory: $(pwd)"

echo "==== Setting configuration variables ===="
# Point to our config path and file
export CONF_FILE=hf_llama3.1_8B_SFT_lora_config
export COMPILE=0
echo "Configuration file set to: $CONF_FILE"
echo "COMPILE flag set to: $COMPILE (will perform actual training, not just compilation)"

echo "==== Starting model training process ===="
bash train.sh

echo "==== Model training process complete! ===="