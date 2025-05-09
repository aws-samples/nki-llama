#!/usr/bin/env bash
set -e

cd ~/nki-llama/src/fine-tune/neuronx-distributed-training/examples
# Point to our config path and file
export CONF_FILE=hf_llama3.1_8B_SFT_lora_config
export COMPILE=0

bash train.sh
