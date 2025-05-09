#!/usr/bin/env bash
set -e

# Find latest experiment dir
EXP_DIR=$(ls -dt ./training_repo/examples/nemo_experiments/hf_llama3_8B/* | head -1)
cd $EXP_DIR

# Launch TensorBoard
tensorboard --logdir . --port 6006
~/neuronx-distributed-training/examples/nemo_experiments/hf_llama3_8B/