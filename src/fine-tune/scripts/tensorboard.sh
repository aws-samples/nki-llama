#!/usr/bin/env bash
set -e

echo "==== Starting TensorBoard for model training visualization ===="

echo "==== Finding latest experiment directory ===="
# Find latest experiment dir
EXP_DIR=$(ls -dt ./training_repo/examples/nemo_experiments/hf_llama3_8B/* | head -1)
echo "Latest experiment directory found: $EXP_DIR"

echo "==== Changing to experiment directory ===="
cd $EXP_DIR
echo "Current directory: $(pwd)"

echo "==== Launching TensorBoard on port 6006 ===="
# Launch TensorBoard
tensorboard --logdir . --port 6006

# Note: This path appears to be unnecessary or might cause an error since we're already in the experiment directory
echo "==== TensorBoard session ended ===="