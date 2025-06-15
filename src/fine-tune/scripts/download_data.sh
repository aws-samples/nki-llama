#!/usr/bin/env bash
set -e

echo "==== Starting dataset download script ===="

echo "==== Setting dataset directory path ===="
DATA_DIR=~/nki-llama/src/fine-tune/datasets/llama3-1_8B
export DATA_DIR
echo "Dataset directory set to: $DATA_DIR"

echo "==== Creating dataset directory if it doesn't exist ===="
mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
echo "Changed to directory: $(pwd)"

echo "==== Downloading Dolly SFT training dataset from S3 ===="
# Download Dolly SFT dataset from S3 (no sign-request)
aws s3 cp s3://neuron-s3/training_datasets/llama/sft/training.jsonl ${DATA_DIR}/training.jsonl --no-sign-request
echo "Training dataset downloaded to ${DATA_DIR}/training.jsonl"

echo "==== Downloading Dolly SFT validation dataset from S3 ===="
aws s3 cp s3://neuron-s3/training_datasets/llama/sft/validation.jsonl ${DATA_DIR}/validation.jsonl --no-sign-request
echo "Validation dataset downloaded to ${DATA_DIR}/validation.jsonl"

echo "==== Dataset download complete! ===="