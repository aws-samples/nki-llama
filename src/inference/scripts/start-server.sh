#!/bin/bash
# start-server.sh - Start vLLM OpenAI-compatible API server

set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../nki-llama.config"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting vLLM API Server${NC}"
echo -e "${BLUE}Model: ${MODEL_NAME}${NC}"
echo -e "${BLUE}Port: ${INFERENCE_PORT}${NC}"
echo -e "${BLUE}Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}${NC}"
echo

# Check model exists
if [[ ! -d "${NKI_MODELS}/${MODEL_NAME}" ]]; then
    echo -e "${RED}Error: Model not found at ${NKI_MODELS}/${MODEL_NAME}${NC}"
    echo "Run: ./nki-llama.sh inference download"
    exit 1
fi

# Set Neuron environment variables
export VLLM_NEURON_FRAMEWORK="${VLLM_NEURON_FRAMEWORK}"
export NEURON_COMPILED_ARTIFACTS="${NKI_COMPILED}/${MODEL_NAME}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES}"

# Create compiled model directory if needed
mkdir -p "$NEURON_COMPILED_ARTIFACTS"

echo -e "${YELLOW}Starting server on http://0.0.0.0:${INFERENCE_PORT}${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo

# Start vLLM server
#if using a reasoning model, make sure 
cd "$HOME"
python -m vllm.entrypoints.openai.api_server \
    --model="${NKI_MODELS}/${MODEL_NAME}" \
    --max-num-seqs="${MAX_NUM_SEQS}" \
    --max-model-len="${MAX_MODEL_LEN}" \
    --tensor-parallel-size="${TENSOR_PARALLEL_SIZE}" \
    --port="${INFERENCE_PORT}" \
    --device="neuron" \
    --override-neuron-config='{"enable_bucketing":false}'