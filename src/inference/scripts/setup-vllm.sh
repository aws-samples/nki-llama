#!/bin/bash
# setup-vllm.sh - Setup vLLM for Neuron inference

set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../nki-llama.config"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Setting up vLLM for Neuron...${NC}"

# Check if in correct environment
if [[ "$VIRTUAL_ENV" != *"inference"* ]]; then
    echo -e "${RED}Error: Not in inference environment${NC}"
    echo -e "Run: source ${NEURON_INFERENCE_VENV}/bin/activate"
    exit 1
fi

# Clone or update vLLM repository
if [[ -d "$VLLM_REPO" ]]; then
    echo "Updating existing vLLM repository..."
    cd "$VLLM_REPO"
    git fetch
    git checkout "$VLLM_BRANCH"
    git pull
else
    echo "Cloning vLLM repository..."
    cd "$(dirname "$VLLM_REPO")"
    git clone -b "$VLLM_BRANCH" https://github.com/aws-neuron/upstreaming-to-vllm.git
fi

# Install requirements
cd "$VLLM_REPO"
echo "Installing vLLM requirements..."
pip install -r requirements-neuron.txt

# Install vLLM
echo "Installing vLLM for Neuron..."
VLLM_TARGET_DEVICE="neuron" pip install -e .

echo -e "${GREEN}✓ vLLM setup complete${NC}"