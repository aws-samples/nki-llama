#!/bin/bash
# download-model.sh - Download model from Hugging Face

set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../nki-llama.config"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Downloading model from Hugging Face...${NC}"

# Check HF token
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo -e "${YELLOW}HF_TOKEN not set${NC}"
    echo "Get a token at: https://huggingface.co/settings/tokens"
    read -p "Enter your Hugging Face token: " HF_TOKEN
    if [[ -z "$HF_TOKEN" ]]; then
        echo -e "${RED}Error: HF_TOKEN is required${NC}"
        exit 1
    fi
fi

# Ensure huggingface-cli is installed
pip install -q huggingface_hub[cli]

# Ensure transformers < 4.50 (needed by Neuron hf_adapter)
python - <<'PY'
import subprocess, pkg_resources, sys
req = "4.50.0"
try:
    ver = pkg_resources.get_distribution("transformers").version
except pkg_resources.DistributionNotFound:
    ver = ""
if not ver or pkg_resources.parse_version(ver) >= pkg_resources.parse_version(req):
    print("Installing transformers<%s …" % req)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"transformers<{req}"])
PY

# Create models directory
mkdir -p "$NKI_MODELS"

# Download model
echo "Downloading ${MODEL_ID} to ${NKI_MODELS}/${MODEL_NAME}"
huggingface-cli download \
    --token "$HF_TOKEN" \
    "$MODEL_ID" \
    --local-dir "${NKI_MODELS}/${MODEL_NAME}"

echo -e "${GREEN}✓ Model downloaded successfully${NC}"
echo "Location: ${NKI_MODELS}/${MODEL_NAME}"

# Save configuration hint
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo
    echo "To save your token, add to .env file:"
    echo "HF_TOKEN=$HF_TOKEN"
fi