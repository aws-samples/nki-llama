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
    git pull
else
    echo "Cloning vLLM repository..."
    cd "$(dirname "$VLLM_REPO")"
    git clone -b neuron-2.22-vllm-v0.7.2 https://github.com/aws-neuron/upstreaming-to-vllm.git
fi

# ---- NEW: make sure no wheel shadows the editable install ---------------
echo "Removing any previously installed vLLM wheels..."
pip uninstall -y vllm vllm-nightly vllm-neuron 2>/dev/null || true

# Install requirements
cd /home/ubuntu/upstreaming-to-vllm/
echo "Installing vLLM requirements..."
pip install -r requirements-neuron.txt

# Install vLLM
echo "Installing vLLM for Neuron..."
VLLM_TARGET_DEVICE="neuron" pip install -e .

# Ensure transformers < 4.50 (needed by Neuron hf_adapter)
python - <<'PY'
import subprocess, pkg_resources, sys
req = "4.48.2"
try:
    ver = pkg_resources.get_distribution("transformers").version
except pkg_resources.DistributionNotFound:
    ver = ""
if not ver or pkg_resources.parse_version(ver) >= pkg_resources.parse_version(req):
    print("Installing transformers<%s …" % req)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"transformers<{req}"])
PY

echo -e "${GREEN}✓ vLLM setup complete${NC}"