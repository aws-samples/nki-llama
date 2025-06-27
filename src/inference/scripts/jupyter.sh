#!/bin/bash
# jupyter.sh - Jupyter Lab setup and launcher

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

# Setup Jupyter environment
setup_jupyter() {
    echo -e "${BLUE}Setting up Jupyter environment...${NC}"
    
    # Create virtual environment if needed
    if [[ ! -d "$JUPYTER_VENV" ]]; then
        echo "Creating virtual environment..."
        python3 -m venv "$JUPYTER_VENV"
    fi
    
    # Activate and install packages
    source "${JUPYTER_VENV}/bin/activate"
    
    echo "Installing Jupyter packages..."
    pip install --upgrade pip
    pip install jupyter jupyterlab ipykernel python-dotenv
    pip install langchain langgraph langchain_community
    
    # Install kernel
    echo "Installing Jupyter kernel..."
    python -m ipykernel install --user \
        --name="nki-llama" \
        --display-name="Python (NKI-LLAMA)"
    
    echo -e "${GREEN}✓ Jupyter setup complete${NC}"
}

# Start Jupyter Lab
start_jupyter() {
    # Check if setup is needed
    if [[ ! -d "$JUPYTER_VENV" ]]; then
        echo -e "${YELLOW}Jupyter not set up. Running setup first...${NC}"
        setup_jupyter
    fi
    
    # Activate environment
    source "${JUPYTER_VENV}/bin/activate"
    
    # Start Jupyter Lab
    echo -e "${GREEN}Starting Jupyter Lab on port ${JUPYTER_PORT}...${NC}"
    echo -e "${YELLOW}URL: http://0.0.0.0:${JUPYTER_PORT}${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"
    
    cd "$NKI_ROOT"
    jupyter lab --no-browser --ip="0.0.0.0" --port="${JUPYTER_PORT}"
}

# Main
case "${1:-start}" in
    setup)
        setup_jupyter
        ;;
    start|"")
        start_jupyter
        ;;
    *)
        echo "Usage: $0 [setup|start]"
        exit 1
        ;;
esac