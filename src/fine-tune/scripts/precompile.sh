#!/usr/bin/env bash
set -e

# ==============================================================================
# NxDT Model Compilation Script
# ==============================================================================
# This script sets up and compiles NeuronX Distributed Training models
# 
# Usage:
#   ./script.sh                                    # Install both packages
#   SKIP_TRANSFORMER_ENGINE=true ./script.sh      # Skip transformer-engine
#   
# Environment Variables:
#   SKIP_TRANSFORMER_ENGINE=true    Skip transformer-engine installation
#                                   (useful for non-CUDA environments)
#
# Requirements:
#   - Python 3.x with pip
#   - Git (for cloning repositories)
#   - For transformer-engine: CUDA toolkit (optional, fallback available)
# ==============================================================================

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to log errors
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Function to log command execution
run_cmd() {
    log "Executing: $*"
    "$@"
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log "✓ Command completed successfully"
    else
        log_error "✗ Command failed with exit code $exit_code"
        exit $exit_code
    fi
}

log "==== Starting NxDT model compilation script ===="
log "Script started by user: $(whoami)"
log "Working directory: $(pwd)"

log "==== Step 1: Navigating to NxDT examples directory ===="
TARGET_DIR=~/nki-llama/src/fine-tune/neuronx-distributed-training/examples

if [ ! -d "$TARGET_DIR" ]; then
    log_error "Target directory does not exist: $TARGET_DIR"
    exit 1
fi

run_cmd cd "$TARGET_DIR"
log "✓ Successfully navigated to: $(pwd)"

log "==== Step 2: Copying configuration files ===="
CONFIG_SOURCE=~/nki-llama/src/fine-tune/configs/YAML
CONFIG_DEST=~/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf

if [ ! -d "$CONFIG_SOURCE" ]; then
    log_error "Configuration source directory does not exist: $CONFIG_SOURCE"
    exit 1
fi

if [ ! -d "$CONFIG_DEST" ]; then
    log_error "Configuration destination directory does not exist: $CONFIG_DEST"
    exit 1
fi

# Check if YAML files exist
YAML_COUNT=$(find "$CONFIG_SOURCE" -name "*.yaml" | wc -l)
log "Found $YAML_COUNT YAML configuration files to copy"

if [ "$YAML_COUNT" -eq 0 ]; then
    log_error "No YAML files found in source directory: $CONFIG_SOURCE"
    exit 1
fi

run_cmd cp -v "$CONFIG_SOURCE"/*.yaml "$CONFIG_DEST"/
log "✓ Configuration files copied successfully"

log "==== Step 3: Setting configuration variables ===="
export CONF_FILE=hf_llama3.1_8B_SFT_lora_config
export COMPILE=1

log "Environment variables set:"
log "  CONF_FILE: $CONF_FILE"
log "  COMPILE: $COMPILE (AOT compilation mode enabled)"

# Verify the config file exists
CONFIG_FILE_PATH="$CONFIG_DEST/${CONF_FILE}.yaml"
if [ -f "$CONFIG_FILE_PATH" ]; then
    log "✓ Configuration file verified: $CONFIG_FILE_PATH"
else
    log_error "Configuration file not found: $CONFIG_FILE_PATH"
    log "Available configuration files:"
    ls -la "$CONFIG_DEST"/*.yaml 2>/dev/null || log "No YAML files found"
    exit 1
fi

log "==== Step 4: Installing dependencies ===="

# Check if user wants to skip transformer-engine installation
SKIP_TE=${SKIP_TRANSFORMER_ENGINE:-true}
if [ "$SKIP_TE" = "true" ]; then
    log "Skipping transformer-engine installation (SKIP_TRANSFORMER_ENGINE=true)"
    log "Installing only megatron-core from GitHub repository..."
else
    log "Installing transformer-engine and megatron-core from GitHub repositories..."
fi

# Enhanced pip install with detailed logging
if command -v pip >/dev/null 2>&1; then
    log "Using pip version: $(pip --version)"
    
    # Define the specific tags to install (matching the Docker script)
    TE_TAG="7d576ed25266a17a7b651f2c12e8498f67e0baea"
    MCORE_TAG="core_r0.10.0"
    TE_REPO_URL="git+https://github.com/NVIDIA/TransformerEngine.git@${TE_TAG}"
    MEGATRON_REPO_URL="git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_TAG}"
    
    log "Target repositories:"
    log "  TransformerEngine: $TE_REPO_URL"
    log "  Megatron-LM: $MEGATRON_REPO_URL"
    log "Git tags:"
    log "  TE_TAG: $TE_TAG"
    log "  MCORE_TAG: $MCORE_TAG"
    
    # Check and uninstall existing packages
    packages_to_check=("megatron-core")
    if [ "$SKIP_TE" != "true" ]; then
        packages_to_check=("transformer-engine" "megatron-core")
    fi
    
    for package in "${packages_to_check[@]}"; do
        if pip show "$package" >/dev/null 2>&1; then
            INSTALLED_VERSION=$(pip show "$package" | grep Version | cut -d' ' -f2)
            log "$package is already installed (version: $INSTALLED_VERSION)"
            log "Uninstalling existing $package..."
            if pip uninstall -y "$package" 2>&1 | tee -a /tmp/pip_install.log; then
                log "✓ Successfully uninstalled existing $package"
            else
                log_error "Failed to uninstall existing $package"
                exit 1
            fi
        else
            log "$package not currently installed"
        fi
    done
    
    log "Starting pip install from Git repositories..."
    log "This may take several minutes as it needs to clone and build from source..."
    
    # Install transformer-engine only if not skipped
    if [ "$SKIP_TE" != "true" ]; then
    log "Installing transformer-engine @ $TE_REPO_URL"
    
    # Check if we're in a CUDA environment
    if command -v nvcc >/dev/null 2>&1 && [ -n "$CUDA_HOME" ]; then
        log "CUDA environment detected (nvcc found, CUDA_HOME set)"
        TE_INSTALL_CMD="pip install --no-cache-dir --verbose \"transformer-engine @ $TE_REPO_URL\""
    else
        log "Non-CUDA environment detected - attempting CPU-only installation"
        log "Setting environment variables for CPU-only build..."
        
        # Try to find CUDA installation or set minimal environment
        if [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME="/usr/local/cuda"
            log "Found CUDA at /usr/local/cuda, setting CUDA_HOME"
        elif [ -d "/opt/cuda" ]; then
            export CUDA_HOME="/opt/cuda"
            log "Found CUDA at /opt/cuda, setting CUDA_HOME"
        else
            log "No CUDA installation found - this may cause transformer-engine installation to fail"
            log "Attempting installation anyway..."
        fi
        
        # Set environment variables to potentially bypass CUDA requirements
        export NVTE_FRAMEWORK=pytorch
        export NVTE_WITH_USERBUFFERS=0
        
        TE_INSTALL_CMD="pip install --no-cache-dir --verbose \"transformer-engine @ $TE_REPO_URL\""
    fi
    
    log "Executing: $TE_INSTALL_CMD"
    if eval "$TE_INSTALL_CMD" 2>&1 | tee /tmp/pip_install_te.log; then
        if pip show transformer-engine >/dev/null 2>&1; then
            TE_VERSION=$(pip show transformer-engine | grep Version | cut -d' ' -f2)
            log "✓ Successfully installed transformer-engine version: $TE_VERSION"
        else
            log_error "transformer-engine installation appeared to succeed but package not found"
            exit 1
        fi
    else
        log_error "Failed to install transformer-engine from Git repository"
        log_error "This is likely due to CUDA requirements. Attempting alternative approaches..."
        
        # Try installing a pre-built version from PyPI as fallback
        log "Attempting fallback: installing transformer-engine from PyPI..."
        if pip install --no-cache-dir transformer-engine 2>&1 | tee /tmp/pip_install_te_fallback.log; then
            if pip show transformer-engine >/dev/null 2>&1; then
                TE_VERSION=$(pip show transformer-engine | grep Version | cut -d' ' -f2)
                log "✓ Successfully installed transformer-engine version: $TE_VERSION (PyPI fallback)"
            else
                log_error "PyPI fallback also failed"
                exit 1
            fi
        else
            log_error "Both Git and PyPI installation methods failed"
            log_error "You may need to:"
            log_error "1. Install CUDA toolkit and set CUDA_HOME environment variable"
            log_error "2. Use a pre-built Docker container with transformer-engine"
            log_error "3. Skip transformer-engine installation if not required for your use case"
            log_error "Installation logs saved to:"
            log_error "  - Git install: /tmp/pip_install_te.log"
            log_error "  - PyPI fallback: /tmp/pip_install_te_fallback.log"
            exit 1
        fi
    fi
    else
        log "Skipping transformer-engine installation as requested"
        TE_VERSION="skipped"
    fi
    
    # Then install megatron-core
    log "Installing megatron_core @ $MEGATRON_REPO_URL"
    if pip install --no-cache-dir --verbose "megatron_core @ $MEGATRON_REPO_URL" 2>&1 | tee /tmp/pip_install_mc.log; then
        if pip show megatron-core >/dev/null 2>&1; then
            MCORE_VERSION=$(pip show megatron-core | grep Version | cut -d' ' -f2)
            log "✓ Successfully installed megatron-core version: $MCORE_VERSION"
        else
            log_error "megatron-core installation appeared to succeed but package not found"
            exit 1
        fi
    else
        log_error "Failed to install megatron-core from Git repository"
        log_error "Installation log saved to: /tmp/pip_install_mc.log"
        log_error "Last 10 lines of installation log:"
        tail -10 /tmp/pip_install_mc.log >&2
        exit 1
    fi
    
    # Show installation summary
    log "✓ All dependencies installed successfully!"
    log "Installation summary:"
    if [ "$SKIP_TE" != "true" ]; then
        log "  transformer-engine:"
        log "    Version: $TE_VERSION"
        if [ "$TE_VERSION" != "skipped" ]; then
            log "    Git tag/commit: $TE_TAG"
            log "    Location: $(pip show transformer-engine | grep Location | cut -d' ' -f2-)"
        fi
    else
        log "  transformer-engine: skipped (SKIP_TRANSFORMER_ENGINE=true)"
    fi
    log "  megatron-core:"
    log "    Version: $MCORE_VERSION"
    log "    Git tag: $MCORE_TAG"
    log "    Location: $(pip show megatron-core | grep Location | cut -d' ' -f2-)"
    
    # Show package dependencies
    log "Package dependencies:"
    if [ "$SKIP_TE" != "true" ] && [ "$TE_VERSION" != "skipped" ]; then
        log "  transformer-engine requires:"
        pip show transformer-engine | grep Requires | cut -d' ' -f2- | tr ',' '\n' | sed 's/^/    /'
    fi
    log "  megatron-core requires:"
    pip show megatron-core | grep Requires | cut -d' ' -f2- | tr ',' '\n' | sed 's/^/    /'
    
else
    log_error "pip command not found. Please ensure Python and pip are installed."
    exit 1
fi

log "==== Step 5: Running AOT compilation ===="
if [ ! -f "train.sh" ]; then
    log_error "train.sh script not found in current directory: $(pwd)"
    log "Available files:"
    ls -la
    exit 1
fi

log "Starting train.sh for AOT compilation..."
log "This process may take a significant amount of time..."

# Run train.sh with output logging
if bash -x train.sh 2>&1 | tee /tmp/train_output.log; then
    log "✓ AOT compilation completed successfully!"
else
    log_error "AOT compilation failed"
    log_error "Training log saved to: /tmp/train_output.log"
    log_error "Last 20 lines of training log:"
    tail -20 /tmp/train_output.log >&2
    exit 1
fi

log "==== Script execution completed successfully! ===="
log "Total execution time: $SECONDS seconds"
log "Log files created:"
log "  - pip install log: /tmp/pip_install.log"
log "  - training log: /tmp/train_output.log"