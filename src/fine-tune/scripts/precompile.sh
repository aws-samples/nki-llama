#!/usr/bin/env bash
set -e

# ==============================================================================
# NxDT Model Compilation Script
# ==============================================================================
# This script sets up and compiles NeuronX Distributed Training models
# 
# Usage:
#   ./precompile.sh                     # Run setup and compile
#   
# Requirements:
#   - Python 3.x with pip
#   - Git (for cloning repositories)
#   - install_setup.sh script in the same directory
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

log "==== Step 4: Running install_setup.sh ===="
INSTALL_SETUP_DIR=~/nki-llama/src/fine-tune/neuronx-distributed-training/install_setup.sh

# Check if install_setup.sh exists
if [ ! -f $INSTALL_SETUP_DIR ]; then
    log_error "install_setup.sh script not found in current directory: $(pwd)"
    log "Available files:"
    ls -la
    exit 1
fi

# Make sure install_setup.sh is executable
chmod +x $INSTALL_SETUP_DIR

log "Running install_setup.sh to install dependencies..."
log "This will install megatron-core and apply necessary patches..."

# Run install_setup.sh with output logging
if $INSTALL_SETUP_DIR 2>&1 | tee /tmp/install_setup_output.log; then
    log "✓ install_setup.sh completed successfully!"
else
    log_error "install_setup.sh failed"
    log_error "Setup log saved to: /tmp/install_setup_output.log"
    log_error "Last 20 lines of setup log:"
    tail -20 /tmp/install_setup_output.log >&2
    exit 1
fi

# Verify megatron-core installation
if pip show megatron-core >/dev/null 2>&1; then
    MCORE_VERSION=$(pip show megatron-core | grep Version | cut -d' ' -f2)
    log "✓ Verified megatron-core installation - version: $MCORE_VERSION"
else
    log_error "megatron-core not found after running install_setup.sh"
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
log "  - Setup log: /tmp/install_setup_output.log"
log "  - Training log: /tmp/train_output.log"