#!/usr/bin/env bash
# Enhanced bootstrap script with error handling and resilience

# Don't exit immediately on error - we want to handle errors gracefully
set +e

LOG_FILE="bootstrap_$(date +%Y%m%d_%H%M%S).log"
MAX_RETRIES=3
REQUIREMENTS_URL="https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-training/master/requirements.txt"
LOCAL_REQUIREMENTS="./cached_requirements.txt"

log_message() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

run_with_retry() {
    local cmd="$1"
    local desc="$2"
    local critical="${3:-non-critical}"  # Default to non-critical
    local attempts=0
    local success=false

    log_message "Starting: $desc"
    
    while [ $attempts -lt $MAX_RETRIES ] && [ "$success" = false ]; do
        attempts=$((attempts+1))
        log_message "Attempt $attempts of $MAX_RETRIES: $desc"
        
        # Create a temporary file for capturing output
        local temp_output=$(mktemp)
        
        # Run command and capture both stdout and stderr
        if eval "$cmd" > >(tee -a "$temp_output" "$LOG_FILE") 2>&1; then
            success=true
            log_message "SUCCESS: $desc"
        else
            if [ $attempts -lt $MAX_RETRIES ]; then
                log_message "FAILED: $desc - Retrying in 5 seconds..."
                sleep 5
            else
                log_message "FAILED: $desc - All attempts exhausted"
                if [ "$critical" = "critical" ]; then
                    log_message "Critical failure, exiting script"
                    rm -f "$temp_output"
                    exit 1
                fi
            fi
        fi
        
        rm -f "$temp_output"
    done
    
    if [ "$success" = true ]; then
        return 0
    else
        return 1
    fi
}

log_message "==== Starting NeuronX setup script ===="

# Save the original directory
ORIGINAL_DIR=$(pwd)

# 1. Activate your Neuron venv
#source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate
log_message "==== Neuron venv activation commented out - uncomment if needed ===="

# 2. Upgrade pip, neuron packages active cause venv enabled, cython error - ensure installed
log_message "==== Upgrading pip and installing prerequisites ===="
run_with_retry "pip install -U pip" "Upgrading pip" "critical"
run_with_retry "pip install Cython" "Installing Cython" "critical"
run_with_retry "pip install --upgrade setuptools" "Upgrading setuptools" "critical"

# 3. Build a slim CPU-only Apex (NeMo dependency)
log_message "==== Checking for Apex repository ===="
if [ ! -d "apex" ]; then
    log_message "==== Cloning Apex repository ===="
    run_with_retry "git clone https://github.com/NVIDIA/apex.git apex" "Cloning Apex" "critical"
else
    log_message "==== Apex repository already exists, checking if we can update it ===="
    cd apex
    if git rev-parse --git-dir > /dev/null 2>&1; then
        log_message "Valid git repository found, fetching updates"
        git fetch origin
    fi
    cd "$ORIGINAL_DIR"
fi

cd apex || { log_message "Failed to change directory to apex"; exit 1; }
log_message "==== Checking out Apex version 23.05 ===="
run_with_retry "git checkout 23.05" "Checkout Apex 23.05" "critical"

# 3a. Overwrite setup.py with the nxd specific setup
log_message "==== Creating custom setup.py for CPU-only Apex ===="
cat > setup.py <<'EOF'
import sys
import warnings
import os
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, load

setup(
    name="apex",
    version="0.1",
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info",)
    ),
    install_requires=["packaging>20.6",],
    description="PyTorch Extensions written by NVIDIA",
)
EOF

# Clean up any previous build artifacts
log_message "==== Cleaning up previous build artifacts ===="
rm -rf build dist *.egg-info

# 3b. Install build prerequisites and make the wheel
log_message "==== Installing packaging and wheel for building Apex ===="
run_with_retry "pip install packaging wheel" "Installing packaging and wheel" "critical"
log_message "==== Building Apex wheel ===="
run_with_retry "python setup.py bdist_wheel" "Building Apex wheel" "critical"

# Copy out the generated wheel path for later
log_message "==== Saving Apex wheel path ===="
APEX_WHEEL=$(find "$(pwd)"/dist -name "apex-*.whl" -type f 2>/dev/null | head -1)
if [ -z "$APEX_WHEEL" ] || [ ! -f "$APEX_WHEEL" ]; then
    log_message "ERROR: Failed to find Apex wheel"
    exit 1
fi
log_message "Apex wheel: $APEX_WHEEL"

# 3c. Return to root of our example
log_message "==== Returning to script directory ===="
cd "$ORIGINAL_DIR"

# 4. Improved requirements.txt handling with caching and batch installation
log_message "==== Handling NxDT requirements.txt ===="

# Try to download fresh requirements, fall back to cached if available
if ! wget -q --timeout=30 "$REQUIREMENTS_URL" -O requirements.txt.new 2>/dev/null; then
    log_message "WARNING: Failed to download fresh requirements.txt"
    if [ -f "$LOCAL_REQUIREMENTS" ]; then
        log_message "Using cached requirements file"
        cp "$LOCAL_REQUIREMENTS" requirements.txt.new
    else
        log_message "ERROR: No requirements file available"
        exit 1
    fi
else
    # Cache the successfully downloaded requirements
    cp requirements.txt.new "$LOCAL_REQUIREMENTS"
    log_message "Successfully downloaded and cached requirements.txt"
fi

# Process requirements in batches for better error handling
log_message "==== Installing NxDT dependencies in batches ===="

# Clean and prepare requirements
grep -v "^#" requirements.txt.new | grep -v "^$" | grep -v "^[[:space:]]*$" > requirements.txt.clean

# Check if we have any requirements to install
if [ ! -s requirements.txt.clean ]; then
    log_message "WARNING: No requirements found to install"
else
    # Create temporary directory for batch files
    BATCH_DIR=$(mktemp -d)
    
    # Split into batches (5 packages per batch)
    split -l 5 requirements.txt.clean "$BATCH_DIR/req_batch_"
    
    # Install requirements in batches
    for batch in "$BATCH_DIR"/req_batch_*; do
        if [ -f "$batch" ]; then
            log_message "Installing batch: $(basename "$batch")"
            # Show what we're installing
            cat "$batch" | tee -a "$LOG_FILE"
            
            if ! run_with_retry "pip install -r '$batch'" "Installing $(basename "$batch")"; then
                log_message "WARNING: Failed to install some packages in $(basename "$batch")"
                # Continue anyway - we'll check key packages later
            fi
        fi
    done
    
    # Clean up batch directory
    rm -rf "$BATCH_DIR"
fi

# Clean up temporary files
rm -f requirements.txt.clean requirements.txt.new

# 5. Finally, install our freshly built Apex wheel
log_message "==== Installing Apex wheel ===="
run_with_retry "pip install '$APEX_WHEEL'" "Installing Apex wheel" "critical"

# 6. And install the NxDT training framework itself
log_message "==== Installing neuronx_distributed_training package ===="
run_with_retry "pip install neuronx_distributed_training --extra-index-url https://pip.repos.neuron.amazonaws.com" "Installing neuronx_distributed_training" "critical"

# 7. Verify critical dependencies
log_message "==== Verifying critical dependencies ===="
missing_deps=()

check_package() {
    local package="$1"
    if ! pip show "$package" >/dev/null 2>&1; then
        missing_deps+=("$package")
        log_message "WARNING: Required package $package is missing"
        return 1
    else
        local version=$(pip show "$package" | grep Version | cut -d' ' -f2)
        log_message "Package $package is installed (version: $version)"
        return 0
    fi
}

# Check essential packages
check_package "apex"
check_package "neuronx_distributed_training"
check_package "torch-neuronx"

# Report results
if [ ${#missing_deps[@]} -eq 0 ]; then
    log_message "==== All critical dependencies installed successfully! ===="
else
    log_message "==== WARNING: Some critical dependencies are missing: ${missing_deps[*]} ===="
    log_message "==== You may need to install them manually or troubleshoot the installation ===="
    # Don't exit with error - let the calling script decide
fi

log_message "==== Setup complete! Log saved to $LOG_FILE ===="
log_message "==== Final working directory: $(pwd) ===="