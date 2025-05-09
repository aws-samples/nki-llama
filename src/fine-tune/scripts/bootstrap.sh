#!/usr/bin/env bash
set -e

# 1. Activate your Neuron venv
#source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate

# 2. Upgrade pip, neuron packages active cause venv enabled, cython error - ensure instaled
pip install -U pip
pip install Cython
pip install --upgrade setuptools


# 3. Build a slim CPU-only Apex (NeMo dependency)
git clone https://github.com/NVIDIA/apex.git apex
cd apex
git checkout 23.05

# 3a. Overwrite setup.py with the nxd specfic setup
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
    install_requires=["packaging>20.6"],
    description="PyTorch Extensions written by NVIDIA",
)
EOF

# 3b. Install build prerequisites and make the wheel
pip install packaging wheel
python setup.py bdist_wheel

# Copy out the generated wheel path for later
APEX_WHEEL=$(ls ~/nki-llama/src/fine-tune/apex/dist/apex-*.whl | head -1)

# 3c. Return to root of our example
cd ..

# 4. Install the rest of NxDT’s Python requirements
#    (this includes NeMo, PyTorch-Lightning, etc.)
wget -q https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-training/master/requirements.txt
pip install -r requirements.txt

# 5. Finally, install our freshly built Apex wheel
pip install "$APEX_WHEEL"

# 6. And install the NxDT training framework itself
pip install neuronx_distributed_training --extra-index-url https://pip.repos.neuron.amazonaws.com
