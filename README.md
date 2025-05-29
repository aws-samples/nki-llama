# NKI Llama

A unified project for fine-tuning, inference, and agent development of Llama models on AWS Trainium and Inferentia.


## Project Workflow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│   Fine-tune    │────▶│   Inference    │────▶│     Agent      │
│                │     │                │     │  Development   │
│                │     │                │     │                │
└────────────────┘     └────────────────┘     └────────────────┘
```

This project follows a three-stage workflow:
1. **Fine-tune** a model using Neuron hardware with NxD
2. **Inference** using the fine-tuned model with vLLM, NKI compilation, and NxDI (Neuron Distributed Inference)
3. **Agent Development** using LangChain/LangGraph connected to your model

## Technical Infrastructure

### Compute Resources
- **Required Instance**: trn1.32xlarge 
- **Base AMI**: Deep Learning AMI Neuron (Ubuntu 22.04) with Neuron SDK 2.23.
- **Base Packages**:
  - NxD (NeuronX Distributed Training)
  - NKI (Neuron Kernel Interface)
  - NxDI (Neuron Distributed Inference)

## Project Structure

This repository contains three main components:
- **Fine-tuning**: Tools for fine-tuning LLMs on Neuron hardware using NxD
- **Inference**: Infrastructure for efficient inference using vLLM with NKI compilation and NxDI optimization
- **Agent Development**: Building intelligent agents with LangChain/LangGraph

## Setup Steps

1. Create a Trainium instance with AWS Neuron SDK v2.23 using EC2 with the following settings:
    1. **Name:** nki-llama
    2. **AMI:** Deep Learning AMI Neuron (Ubuntu 22.04)
    3. **Instance type:** trn1.32xlarge
    4. **Key pair (login):** create a new key pair
    5. When connecting to these instances via SSH, use the username of *ubuntu*.

2. Clone this repository and navigate to it:

```bash
git clone [REPO_URL]
cd [PATH]/nki-llama
```

3. Create your `.env` file by copying the provided example:

```bash
cp .env.example .env
# Edit .env file with your preferred settings
nano .env
```

## Environment Setup

This project requires three different Python environments:

1. **Fine-tuning Environment**:

```bash
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
```

2. **Inference Environment**:

```bash
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
```

3. **Jupyter Environment** (for agent development):

```bash
python3 -m venv venv
source venv/bin/activate
make inference-jupyter  # Sets up Jupyter and installs required packages
```

## Fine-tuning Workflow

Our Makefile simplifies the fine-tuning process:

```bash
# Activate the fine-tuning environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Install dependencies
make finetune-deps

# Download dataset
make finetune-data

# Download model
make finetune-model

# Convert checkpoint to NxDT format
make finetune-convert

# Pre-compile graphs (AOT)
make finetune-precompile

# Run fine-tuning job
make finetune-train
```

## Inference Workflow

The inference pipeline includes NKI (Neuron Kernel Interface) compilation and NxDI (Neuron Distributed Inference) integration with vLLM for optimal performance on Neuron hardware.

Use our Makefile to simplify the setup and execution process for inference:

```bash
# Activate the inference environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Setup vLLM for Neuron
make inference-setup

# Download model from Hugging Face (you'll need a HF token)
# (skip this step if using your fine-tuned model)
make inference-download

# The model will be automatically compiled with NKI and optimized for NxDI
# when the server starts for the first time

# Start the vLLM OpenAI-compatible API server with NxDI
make inference-server
```

### Environment Configuration

The repository includes a `.env.example` file with template configuration. Copy this file to create your own `.env`:

```bash
# .env file
# Model configuration
## HuggingFace Model ID (https://huggingface.co/meta-llama/Meta-Llama-3-8B)
MODEL_ID=meta-llama/Meta-Llama-3-8B
## Short name for model ID
MODEL_NAME=meta-llama-3-8b

# Server configurations
PORT=8080
MAX_MODEL_LEN=2048
TENSOR_PARALLEL_SIZE=32

HF_TOKEN=your_token_here
```

The Makefile will automatically load this configuration if present, or prompt you for values if not set.

### Running Inference

The Makefile provides several commands for running inference and evaluation:

```bash
# Activate the inference environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Download model from Hugging Face (you'll need a HF token)
# (skip this step if using your fine-tuned model)
make inference-download

# Run inference in generate mode
make inference-infer

# Run in evaluate-all mode
make inference-evaluate-all
```

## Agent Development

This repository includes support for building LLM-powered agents using LangGraph and LangChain. A sample travel planning agent is included that demonstrates how to build a stateful agent workflow with the following capabilities:

- Context-aware travel itinerary generation
- Multi-turn conversation with memory
- Dynamic workflow management using LangGraph
- Integration with VLLMOpenAI for efficient inference on Trainium

### Jupyter Notebook

The repository includes a Jupyter notebook for developing and testing agents. To use it:

1. Ensure you've started the vLLM server in one terminal: `make inference-server`
2. Start Jupyter Lab in another terminal:

```bash
# Activate the Jupyter environment
source venv/bin/activate

# Start Jupyter Lab
make inference-lab
```

3. Open the travel planning notebook and select the "neuron_agents" kernel

## Makefile Commands

| Command | Description |
|---------|-------------|
| **General** |
| `make help` | Show help message for all commands |
| `make clean` | Clean all generated files |
| **Fine-tuning** |
| `make finetune` | Run all fine-tuning steps |
| `make finetune-deps` | Install fine-tuning dependencies |
| `make finetune-data` | Download datasets for fine-tuning |
| `make finetune-model` | Download model for fine-tuning |
| `make finetune-convert` | Convert checkpoint to NxDT format |
| `make finetune-precompile` | Pre-compile graphs (AOT) |
| `make finetune-train` | Run fine-tuning job |
| `make finetune-clean` | Clean up fine-tuning files |
| **Inference** |
| `make inference` | Run inference (shortcut to inference-infer) |
| `make inference-setup` | Setup vLLM for Neuron |
| `make inference-jupyter` | Setup Jupyter environment |
| `make inference-download` | Download model from Hugging Face |
| `make inference-infer` | Run inference in generate mode (wip) |
| `make inference-evaluate` | Run inference in evaluate mode |
| `make inference-server` | Start vLLM OpenAI-compatible API server |
| `make inference-lab` | Run Jupyter Lab server |
| `make inference-clean` | Clean up inference files |

## Environment Requirements

- For fine-tuning: `source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate`
- For inference: `source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate`
- For agent development (Jupyter): `source venv/bin/activate`

## Full Workflow Example

Here's a complete workflow example combining all components:

1. **Fine-tune a model**:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
   make finetune
   ```

2. **Serve the model** for inference:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
   make inference-setup
   # You can either use your fine-tuned model or download one
   # make inference-download
   
   # The model will be compiled with NKI and optimized for NxDI
   # when you first start the server (this may take a few minutes)
   make inference-server
   ```

3. **Build agents** with the served model:

```bash
# In a new terminal
source venv/bin/activate
make inference-jupyter
make inference-lab
# Open the Jupyter notebook and connect to your model
```

---

© 2025 Amazon Web Services. All rights reserved.
