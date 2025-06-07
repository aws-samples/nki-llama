# NKI-LLAMA: Unified Interface for AWS Neuron

A unified project for fine-tuning, inference, and agent development of LLaMA models on AWS Trainium and Inferentia using a streamlined bash-based interface.

## 📋 Requirements

### Neuron SDK Version
- **Neuron 2.23.0 Release**
- **NeuronX Distributed Inference**: 0.3.5591
- **NeuronX Distributed Training**: 1.3.0

### Hardware & AMI
- **Required Instance**: trn1.32xlarge
- **Base AMI**: Deep Learning AMI Neuron (Ubuntu 22.04) with Neuron SDK 2.23
- **Base Packages**:
  - NxD (NeuronX Distributed Training)
  - NKI (Neuron Kernel Interface)
  - NxDI (NeuronX Distributed Inference)

## 🔄 Project Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│                 │     │                  │     │                 │     │              │
│   Fine-tuning   │────▶│ NKI Compilation  │────▶│ vLLM Inference  │────▶│    Agent     │
│      (NxD)      │     │  & Benchmarking  │     │     (NxDI)      │     │ Development  │
│                 │     │                  │     │                 │     │              │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┐
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
  Trained Model            NKI-Optimized              API Endpoint
                          Model Artifacts            (OpenAI Compatible)
```

### Detailed Workflow:

1. **Fine-tune** a model using NeuronX Distributed (NxD) on Trainium
2. **NKI Compilation & Benchmarking**:
   - Compiles model graphs with Neuron Kernel Interface (NKI)
   - Creates optimized artifacts for inference
   - Benchmarks performance characteristics
   - Supports two modes: `evaluate_single` and `evaluate_all`
3. **vLLM Inference** serves the NKI-compiled model using NeuronX Distributed Inference (NxDI)
4. **Agent Development** connects to the inference endpoint for application building

### Key Components:
- **NKI (Neuron Kernel Interface)**: Optimizes model operations for AWS Neuron hardware
- **NxD (NeuronX Distributed)**: Enables distributed training across Neuron cores
- **NxDI (NeuronX Distributed Inference)**: Provides optimized inference runtime
- **vLLM**: Serves models with OpenAI-compatible API using Neuron optimizations

## 🚀 Quick Start

```bash
# Install
chmod +x install.sh
./install.sh

# Setup Guide
./nki-llama setup

# Run full benchmark (with NKI compilation)
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama inference benchmark

# Run quick single evaluation
# ./nki-llama inference benchmark single (TODO: FIX)

# Start inference server
# ./nki-llama server (TODO: DOCS)
```

## 🏗️ Initial Setup

### 1. Create Trainium Instance

Create a trn1.32xlarge instance on AWS EC2:
- **Name**: nki-llama
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04)
- **Instance type**: trn1.32xlarge
- **Key pair**: Create new key pair
- **Username**: ubuntu (when connecting via SSH)

### 2. Clone and Install

```bash
# Clone repository
git clone [REPO_URL]
cd nki-llama

# Run installation
chmod +x install.sh
./install.sh

# Configure environment
cp .env.example .env
nano .env  # Add your HF_TOKEN and adjust settings
```

## 📁 Project Structure

```
/home/ubuntu/nki-llama/
├── nki-llama.sh          # Main CLI interface
├── nki-llama.config      # Shared configuration
├── .env                  # Your environment variables
├── .env.example          # Example configuration
├── install.sh            # Installation script
├── src/
│   ├── fine-tune/
│   │   └── scripts/      # Fine-tuning scripts
│   │       ├── bootstrap.sh
│   │       ├── download_data.sh
│   │       ├── download_model.sh
│   │       ├── convert_checkpoints.sh
│   │       ├── precompile.sh
│   │       └── run_training.sh
│   └── inference/
│       ├── main.py       # Inference entry point
│       └── scripts/      # Inference helper scripts
│           ├── setup-vllm.sh
│           ├── download-model.sh
│           ├── run-nki-benchmark.sh  # Supports both evaluate_single and evaluate_all modes
│           ├── start-server.sh
│           └── jupyter.sh
└── logs/                 # Unified logs
    └── benchmarks/       # Benchmark results
```

## 🔧 Environment Setup

This project requires three different Python environments:

### 1. Fine-tuning Environment
```bash
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
```

### 2. Inference Environment
```bash
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
```

### 3. Jupyter Environment (for agent development)
```bash
./nki-llama jupyter setup
source ~/nki-llama/venv/bin/activate
```

## 💻 Commands

### Quick Commands
- `./nki-llama setup` - Interactive setup wizard
- `./nki-llama train` - Start fine-tuning (shortcut)
- `./nki-llama server` - Start inference server (shortcut)
- `./nki-llama jupyter` - Launch Jupyter Lab
- `./nki-llama status` - Check system status
- `./nki-llama config` - Show configuration

### Fine-tuning Workflow

```bash
# Activate fine-tuning environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Run individual steps
./nki-llama finetune deps      # Install dependencies
./nki-llama finetune data      # Download dataset
./nki-llama finetune model     # Download model
./nki-llama finetune convert   # Convert checkpoints to NxDT format
./nki-llama finetune compile   # Pre-compile graphs (AOT)
./nki-llama finetune train     # Start fine-tuning

# Or run all at once
./nki-llama finetune all
```

### NKI Benchmark Modes

The benchmark script supports two evaluation modes:

#### 1. evaluate_single Mode
Quick single evaluation using the repository test script:
```bash
# Activate inference environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Run single evaluation
./nki-llama inference benchmark single

# Or explicitly specify mode
./nki-llama inference benchmark --mode evaluate_single
```

#### 2. evaluate_all Mode (Default)
Comprehensive benchmark with NKI compilation and all configurations:
```bash
# Download model if not already available
./nki-llama inference download

# Run full benchmark (compiles model on first run)
./nki-llama inference benchmark

# Or with custom parameters
./nki-llama inference benchmark --seq-len 1024 --tp-degree 8

# Run without NKI optimizations
./nki-llama inference benchmark --no-nki
```

**Key differences:**
- **evaluate_single**: Quick validation, runs from repository test script
- **evaluate_all**: Full benchmark with model compilation, creates cached artifacts for vLLM

**Note**: The `evaluate_all` mode automatically compiles the model with NKI optimizations on the first run (10-30 minutes). Subsequent runs use the cached compilation.

### Inference Serving

```bash
# After benchmarking (which compiles the model), start the API server

# Setup vLLM
./nki-llama inference setup      # Setup vLLM for Neuron

# Start the API server
./nki-llama inference server     # Start OpenAI-compatible API
```

## 🤖 Agent Development

This repository includes support for building LLM-powered agents using LangGraph and LangChain. A sample travel planning agent demonstrates:

- Context-aware travel itinerary generation
- Multi-turn conversation with memory
- Dynamic workflow management using LangGraph
- Integration with vLLM for efficient inference on Trainium

### Using Jupyter for Agent Development

```bash
# Terminal 1: Start the inference server
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama server

# Terminal 2: Start Jupyter Lab
./nki-llama jupyter
# Access at http://your-ip:8888
# Select the "nki-llama" kernel in Jupyter
```

## ⚙️ Configuration

All configuration is managed through:
1. `nki-llama.config` - System paths and defaults (includes TEST_SCRIPT, MAIN_SCRIPT, etc.)
2. `.env` - Your personal configuration

### Key Variables

```bash
# Model Configuration
MODEL_ID=meta-llama/Meta-Llama-3-8B
MODEL_NAME=llama-3-8b
HF_TOKEN=your_huggingface_token

# Training Parameters
BATCH_SIZE=1
MAX_STEPS=1000
SEQ_LENGTH=2048
TENSOR_PARALLEL_SIZE=8
LEARNING_RATE=5e-5

# Inference Parameters
INFERENCE_PORT=8080
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=4

# Dataset Configuration
DATASET_NAME=databricks/databricks-dolly-15k
```

## 📊 Monitoring

### Check Status
```bash
./nki-llama status
```

### View Logs
```bash
# Logs are stored with timestamps
ls logs/
tail -f logs/nki-llama_*.log

# Benchmark results
ls logs/benchmarks/
cat logs/benchmarks/*/metadata.json
```

### Neuron Monitoring
```bash
neuron-ls    # List Neuron devices
neuron-top   # Monitor Neuron usage
```

## 🔍 Complete Workflow Example

Here's a complete end-to-end workflow with tmux best practices:

### 1. Fine-tune a Model
```bash
# Create tmux session for training
tmux new -s training

# Inside tmux: activate environment and run training
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
./nki-llama finetune all

# Detach from tmux: Ctrl+B, D
# Check progress later: tmux attach -t training
```

### 2. Benchmark Model with NKI
```bash
# Create tmux session for benchmarking
tmux new -s benchmark

# Inside tmux: run benchmarks
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Quick single evaluation with compilation  (first run compiles)
# ./nki-llama inference benchmark single (TODO: FIX COMMAND)

# Or full benchmark with compilation (first run compiles)
./nki-llama inference benchmark

# Detach and let it run: Ctrl+B, D
```

### 3. View Benchmark Results
```bash
# After benchmarking completes
ls logs/benchmarks/
cat logs/benchmarks/*/metadata.json

# View detailed logs
cat logs/benchmarks/*/benchmark.log
```

### 4. Serve with vLLM
```bash
# Create tmux session for server
tmux new -s vllm

# Inside tmux: start the server
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama server

# Server automatically uses NKI-compiled artifacts
# Detach: Ctrl+B, D
```

### 5. Build Agents
```bash
# In a new terminal
./nki-llama jupyter

# Your model is now available at http://localhost:8080
# Build agents using the OpenAI-compatible API
```

### Managing tmux Sessions
```bash
# List all sessions
tmux ls

# Attach to a session
tmux attach -t training
tmux attach -t benchmark
tmux attach -t vllm

# Kill a session
tmux kill-session -t training
```

## 🚨 Troubleshooting

### Environment Issues
```bash
# Check active environment
./nki-llama status

# Wrong environment error?
# For fine-tuning:
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# For inference:
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
```

### Benchmark Modes
- **evaluate_single**: Use for quick validation tests
- **evaluate_all**: Use for full performance evaluation and model compilation
- First-time compilation with NKI (in evaluate_all mode) can take 10-30 minutes
- Compiled models are cached in `~/traced_model/`
- Subsequent benchmark runs will use the cached compilation

### Memory Issues
- Ensure you're using trn1.32xlarge for full model support
- Monitor memory usage with `neuron-top`
- Adjust `TENSOR_PARALLEL_SIZE` if needed

### Using with tmux
For long-running operations like training, benchmarking, or serving:

```bash
# Create a new tmux session
tmux new -s session-name

# Run your command
./nki-llama [command]

# Detach from session
Ctrl+B, then D

# List sessions
tmux ls

# Reattach to session
tmux attach -t session-name
```

## 🤝 Contributing

The modular design makes it easy to add new features:

1. Add new scripts to `scripts/` directory
2. Update command handlers in `nki-llama.sh`
3. Add configuration to `nki-llama.config`

## 📄 License

© 2025 Amazon Web Services. All rights reserved.

This project integrates with AWS Neuron SDK and follows its licensing terms.