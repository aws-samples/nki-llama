# NKI-LLAMA: AWS Neuron Development Platform

A unified platform for fine-tuning, benchmarking, and serving LLaMA models on AWS Trainium and Inferentia using Neuron SDK's advanced optimization capabilities.

## 🎯 Overview

NKI-LLAMA provides a streamlined interface for the complete LLM development lifecycle on AWS Neuron hardware:

- **Fine-tune** models using NeuronX Distributed (NxD)
- **Optimize** with Neuron Kernel Interface (NKI) compilation
- **Benchmark** performance with comprehensive evaluation tools
- **Serve** models with vLLM's OpenAI-compatible API
- **Build** LLM-powered applications and agents

## 🔄 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│                 │     │                  │     │                 │     │              │
│   Fine-tuning   │────▶│ NKI Compilation  │────▶│ vLLM Inference  │────▶│  Reasoning   │-> cumulative score ()
│      (NxD)      │     │  & Benchmarking  │     │     (NxDI)      │     │  Benchmark   │
│                 │     │                  │     │                 │     │              │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
        │                         │                         │                    │
        ▼                         ▼                         ▼                    ▼
  Trained Model            NKI-Optimized              API Endpoint          test PT   
                          Model Artifacts            (OpenAI Compatible)   model reasoning
```

### Key Technologies

- **NKI (Neuron Kernel Interface)**: Custom kernel optimizations for AWS Neuron
- **NxD (NeuronX Distributed)**: Distributed training framework
- **NxDI (NeuronX Distributed Inference)**: Optimized inference runtime
- **vLLM**: High-performance serving with Neuron backend

## 📋 Requirements

### System Requirements
- **Instance**: trn1.32xlarge (recommended)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04)
- **Neuron SDK**: 2.23.0
- **Python**: 3.10

### SDK Components
- NeuronX Distributed Training: 1.3.0
- NeuronX Distributed Inference: 0.3.5591
- Neuron Compiler: 2.18.121.0

## 🚀 Quick Start

### 1. Instance Setup
```bash
# Create EC2 instance
# - Type: trn1.32xlarge
# - AMI: Deep Learning AMI Neuron (Ubuntu 22.04)
# - Storage: 512GB+ recommended
```

### 2. Installation
```bash
# Clone repository
git clone https://github.com/aws-neuron/nki-llama.git
cd nki-llama

# Install
chmod +x install.sh
./install.sh

# Configure
cp .env.example .env
nano .env  # Add your HF_TOKEN
# inference env vars, ensure max_model_len= seq_len
```

### 3. First Run
```bash
# Interactive setup
./nki-llama setup

# Download model
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama inference download

# Run benchmark (compiles model on first run)
tmux new -s benchmark
./nki-llama inference benchmark
```

## 📊 Score Calculation Workflow

The NKI-LLAMA platform includes a comprehensive score calculation system that evaluates training performance, inference performance, and reasoning capabilities. For detailed information about the scoring system, see the [Score Calculation README](src/README.md).

### Workflow Overview

1. **Pre-compile Phase**: 
   - Execute the pre-compile job using `./nki-llama finetune compile`
   - This generates a compile directory in the neuron cache
   - The pre-compile job creates a log file in `logs/nki-llama_*.log`
   - **Important**: Note the compile directory path from the "Pre-compile graphs" log output
   - Example: `/home/ubuntu/neuron_cache/neuronxcc-2.18.121.0+9e31e41a/MODULE_15329989265349737271+a65e371e`

2. **Training Execution**:
   - Execute the pre-compile job using `./nki-llama finetune train`
   - The training job creates a log file in `logs/nki-llama_*.log`
   - This log contains metrics like latency, throughput, and MFU
   - The benchmark inference file is always generated at: `benchmark_inference.json`

3. **Reasoning Evaluation** (Optional):
   - Run reasoning benchmarks using the dedicated script
   - Results are automatically discovered and integrated by the handler
   - Supports multiple reasoning datasets (GSM8K, MMLU, etc.)

4. **Score Collection**:
   - Once components complete, scores can be calculated using the handler
   - Training-only: NKI kernel training score
   - Training + Inference: Full NKI-LLAMA score
   - Training + Inference + Reasoning: Complete benchmark score with reasoning metrics

### Example Test Run

```bash
# Step 1: Run full fine-tuning job and note the compile directory
tmux new -s training
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
./nki-llama finetune all
# Look for "Pre-compile graphs" in output to find compile directory path

# Step 2: Run inference benchmark (optional for full score)
tmux new -s benchmark
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama inference benchmark 
# or directly run from main.py in src/inference/ for full use of flags for bucketing/context encoding and others

# Step 3: Run reasoning evaluation (optional for complete benchmark)
tmux new -s reasoning
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama/src/inference/scripts/reasoning-bench-lm-eval.sh
# Results are automatically saved to aws-neuron-samples/inference-benchmarking/results/

# Step 4: Calculate scores
# For training-only score:
python /home/ubuntu/nki-llama/src/handler.py \
    --config /home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml \
    --model-config /home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json \
    --log-file /home/ubuntu/nki-llama/logs/nki-llama_20250610_014432.log \
    --compile-dir /home/ubuntu/neuron_cache/neuronxcc-2.18.121.0+9e31e41a/MODULE_15329989265349737271+a65e371e \
    --throughput 2.1 \
    --output benchmark_results.json \
    --training-weight 0.5 \
    --inference-weight 0.5 \
    --hw-backend trn1 \
    --per-file-scores \
    --calculate-score \
    --detailed \
    --verbose

# For full score (with inference):
python /home/ubuntu/nki-llama/src/handler.py \
    --config /home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml \
    --model-config /home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json \
    --log-file /home/ubuntu/nki-llama/logs/nki-llama_20250610_014432.log \
    --compile-dir /home/ubuntu/neuron_cache/neuronxcc-2.18.121.0+9e31e41a/MODULE_15329989265349737271+a65e371e \
    --inference-results /home/ubuntu/nki-llama/src/inference/benchmark_inference.json \
    --throughput 2.1 \
    --output benchmark_results.json \
    --training-weight 0.5 \
    --inference-weight 0.5 \
    --hw-backend trn1 \
    --per-file-scores \
    --calculate-score \
    --detailed \
    --verbose

# For complete score (with reasoning - handler automatically discovers results):
python /home/ubuntu/nki-llama/src/handler.py \
    --config /home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml \
    --model-config /home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json \
    --log-file /home/ubuntu/nki-llama/logs/nki-llama_20250610_014432.log \
    --compile-dir /home/ubuntu/neuron_cache/neuronxcc-2.18.121.0+9e31e41a/MODULE_15329989265349737271+a65e371e \
    --inference-results /home/ubuntu/nki-llama/src/inference/benchmark_inference.json \
    --reasoning-results \
    --throughput 2.1 \
    --output benchmark_results.json \
    --training-weight 0.33 \
    --inference-weight 0.33 \
    --reasoning-weight 0.34 \
    --hw-backend trn1 \
    --per-file-scores \
    --calculate-score \
    --detailed \
    --verbose
```

The score calculation provides insights into:
- **Training Performance**: MFU improvement and throughput gains
- **Inference Performance**: Latency reduction and throughput increase
- **NKI Optimization**: Ratio of NKI-optimized operations

## 💻 Command Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `./nki-llama setup` | Interactive setup wizard |
| `./nki-llama status` | System and project status |
| `./nki-llama config` | Display configuration |
| `./nki-llama clean` | Clean artifacts and cache |

### Fine-tuning Pipeline

```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Complete pipeline
./nki-llama finetune all

# Or run individual steps
./nki-llama finetune deps      # Install dependencies
./nki-llama finetune data      # Download dataset
./nki-llama finetune model     # Download base model
./nki-llama finetune convert   # Convert to NxDT format
./nki-llama finetune compile   # Pre-compile graphs
./nki-llama finetune train     # Start training
```

### Benchmarking & Compilation

```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Download model (if not already done)
./nki-llama inference download

# Full benchmark with NKI compilation (default)
./nki-llama inference benchmark

# Benchmark with options
./nki-llama inference benchmark --seq-len 1024
./nki-llama inference benchmark --clear-cache  # Clear compilation cache
./nki-llama inference benchmark --no-nki       # Without NKI optimizations
```

#### Benchmark Modes

| Mode | Description | Status |
|------|-------------|--------|
| `evaluate_all` | Full benchmark with NKI compilation and caching | ✅ Working |
| `evaluate_single` | Quick validation test | ⚠️ Not implemented |

> **Note**: The `evaluate_single` mode is currently not functional. Use `evaluate_all` (default) for all benchmarking needs.

### Reasoning Evaluation

```bash
# Activate inference environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Run reasoning benchmarks (requires vLLM server)
./nki-llama/src/inference/scripts/reasoning-bench-lm-eval.sh

# The script will:
# 1. Start vLLM server with your compiled model
# 2. Run lm-eval on reasoning datasets (GSM8K, MMLU, etc.)
# 3. Save results to aws-neuron-samples/inference-benchmarking/results/
# 4. Results are automatically discovered by handler.py
```

#### Reasoning Datasets

The reasoning evaluation includes multiple datasets:
- **GSM8K CoT**: Grade school math with chain-of-thought reasoning
- **MMLU Pro**: Massive multitask language understanding (professional level)
- **MMLU Flan CoT**: MMLU with chain-of-thought prompting

#### Result Integration

The handler automatically discovers reasoning results by:
1. **Model Mapping**: Maps your model configuration to result directory paths
2. **Score Extraction**: Parses "exact_match,strict-match" scores from JSON files
3. **Path Discovery**: Searches `aws-neuron-samples/inference-benchmarking/results/accuracy/mytest/`
4. **Graceful Fallback**: Works without reasoning results (training+inference only)

### Inference Serving

```bash
# Setup vLLM (one-time)
./nki-llama inference setup

# Start API server
tmux new -s vllm
./nki-llama inference server

# Server will use NKI-compiled artifacts from benchmarking
```

### Reasoning Evaluation

```bash
# Activate inference environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Run complete reasoning evaluation
./nki-llama/src/inference/scripts/reasoning-bench-lm-eval.sh

# Results are automatically saved and discovered by handler.py
# No additional flags needed - handler finds results by model path
```

### Development Tools

```bash
# Start Jupyter Lab
./nki-llama jupyter

# Access at http://your-instance-ip:8888
```

## 🛠️ Advanced Usage

### Cache Management

The compilation cache can accumulate failed entries. Monitor and manage it:

```bash
# Check cache status
./nki-llama status

# Clear cache before benchmark
./nki-llama inference benchmark --clear-cache

# Manual cache cleanup
./nki-llama clean
```

### Using tmux (Recommended)

Long-running operations should use tmux to prevent disconnection issues:

```bash
# Create session
tmux new -s session-name

# Run command
./nki-llama [command]

# Detach: Ctrl+B, then D

# List sessions
tmux ls

# Reattach
tmux attach -t session-name
```

### Environment Management

Different operations require specific environments:

```bash
# Fine-tuning
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Inference & Benchmarking
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Agent Development
source ~/nki-llama/venv/bin/activate
```

## 📊 Monitoring & Debugging

### System Monitoring
```bash
# Neuron device status
neuron-ls

# Real-time usage
neuron-top

# Project status
./nki-llama status
```

### Log Files
```bash
# View recent logs
ls -la logs/
tail -f logs/nki-llama_*.log

# Benchmark results
cat logs/benchmarks/*/metadata.json
```

### Common Issues

#### Compilation Cache Errors
```bash
# Symptoms: "Got a cached failed neff" errors
# Solution:
./nki-llama inference benchmark --clear-cache
```

#### SIGHUP Errors
```bash
# Symptoms: Process terminated during compilation
# Solution: Always use tmux for long operations
tmux new -s benchmark
```

#### Memory Issues
```bash
# Monitor memory usage
neuron-top

# Adjust parallelism if needed
export TENSOR_PARALLEL_SIZE=4  # Reduce from 8
```

#### Reasoning Evaluation Issues
```bash
# Symptoms: "No reasoning results found" in handler output
# Check if results directory exists:
ls -la ~/aws-neuron-samples/inference-benchmarking/results/accuracy/mytest/

# Symptoms: vLLM server connection errors during reasoning eval
# Ensure server is running and accessible:
curl http://localhost:8080/v1/models

# Symptoms: Reasoning script fails with transformers version error
# The script automatically installs transformers<4.50 for compatibility
# If issues persist, manually install:
pip install "transformers<4.48.2"

# Symptoms: Handler can't find reasoning results for your model
# Check model path mapping in results directory:
# Results are stored by model path: /home/ubuntu/models/llama-3-1-8b
# becomes: __home__ubuntu__models__llama-3-1-8b
ls -la ~/aws-neuron-samples/inference-benchmarking/results/accuracy/mytest/*/
```

## 🏗️ Project Structure

```
nki-llama/
├── nki-llama.sh          # Main CLI interface
├── nki-llama.config      # System configuration
├── .env                  # User configuration
├── install.sh            # Installation script
├── README.md             # This file
├── src/
│   ├── README.md         # Score calculation documentation
│   ├── handler.py        # Score calculation handler
│   ├── fine-tune/        # Training pipeline
│   │   └── scripts/      # Training automation
│   └── inference/        # Inference pipeline
│       ├── main.py       # Benchmark entry point
│       └── scripts/      # Inference automation
├── notebooks/            # Example notebooks
│   └── neuron_agents.ipynb
├── logs/                 # Operation logs
│   └── benchmarks/       # Benchmark results
└── models/              # Downloaded models
    └── compiled/        # NKI-compiled artifacts
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Hugging Face Access
HF_TOKEN=your_token_here

# Model Selection
MODEL_ID=meta-llama/Meta-Llama-3-8B
MODEL_NAME=llama-3-8b

# Hardware Configuration
TENSOR_PARALLEL_SIZE=8
NEURON_RT_NUM_CORES=8

# Inference Parameters
INFERENCE_PORT=8080
MAX_MODEL_LEN=2048

# Reasoning Evaluation (Optional)
# These are automatically configured by reasoning-bench-lm-eval.sh
# REASONING_DATASETS="mmlu_pro, gsm8k_cot, mmlu_flan_cot_zeroshot"
# REASONING_LIMIT=200  # Number of samples per dataset
# REASONING_TIMEOUT=3600  # Timeout in seconds
```

## 🎓 Complete Workflow Example

### Step 1: Fine-tune a Model
```bash
tmux new -s training
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
./nki-llama finetune all
# Note the compile directory from "Pre-compile graphs" output
# Detach: Ctrl+B, D
```

### Step 2: Benchmark & Compile
```bash
tmux new -s benchmark
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama inference download
./nki-llama inference benchmark
# First run compiles with NKI (10-30 minutes)
# Detach: Ctrl+B, D
```

### Step 3: Reasoning Evaluation (Optional)
```bash
tmux new -s reasoning
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama/src/inference/scripts/reasoning-bench-lm-eval.sh
# Runs GSM8K, MMLU Pro, and MMLU Flan CoT evaluations
# Results saved to aws-neuron-samples/inference-benchmarking/results/
# Detach: Ctrl+B, D
```

### Step 4: Calculate Performance Score
```bash
# After training and/or inference completes
# For complete score with reasoning (if Step 3 was run):
python /home/ubuntu/nki-llama/src/handler.py \
    --compile-dir /path/from/training/logs \
    --log-file logs/nki-llama_latest.log \
    --inference-results benchmark_inference.json \
    --reasoning-results \
    --calculate-score

# For training+inference only (without reasoning):
python /home/ubuntu/nki-llama/src/handler.py \
    --compile-dir /path/from/training/logs \
    --log-file logs/nki-llama_latest.log \
    --inference-results benchmark_inference.json \
    --calculate-score
```

### Step 5: Serve Model
```bash
tmux new -s vllm-server
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama inference server
# API available at http://localhost:8080
# Detach: Ctrl+B, D
```

### Step 6: Build Applications
```bash
# Terminal 1: Keep server running
# Terminal 2: Development
./nki-llama jupyter
# Open browser to http://your-ip:8888
```

## 📚 Additional Resources

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NeuronX Distributed Training Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html)
- [NKI Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- [vLLM Neuron Integration](https://docs.vllm.ai/en/latest/getting_started/neuron-installation.html)

## 🐛 Known Issues

- **First compilation**: Initial NKI compilation can take 10-30 minutes. Subsequent runs use cache.
- **Cache corruption**: If benchmark fails with cache errors, use `--clear-cache` flag.

## 📄 License

© 2025 Amazon Web Services. All rights reserved.

This project is provided under the AWS Customer Agreement and integrates with AWS Neuron SDK components subject to their respective licenses.