# NKI-LLAMA Benchmark Handler

A benchmarking system for evaluating NKI-LLAMA model performance across both training and inference metrics.

## 🚀 Overview

The NKI-LLAMA Benchmark Handler calculates a unified performance score that combines:
- **Training metrics**: MFU (Model FLOPs Utilization), throughput, and NKI kernel usage
- **Inference metrics**: Latency, throughput, and accuracy (optional)
- **Reasoning metrics**: Accuracy scores from reasoning benchmarks (GSM8K, MMLU, etc.)
- **NKI optimization**: Ratio of NKI (Neuron Kernel Interface) operations to total operations

The system supports multiple modes:
- **Training-only mode**: When inference results are not available, provides NKI kernel training score
- **Combined mode**: When both training and inference results are available, provides full NKI-LLAMA score
- **Full integration mode**: When training, inference, and reasoning results are available, provides comprehensive NKI-LLAMA score with reasoning component

The final score follows the formula:
```
# Combined mode (training + inference):
Score = Accuracy × Reduced Latency × Increased Throughput × (1 + Normalized NKI FLOPS)

# Full integration mode (training + inference + reasoning):
Score = (Base Score) × (1 + Reasoning Score Weight × Reasoning Accuracy)
```

## 💻 Usage

### Basic Usage

Run with default parameters:
```bash
python handler.py
```

This will:
1. Calculate training metrics using `calculate_training_metrics.py`
2. Load inference results from `benchmark_inference.json` (if available)
3. Calculate the NKI-LLAMA score (combined or training-only)
4. Save results to `benchmark_results.json`

### Training-Only Mode

If the inference benchmark file doesn't exist, the handler automatically runs in training-only mode:
```bash
python handler.py --calculate-score
```

This provides immediate feedback on NKI kernel optimization progress without requiring inference implementation.

### Reasoning Results Integration

The handler can automatically discover and integrate reasoning benchmark results from the aws-neuron-samples inference-benchmarking framework.

#### Automatic Reasoning Result Discovery

When reasoning results are available, the handler automatically discovers them based on your model configuration:

```bash
# Run with reasoning integration (automatic discovery)
python handler.py --reasoning-results
```

The handler searches for reasoning results in:
```
aws-neuron-samples/inference-benchmarking/results/accuracy/mytest/
├── gsm8k_cot/
├── mmlu_pro/
└── mmlu_flan_cot_zeroshot/
```

#### Custom Reasoning Results Path

You can specify a custom path to reasoning results:

```bash
# Use custom reasoning results directory
python handler.py --reasoning-results /path/to/custom/reasoning/results
```

#### Running Complete Workflow

1. **First, run reasoning benchmarks** using the provided script:
   ```bash
   # Activate inference environment
   source /path/to/inference/venv/bin/activate
   
   # Run reasoning benchmarks (this takes time)
   cd nki-llama/src/inference/scripts
   ./reasoning-bench-lm-eval.sh
   ```

2. **Then run the handler** to get comprehensive scores:
   ```bash
   # Get full NKI-LLAMA score with reasoning integration
   python handler.py --reasoning-results --calculate-score
   ```

### Advanced Usage

#### Custom Training Configuration
```bash
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
```

#### Custom Inference Results
```bash
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
```

#### Adjust Score Weights
```bash
# Adjust training and inference weights
python handler.py \
    --training-weight 0.3 \
    --inference-weight 0.7

# Include reasoning with custom weights
python handler.py \
    --training-weight 0.3 \
    --inference-weight 0.5 \
    --reasoning-weight 0.2 \
    --reasoning-results
```

#### Reasoning Integration Examples
```bash
# Full workflow with reasoning integration
python handler.py \
    --reasoning-results \
    --calculate-score \
    --verbose

# Use custom reasoning results directory
python handler.py \
    --reasoning-results-path /custom/reasoning/results \
    --reasoning-weight 0.25

# Training + reasoning only (skip inference)
python handler.py \
    --reasoning-results \
    --training-weight 0.8 \
    --reasoning-weight 0.2
```

#### Verbose Output
```bash
python handler.py --verbose
```

### Command Line Options

#### Training Metrics Options
| Option | Default | Description |
|--------|---------|-------------|
| `--training-script` | `/home/ubuntu/nki-llama/src/fine-tune/scripts/calculate_training_metrics.py` | Path to training metrics script |
| `--config` | `/home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml` | Training config YAML |
| `--model-config` | `/home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json` | Model config JSON |
| `--log-file` | `/home/ubuntu/nki-llama/logs/nki-llama_20250610_014432.log` | Training log file |
| `--compile-dir` | `/home/ubuntu/neuron_cache` | Neuron compile cache directory |
| `--throughput` | `2.1` | Training throughput (seq/s) |
| `--hw-backend` | `trn1` | Hardware backend (trn1/trn2) |

#### Inference Metrics Options
| Option | Default | Description |
|--------|---------|-------------|
| `--inference-results` | `benchmark_inference.json` | Inference benchmark results file (optional - if not provided, only training score is calculated) |
| `--reference-latency` | `50000` | Reference implementation latency (ms) |
| `--reference-throughput` | `10` | Reference implementation throughput (tokens/s) |

#### Reasoning Metrics Options
| Option | Default | Description |
|--------|---------|-------------|
| `--reasoning-results` | `None` | Enable reasoning results integration (auto-discovery) |
| `--reasoning-results-path` | `~/aws-neuron-samples/inference-benchmarking/results` | Custom path to reasoning results directory |
| `--reasoning-weight` | `0.2` | Weight for reasoning score component (0-1) |

#### Score Calculation Options
| Option | Default | Description |
|--------|---------|-------------|
| `--training-weight` | `0.4` | Weight for training score (0-1) |
| `--inference-weight` | `0.6` | Weight for inference score (0-1) |

#### Output Options
| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `benchmark_results.json` | Output file for combined results |
| `--training-output` | `benchmark_finetuning.json` | Output file for training metrics |
| `--verbose` | `False` | Enable verbose output |

## 📊 Output Format

### Console Output - Combined Mode
```
======================================================================
NKI-LLAMA BENCHMARK RESULTS
======================================================================

🏆 FINAL NKI-LLAMA SCORE: 0.0046

Score Weights:
  Training: 40%
  Inference: 60%

📊 Component Scores:
  Training Score: 0.0077
  Inference Score: 0.0026
  NKI Ratio: 0.1846

🎯 Training Metrics:
  MFU: 15.48% (baseline: 50.00%)
  Throughput: 2.10 seq/s (baseline: 100.00)
  MFU Improvement: 0.3095x
  Throughput Improvement: 0.0210x

⚡ Inference Metrics:
  Latency: 12131.49ms (reference: 50000.00ms)
  Throughput: 52.76 tokens/s (reference: 10.00)
  Latency Reduction: 4.1220x
  Throughput Increase: 5.2755x
  Accuracy: ✓ Passed

======================================================================
```

### Console Output - Full Integration Mode (with Reasoning)
```
======================================================================
NKI-LLAMA BENCHMARK RESULTS
======================================================================

🏆 FINAL NKI-LLAMA SCORE: 0.0055

Score Weights:
  Training: 30%
  Inference: 50%
  Reasoning: 20%

📊 Component Scores:
  Training Score: 0.0077
  Inference Score: 0.0026
  Reasoning Score: 0.555
  NKI Ratio: 0.1846

🎯 Training Metrics:
  MFU: 15.48% (baseline: 50.00%)
  Throughput: 2.10 seq/s (baseline: 100.00)
  MFU Improvement: 0.3095x
  Throughput Improvement: 0.0210x

⚡ Inference Metrics:
  Latency: 12131.49ms (reference: 50000.00ms)
  Throughput: 52.76 tokens/s (reference: 10.00)
  Latency Reduction: 4.1220x
  Throughput Increase: 5.2755x
  Accuracy: ✓ Passed

🧠 Reasoning Metrics:
  GSM8K CoT: 55.5% (exact match, strict)
  MMLU Pro: Not available
  MMLU Flan CoT: Not available
  Overall Reasoning Score: 0.555

======================================================================
```

### Console Output - Training-Only Mode
```
======================================================================
NKI-LLAMA BENCHMARK RESULTS
======================================================================

⚠️  TRAINING-ONLY MODE (Inference results not available)

🏆 NKI KERNEL TRAINING SCORE: 0.0077
   NKI Ratio: 0.1846

🎯 Training Metrics:
  MFU: 15.48% (baseline: 50.00%)
  Throughput: 2.10 seq/s (baseline: 100.00)
  MFU Improvement: 0.3095x
  Throughput Improvement: 0.0210x

💡 Note: This score represents training performance only.
   To get the full NKI-LLAMA score, run inference benchmarks and provide
   the results file using --inference-results option.

======================================================================
```

### JSON Output (`benchmark_results.json`)
```json
{
  "timestamp": "2025-01-01T12:00:00",
  "mode": "full_integration",
  "nki_kernel_score": 0.0055,
  "component_scores": {
    "training": 0.0077,
    "inference": 0.0026,
    "reasoning": 0.555
  },
  "weights": {
    "training": 0.3,
    "inference": 0.5,
    "reasoning": 0.2
  },
  "nki_ratio": 0.1846,
  "detailed_breakdown": {
    "training": {
      "base_mfu": 50.0,
      "base_throughput": 100.0,
      "achieved_mfu": 15.48,
      "achieved_throughput": 2.1,
      "mfu_improvement": 0.3095,
      "throughput_improvement": 0.021,
      "nki_flop_ratio": 0.1846
    },
    "inference": {
      "accuracy": 1.0,
      "reduced_latency": 4.122,
      "increased_throughput": 5.2755,
      "normalized_nki_flops": 0.1846,
      "reference_latency_ms": 50000,
      "achieved_latency_ms": 12131.49,
      "reference_throughput": 10,
      "achieved_throughput": 52.76
    },
    "reasoning": {
      "gsm8k_cot": {
        "exact_match_strict": 0.555,
        "exact_match_flexible": 0.575,
        "n_samples": 200
      },
      "mmlu_pro": null,
      "mmlu_flan_cot_zeroshot": null,
      "overall_score": 0.555,
      "discovered_results": [
        "gsm8k_cot"
      ]
    }
  }
}
```

## 📈 Score Interpretation

### Training Score Components
- **MFU Improvement**: How much better the model utilizes FLOPs compared to baseline
- **Throughput Improvement**: Training speed improvement over baseline
- **NKI Ratio**: Percentage of operations using optimized NKI kernels

### Inference Score Components
- **Accuracy**: Binary flag (1 if meets threshold, 0 otherwise)
- **Reduced Latency**: How much faster the model responds (higher is better)
- **Increased Throughput**: How many more tokens/second (higher is better)
- **NKI FLOPS**: Bonus for using NKI optimized operations

### Reasoning Score Components
- **GSM8K CoT**: Grade school math problems with chain-of-thought reasoning
- **MMLU Pro**: Massive multitask language understanding (professional level)
- **MMLU Flan CoT**: MMLU with chain-of-thought prompting
- **Overall Score**: Weighted average of available reasoning benchmark scores

### Score Ranges
- **0-1**: Poor performance, needs optimization
- **1-10**: Baseline performance
- **10-50**: Good optimization
- **50+**: Excellent optimization

## 🔧 Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure all paths in command arguments are correct
   ```bash
   python handler.py --verbose  # Shows detailed error messages
   ```

2. **Missing `benchmark_inference.json`**: The handler will automatically run in training-only mode
   ```bash
   # To create a sample inference results file for testing:
   echo '{"e2e_model": {"latency_ms_avg": 12131.49, "throughput": 52.76}}' > benchmark_inference.json
   ```

3. **Training metrics calculation fails**: Check:
   - Training log file exists and has correct format
   - Neuron cache directory contains HLO files
   - Model config JSON is valid

### Reasoning Results Troubleshooting

#### Reasoning Results Not Found

If reasoning results are not discovered automatically:

1. **Check the results directory structure**:
   ```bash
   # Expected structure:
   ls -la ~/aws-neuron-samples/inference-benchmarking/results/accuracy/mytest/
   # Should show: gsm8k_cot/, mmlu_pro/, mmlu_flan_cot_zeroshot/
   ```

2. **Verify model name mapping**:
   ```bash
   # Handler maps model paths to sanitized names
   # /home/ubuntu/models/llama-3-1-8b → __home__ubuntu__models__llama-3-1-8b
   find ~/aws-neuron-samples/inference-benchmarking/results -name "*llama*" -type d
   ```

3. **Check reasoning benchmark results exist**:
   ```bash
   # Look for JSON result files
   find ~/aws-neuron-samples/inference-benchmarking/results -name "results_*.json" | head -5
   ```

#### Reasoning Results Format Issues

If reasoning results are found but parsing fails:

1. **Validate JSON format**:
   ```bash
   # Check if result files are valid JSON
   python -m json.tool /path/to/results_file.json > /dev/null
   ```

2. **Check required fields**:
   ```bash
   # Verify the file contains expected structure
   jq '.results | keys' /path/to/results_file.json
   # Should show reasoning benchmark names like "gsm8k_cot"
   ```

3. **Inspect score fields**:
   ```bash
   # Check for exact_match,strict-match scores
   jq '.results.gsm8k_cot."exact_match,strict-match"' /path/to/results_file.json
   ```

#### Running Reasoning Benchmarks

If you need to generate reasoning results:

1. **Set up the inference environment**:
   ```bash
   # Activate the inference virtual environment
   source /path/to/inference/venv/bin/activate
   
   # Verify vLLM and dependencies are installed
   python -c "import vllm; print('vLLM available')"
   ```

2. **Run the reasoning benchmark script**:
   ```bash
   cd nki-llama/src/inference/scripts
   ./reasoning-bench-lm-eval.sh
   ```

3. **Monitor benchmark progress**:
   ```bash
   # Check server logs
   tail -f ~/aws-neuron-samples/inference-benchmarking/server_*.log
   
   # Check for result files being created
   watch "find ~/aws-neuron-samples/inference-benchmarking/results -name 'results_*.json' | wc -l"
   ```

#### Custom Reasoning Results Path

If using custom reasoning results location:

1. **Specify custom path**:
   ```bash
   python handler.py --reasoning-results-path /custom/path/to/results
   ```

2. **Verify directory structure**:
   ```bash
   # Custom path should have same structure as aws-neuron-samples
   ls -la /custom/path/to/results/accuracy/mytest/
   ```

### Debug Mode
Run with verbose flag to see detailed execution:
```bash
python handler.py --verbose 2>&1 | tee debug.log
```

### Reasoning Integration Debug
For detailed reasoning integration debugging:
```bash
# Enable verbose mode to see reasoning result discovery process
python handler.py --reasoning-results --verbose

# Check what reasoning results are being discovered
python -c "
import json
from pathlib import Path
results_dir = Path.home() / 'aws-neuron-samples/inference-benchmarking/results/accuracy/mytest'
for benchmark_dir in results_dir.iterdir():
    if benchmark_dir.is_dir():
        print(f'Found benchmark: {benchmark_dir.name}')
        for model_dir in benchmark_dir.iterdir():
            if model_dir.is_dir():
                print(f'  Model: {model_dir.name}')
                for result_file in model_dir.glob('results_*.json'):
                    print(f'    Result: {result_file.name}')
"
```

## 📝 Input File Formats

### `benchmark_inference.json`
```json
{
  "e2e_model": {
    "latency_ms_p50": 12143.92,
    "latency_ms_p90": 12169.44,
    "latency_ms_p95": 12182.64,
    "latency_ms_p99": 12189.53,
    "latency_ms_p100": 12191.26,
    "latency_ms_avg": 12131.49,
    "throughput": 52.76
  },
  "context_encoding_model": {
    "latency_ms_avg": 43.01,
    "throughput": 4440.69
  },
  "token_generation_model": {
    "latency_ms_avg": 15.58,
    "throughput": 64.33
  }
}
```

### Training Config YAML
```yaml
data:
  global_batch_size: 64
  seq_length: 4096

model:
  name: "llama3-8b"
  
training:
  num_epochs: 3
  learning_rate: 1e-4
```

### Reasoning Results JSON
```json
{
  "results": {
    "gsm8k_cot": {
      "alias": "gsm8k_cot",
      "exact_match,strict-match": 0.555,
      "exact_match_stderr,strict-match": 0.0352289710609046,
      "exact_match,flexible-extract": 0.575,
      "exact_match_stderr,flexible-extract": 0.03504304603451135
    }
  },
  "n-samples": {
    "gsm8k_cot": {
      "original": 1319,
      "effective": 200
    }
  },
  "model_name": "/home/ubuntu/models/llama-3-1-8b",
  "model_name_sanitized": "__home__ubuntu__models__llama-3-1-8b"
}
```

### Reasoning Results Directory Structure
```
aws-neuron-samples/inference-benchmarking/results/accuracy/mytest/
├── gsm8k_cot/
│   └── __home__ubuntu__models__llama-3-1-8b/
│       └── results_2025-06-23T01-34-27.025863.json
├── mmlu_pro/
│   └── __home__ubuntu__models__llama-3-1-8b/
│       └── results_2025-06-23T01-35-15.123456.json
└── mmlu_flan_cot_zeroshot/
    └── __home__ubuntu__models__llama-3-1-8b/
        └── results_2025-06-23T01-36-42.789012.json
```

---

**Note**: Default paths assume standard NKI-LLAMA directory structure. Adjust paths according to your setup.