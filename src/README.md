# NKI-LLAMA Benchmark Handler

A benchmarking system for evaluating NKI-LLAMA model performance across both training and inference metrics.

## 🚀 Overview

The NKI-LLAMA Benchmark Handler calculates a unified performance score that combines:
- **Training metrics**: MFU (Model FLOPs Utilization), throughput, and NKI kernel usage
- **Inference metrics**: Latency, throughput, and accuracy (optional)
- **NKI optimization**: Ratio of NKI (Neuron Kernel Interface) operations to total operations

The system supports two modes:
- **Training-only mode**: When inference results are not available, provides NKI kernel training score
- **Combined mode**: When both training and inference results are available, provides full NKI-LLAMA score

The final score follows the formula:
```
Score = Accuracy × Reduced Latency × Increased Throughput × (1 + Normalized NKI FLOPS)
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
python handler.py \
    --training-weight 0.3 \
    --inference-weight 0.7
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
  "mode": "combined",
  "nki_kernel_score": 0.0046,
  "component_scores": {
    "training": 0.0077,
    "inference": 0.0026
  },
  "weights": {
    "training": 0.4,
    "inference": 0.6
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

### Debug Mode
Run with verbose flag to see detailed execution:
```bash
python handler.py --verbose 2>&1 | tee debug.log
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

---

**Note**: Default paths assume standard NKI-LLAMA directory structure. Adjust paths according to your setup.