# Training Metrics Calculator

A tool for calculating training metrics including Model FLOPs Utilization (MFU), NKI (Neuron Kernel Interface) usage analysis, and training performance scores from AWS Neuron training logs and HLO files.

## Overview

This tool analyzes training performance on AWS Trainium instances by:
- Calculating MFU (Model FLOPs Utilization) percentage
- Analyzing NKI kernel usage across all compiled HLO modules
- Computing training performance scores both overall and per-file
- Extracting metrics from training logs
- Providing detailed breakdowns of performance improvements

## Features

- **MFU Calculation**: Computes the percentage of theoretical peak FLOPs achieved during training
- **NKI Analysis**: Identifies and quantifies custom NKI kernel usage vs standard operations
- **Training Score**: Calculates a comprehensive performance score based on multiple factors
- **Per-File Analysis**: Breaks down performance metrics for individual HLO modules
- **Log Parsing**: Extracts throughput and loss metrics from training logs
- **Flexible Configuration**: Supports various model configurations and hardware backends

## Usage

### Example

```bash
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Get report over all training jobs in neuron_cache directory
python /home/ubuntu/nki-llama/src/fine-tune/scripts/calculate_training_metrics.py \
      --config /home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml \
      --model-config /home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json \
      --log-file /home/ubuntu/nki-llama/logs/nki-llama_20250610_014432.log \
      --compile-dir /home/ubuntu/neuron_cache \
      --throughput 2.1 \
      --hw-backend trn1 \
      --calculate-score \
      --per-file-scores \
      --detailed \
      --print-per-file \
      --output benchmark_finetuning.json

# Get report over a training job in neuron_cache directory
python /home/ubuntu/nki-llama/src/fine-tune/scripts/calculate_training_metrics.py \
      --config /home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml \
      --model-config /home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json \
      --log-file /home/ubuntu/nki-llama/logs/nki-llama_20250610_014432.log \
      --compile-dir /home/ubuntu/neuron_cache/neuronxcc-2.18.121.0+9e31e41a/MODULE_15329989265349737271+a65e371e \
      --throughput 2.1 \
      --hw-backend trn1 \
      --calculate-score \
      --per-file-scores \
      --detailed \
      --print-per-file \
      --output benchmark_finetuning.json
```

### Basic Usage

```bash
python calculate_training_metrics.py \
  --config training_config.yaml \
  --model-config model_config.json
```

### Advanced Usage with All Features

```bash
python calculate_training_metrics.py \
  --config training_config.yaml \
  --model-config model_config.json \
  --compile-dir /home/ubuntu/neuron_cache \
  --log-file training.log \
  --batch-size 32 \
  --seq-len 2048 \
  --throughput 150.5 \
  --num-nodes 4 \
  --hw-backend trn1 \
  --calculate-score \
  --per-file-scores \
  --print-per-file \
  --detailed \
  --base-mfu 50.0 \
  --base-throughput 100.0 \
  --loss-improvement 1.2 \
  --convergence-rate 1.1 \
  --output metrics_report.json
```

## Command Line Arguments

### Required Arguments

- `--config`: Path to the training configuration YAML file
- `--model-config`: Path to the model configuration JSON file

### Optional Arguments

#### Basic Configuration
- `--compile-dir`: Neuron compile cache directory (default: `/home/ubuntu/neuron_cache`)
- `--log-file`: Training log file to parse for metrics
- `--batch-size`: Global batch size (overrides config file)
- `--seq-len`: Sequence length (overrides config file)
- `--throughput`: Throughput in sequences/second (if known)
- `--num-nodes`: Number of nodes (default: 1)
- `--hw-backend`: Hardware backend - `trn1` or `trn2` (default: `trn1`)
- `--output`: Output metrics file (default: `training_metrics.json`)

#### Display Options
- `--detailed`: Include detailed per-file metrics in JSON output
- `--print-per-file`: Print per-file metrics table to console

#### Training Score Parameters
- `--calculate-score`: Calculate final training score
- `--per-file-scores`: Calculate training scores for each file individually
- `--base-mfu`: Baseline MFU percentage for score calculation (default: 50.0)
- `--base-throughput`: Baseline throughput in seq/s for score calculation (default: 100.0)
- `--loss-improvement`: Loss improvement ratio (baseline_loss/achieved_loss) (default: 1.0)
- `--convergence-rate`: Convergence rate improvement (baseline_steps/achieved_steps) (default: 1.0)

## Output Format

### Console Output

The tool provides detailed console output including:

1. **Configuration Summary**: Shows the parameters used for calculation
2. **Per-File Analysis** (with `--print-per-file`):
   ```
   File Name                                Module              HLO MACs        NKI MACs  NKI Ratio      Score
   --------------------------------------------------------------------------------------------------------
   model.hlo_module.pb                      MODULE_Model      1,234,567,890    123,456,789     0.1000     1.2345
   ```
3. **Training Score Breakdown** (with `--calculate-score`):
   ```
   Training Score Breakdown:
   ==================================================
   MFU improvement: 1.2000 (50.00% → 60.00%)
   Throughput improvement: 1.5000 (100.00 → 150.00 seq/s)
   Loss improvement: 1.2000
   Convergence rate improvement: 1.1000
   NKI flop ratio: 0.1500
   ==================================================
   Final Training Score: 2.7324
   ```
4. **Summary Statistics**: Overall metrics and NKI analysis summary

### JSON Output

The output JSON file contains:

```json
{
    "model_config": "path/to/model_config.json",
    "batch_size": 32,
    "sequence_length": 2048,
    "num_nodes": 4,
    "hardware_backend": "trn1",
    "throughput_seq_per_sec": 150.5,
    "mfu_percent": 60.0,
    "tflops_per_second": 245.6,
    "nki_analysis": {
        "summary": {
            "total_files": 10,
            "successful_analyses": 10,
            "overall_nki_ratio": 0.15,
            "average_nki_ratio": 0.14,
            "min_nki_ratio": 0.05,
            "max_nki_ratio": 0.25
        },
        "per_file_metrics": [...]  // With --detailed flag
    },
    "training_score": 2.7324,
    "training_score_breakdown": {...}
}
```

## Understanding the Metrics

### MFU (Model FLOPs Utilization)
- Percentage of theoretical peak FLOPs achieved
- Higher is better (typical range: 30-70% for LLMs)
- Depends on model architecture, batch size, and sequence length

### NKI Ratio
- Ratio of NKI (custom kernel) MACs to total MACs
- Higher ratio indicates more optimized kernels
- Range: 0.0 (no NKI) to 1.0 (all NKI)

### Training Score
- Composite metric combining multiple factors:
  - MFU improvement over baseline
  - Throughput improvement over baseline
  - Loss improvement (optional)
  - Convergence rate improvement (optional)
  - NKI utilization bonus
- Formula: `score = mfu_improvement × throughput_improvement × loss_improvement × convergence_rate × (1 + nki_ratio)`

## Example Workflows

### 1. Basic Performance Analysis

```bash
# Just get MFU and basic metrics
python calculate_training_metrics.py \
  --config config.yaml \
  --model-config model.json
```

### 2. Full Training Evaluation

```bash
# Complete analysis with scores
python calculate_training_metrics.py \
  --config config.yaml \
  --model-config model.json \
  --log-file training.log \
  --calculate-score \
  --print-per-file
```

### 3. Comparative Analysis

```bash
# Compare against baseline performance
python calculate_training_metrics.py \
  --config config.yaml \
  --model-config model.json \
  --calculate-score \
  --base-mfu 45.0 \
  --base-throughput 80.0 \
  --loss-improvement 1.15
```

### 4. Debugging NKI Usage

```bash
# Detailed per-file NKI analysis
python calculate_training_metrics.py \
  --config config.yaml \
  --model-config model.json \
  --print-per-file \
  --per-file-scores \
  --detailed
```

## Troubleshooting

### Common Issues

1. **"HLO file not found" errors**
   - Ensure the `--compile-dir` points to the correct Neuron cache directory
   - Check that compilation completed successfully

2. **"Failed to parse NKI backend config" warnings**
   - Normal for non-NKI operations
   - Only affects NKI metric calculation

3. **Low MFU values**
   - Check batch size and sequence length
   - Ensure model is properly optimized for Neuron
   - Consider using larger batch sizes

4. **Zero NKI ratio**
   - Verify NKI kernels are enabled in compilation
   - Check Neuron SDK version supports NKI

### Log File Format

The tool expects training logs with patterns like:
```
step_time: 1.234
throughput: 150.5
seq/s: 150.5
loss: 2.345
```

## Best Practices

1. **Baseline Selection**: Choose realistic baselines that represent:
   - Previous model versions
   - Industry standards
   - Unoptimized implementations

2. **Multiple Runs**: Analyze metrics from multiple training runs to ensure consistency

3. **Complete Analysis**: Use both overall and per-file scores to identify optimization opportunities

4. **Version Tracking**: Save output JSON files with model versions for historical comparison
