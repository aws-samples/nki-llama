# NKI-Reasoning Reasoning Benchmark Setup Guide

This guide walks you through setting up and running reasoning benchmarks on your compiled model using the lm-eval. The benchmark evaluates model performance on reasoning tasks including GSM8K Chain-of-Thought, MMLU, and MMLU Pro datasets.

## Prerequisites

- Ensure you are in the pytorch_2_6 nxdi venv
- Ensure you have run `./nki-llama inference setup` first
- tmux for session management - start a new session eg: `tmux new -s bench`

## Step 1: Clone the Repository

Navigate to the home directory and clone the AWS Neuron samples repository:

```bash
cd /home/ubuntu
git clone https://github.com/aws-neuron/aws-neuron-samples.git
```

## Step 2: Configure Benchmark Datasets

For the reasoning benchmark, we'll use three key datasets:
- **gsm8k_cot**: Grade School Math problems with Chain-of-Thought reasoning
- **mmlu_flan_cot_zeroshot**: Massive Multitask Language Understanding with zero-shot prompting
- **mmlu_pro**: Professional-level MMLU questions

### Edit the Accuracy Configuration

Navigate to the inference benchmarking directory and modify the accuracy configuration:

```bash
cd /home/ubuntu/aws-neuron-samples/inference-benchmarking
vim accuracy.py
```

Update line 18 to include the required datasets in the `ACCURACY_CLIENTS_DATASETS` dictionary:

```python
ACCURACY_CLIENTS_DATASETS = {
    "lm_eval": [
        "gsm8k_cot",
        "mmlu_flan_cot_zeroshot", # add
        "mmlu_pro", #add
        "mmlu_flan_n_shot_generative_computer_security",
        "mmlu_flan_n_shot_generative_logical_fallacies",
        "mmlu_flan_n_shot_generative_nutrition",
    ],
}
```

## Step 3: Configure Sequence Length Requirements

**Important**: Reasoning benchmarks require a minimum recommended sequence length of 4096 tokens during model compilation.

### Set Environment Variables

Navigate to your NKI-LLaMA directory and configure the `.env` file:

```bash
cd /home/ubuntu/nki-llama/
vim .env
```

Ensure the following environment variables are set:

```bash
# For compilation with main.py
SEQ_LEN=4096

# For runtime inference
MAX_MODEL_LEN=4096
```

## Step 4: Configure Model Paths

The benchmark script sources environment variables from the top-level `nki-llama` directory. You have two options:

### Option A: Use Environment Variables (Recommended)

Ensure your `.env` file contains:

```bash
MODEL_NAME="your-model-name"
NKI_MODELS="/path/to/your/models"
NKI_COMPILED="/path/to/compiled/models"
TENSOR_PARALLEL_SIZE=1
INFERENCE_PORT=8000
MAX_MODEL_LEN=4096
```

### Option B: Hardcode Model Paths

The benchmark script generates a YAML configuration file. Here's what it looks like:

```yaml
server:
  name: "Reasoning-benchmark server"
  model_path: "${NKI_MODELS}/${MODEL_NAME}"
  model_s3_path: null
  compiled_model_path: "${NKI_COMPILED}/${MODEL_NAME}"
  max_seq_len: ${MAX_MODEL_LEN}
  context_encoding_len: ${MAX_MODEL_LEN}
  tp_degree: ${TENSOR_PARALLEL_SIZE}
  n_vllm_threads: ${TENSOR_PARALLEL_SIZE}
  server_port: ${INFERENCE_PORT}
  continuous_batch_size: 1

test:
  accuracy:
    mytest:
      client: "lm_eval"
      datasets: ["mmlu_pro", "gsm8k_cot", "mmlu_flan_cot_zeroshot"]
      max_concurrent_requests: 1
      timeout: 3600
      client_params:
        limit: 200
        use_chat: False
```

## Step 5: Running Multiple Model Comparisons

If you're comparing base and fine-tuned models, ensure you update the model paths between runs:

### For Base Model Run:
```bash
export MODEL_NAME="base-model-name"
```

### For Fine-tuned Model Run:
```bash
export MODEL_NAME="fine-tuned-model-name"
```

Alternatively, you can directly edit the generated YAML file to hardcode specific model paths.

## Step 6: Execute the Benchmark

### Start a tmux Session

```bash
tmux new-session -d -s reasoning-benchmark
tmux attach-session -t reasoning-benchmark
```

### Activate Virtual Environment and Run Benchmark

```bash
# Navigate to the inference scripts directory
cd /home/ubuntu/nki-llama/src/inference/scripts/

# Activate your virtual environment
source /path/to/your/venv/bin/activate

# Run the reasoning benchmark
./reasoning-bench-lm-eval.sh
```

## Step 7: Results Analysis

After completion, the benchmark will generate results in the `aws-neuron-samples` directory under a path specific to your model.

### Expected Output Format

The results will be saved as a JSON file with the following structure:

```json
{
  "results": {
    "gsm8k_cot": {
      "alias": "gsm8k_cot",
      "exact_match,strict-match": 0.78,
      "exact_match_stderr,strict-match": 0.029365141882663297,
      "exact_match,flexible-extract": 0.72,
      "exact_match_stderr,flexible-extract": 0.03182868716477582
    },
    "mmlu_flan_cot_zeroshot": {
      "alias": "mmlu_flan_cot_zeroshot",
      "acc,none": 0.65,
      "acc_stderr,none": 0.0234
    },
    "mmlu_pro": {
      "alias": "mmlu_pro", 
      "acc,none": 0.42,
      "acc_stderr,none": 0.0189
    }
  },
  "group_subtasks": {
    "gsm8k_cot": [],
    "mmlu_flan_cot_zeroshot": [],
    "mmlu_pro": []
  },
  "configs": {
    "gsm8k_cot": {
      "task": "gsm8k_cot",
      "dataset_path": "gsm8k",
      "test_split": "test",
      "doc_to_text": "Question: {{question}}\nAnswer:",
      "doc_to_target": "{{answer}}",
      "description": "Answer the following question with step-by-step reasoning."
    }
  },
  "versions": {
    "gsm8k_cot": 1,
    "mmlu_flan_cot_zeroshot": 1,
    "mmlu_pro": 1
  },
  "n-shot": {
    "gsm8k_cot": 0,
    "mmlu_flan_cot_zeroshot": 0,
    "mmlu_pro": 0
  },
  "higher_is_better": {
    "gsm8k_cot": {
      "exact_match,strict-match": true,
      "exact_match,flexible-extract": true
    },
    "mmlu_flan_cot_zeroshot": {
      "acc,none": true
    },
    "mmlu_pro": {
      "acc,none": true
    }
  }
}
```

## Understanding the Results

### Key Metrics

- **exact_match,strict-match**: Percentage of exactly correct answers using strict matching
- **exact_match,flexible-extract**: Percentage of correct answers using flexible extraction
- **acc,none**: Overall accuracy percentage
- **stderr**: Standard error of the measurement

### Benchmark Interpretations

- **GSM8K CoT**: Measures mathematical reasoning ability with step-by-step problem solving
- **MMLU**: Evaluates broad knowledge across multiple academic domains
- **MMLU Pro**: Tests professional-level understanding and application

## Troubleshooting

### Common Issues

1. **Sequence Length Errors**: Ensure `SEQ_LEN` and `MAX_MODEL_LEN` are set to at least 4096
2. **Model Path Issues**: Verify environment variables or hardcoded paths are correct
3. **Memory Issues**: Consider adjusting `TENSOR_PARALLEL_SIZE` based on your hardware
4. **Timeout Errors**: Increase the timeout value in the YAML configuration if needed

### Debug Commands

```bash
# Check environment variables
cd /home/ubuntu/nki-llama/
source .env
env | grep -E "(MODEL_NAME|SEQ_LEN|MAX_MODEL_LEN)"

# Verify model paths exist
ls -la ${NKI_MODELS}/${MODEL_NAME}
ls -la ${NKI_COMPILED}/${MODEL_NAME}

# Check tmux sessions
tmux list-sessions
```

## Best Practices

1. **Use tmux**: Long-running benchmarks benefit from persistent sessions
2. **Monitor Resources**: Keep an eye on GPU/CPU usage during execution
3. **Save Results**: Archive results with timestamps for comparison
4. **Document Changes**: Keep track of configuration changes between runs
5. **Version Control**: Use git to track modifications to benchmark scripts

This comprehensive setup ensures reliable and reproducible reasoning benchmark results for your AWS Neuron model evaluations.
