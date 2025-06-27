#!/usr/bin/env bash
# reasoning-bench-lm-eval.sh ─ Start vLLM (Neuron) server and run lm-eval reasoning bench

set -euo pipefail

# ---------------------------------------------------------------------
# 0. Config + constants
# ---------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../nki-llama.config" 

set -a                      # auto-export everything that follows
[ -f "${SCRIPT_DIR}/../../../.env" ] && source "${SCRIPT_DIR}/../../../.env"
set +a
# Where we keep AWS Neuron samples
REASONING_BENCH_DIR="$HOME/aws-neuron-samples"

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Setting up vLLM for Neuron …${NC}"

# ---------------------------------------------------------------------
# 1. Sanity check: are we inside the inference venv?
# ---------------------------------------------------------------------
if [[ "${VIRTUAL_ENV:-}" != *"inference"* ]]; then
  echo -e "${RED}Error:${NC} not inside Neuron inference venv"
  echo    "Run: source ${NEURON_INFERENCE_VENV}/bin/activate"
  exit 1
fi

# ---------------------------------------------------------------------
# 2. Clone or update vLLM repo
# ---------------------------------------------------------------------
if [[ -d "$VLLM_REPO" ]]; then
  echo " vLLM repo exists"
else
  echo "Run ./nki-llama inference setup first"
  exit 1
  #git clone -b neuron-2.22-vllm-v0.7.2 https://github.com/aws-neuron/upstreaming-to-vllm.git
fi

# ---------------------------------------------------------------------
# 4. Ensure transformers < 4.50 for Neuron hf_adapter
# ---------------------------------------------------------------------
python - <<'PY'
import subprocess, pkg_resources, sys
req = "4.48.2"
try:
    ver = pkg_resources.get_distribution("transformers").version
except pkg_resources.DistributionNotFound:
    ver = ""
if not ver or pkg_resources.parse_version(ver) >= pkg_resources.parse_version(req):
    print(f"Installing transformers<{req} …")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", f"transformers<{req}"])
PY

echo -e "${GREEN}✓ vLLM (Neuron) ready${NC}"

# ---------------------------------------------------------------------
# 5. Clone/refresh aws-neuron-samples + its deps
# ---------------------------------------------------------------------
if [[ -d "$REASONING_BENCH_DIR" ]]; then
  echo "Updating aws-neuron-samples repo …"
  git -C "$REASONING_BENCH_DIR" pull --ff-only
else
  git clone https://github.com/aws-neuron/aws-neuron-samples.git \
            "$REASONING_BENCH_DIR"
fi

cd "$REASONING_BENCH_DIR/inference-benchmarking"
pip install --quiet -r requirements.txt
echo -e "${GREEN}✓ Inference-Benchmarking deps ready${NC}"

# ---------------------------------------------------------------------
# 6. Write (or overwrite) reasoning_bench.yaml
# ---------------------------------------------------------------------
cat > reasoning_bench.yaml <<YAML
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
      datasets: ["mmlu_pro, gsm8k_cot, mmlu_flan_cot_zeroshot"]
      max_concurrent_requests: 1
      timeout: 3600
      client_params:
        limit: 200
        use_chat: False
YAML
echo -e "${GREEN}✓ Config file written${NC}"

# ---------------------------------------------------------------------
# 7. Run the benchmark
# ---------------------------------------------------------------------
echo -e "${BLUE}----- reasoning_bench.yaml -----${NC}"
cat reasoning_bench.yaml
echo

echo -e "${GREEN}Starting Reasoning Benchmark …${NC}"
python accuracy.py --config reasoning_bench.yaml
