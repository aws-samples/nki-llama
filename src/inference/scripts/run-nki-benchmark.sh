#!/bin/bash
# /home/ubuntu/nki-llama/src/inference/scripts/run-nki-benchmark.sh
# Run NKI benchmark evaluation for model compilation and performance testing

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# Load configuration
source "${PROJECT_ROOT}/nki-llama.config"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default parameters
MODE="${MODE:-evaluate_single}"
ENABLE_NKI="${ENABLE_NKI:-true}"
SEQ_LEN="${SEQ_LEN:-640}"
TP_DEGREE="${TP_DEGREE:-${TENSOR_PARALLEL_SIZE}}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --no-nki)
            ENABLE_NKI="false"
            shift
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --mode MODE             Benchmark mode (evaluate_single/evaluate_all) [default: evaluate_single]"
            echo "  --model-name NAME       Model name override"
            echo "  --no-nki               Disable NKI optimizations"
            echo "  --seq-len N            Sequence length [default: 640]"
            echo "  --tp-degree N          Tensor parallel degree [default: from config]"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set paths
MODEL_PATH="${NKI_MODELS}/${MODEL_NAME}"
COMPILED_MODEL_PATH="${NKI_COMPILED}/${MODEL_NAME}"

# Function to check if model exists
check_model() {
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo -e "${RED}❌ Model not found at: $MODEL_PATH${NC}"
        echo -e "${YELLOW}Please run: ./nki-llama inference download${NC}"
        exit 1
    fi
}

# Function to check compilation cache
# check_compiled_model() {
#     if [[ -d "$COMPILED_MODEL_PATH" ]]; then
#         echo -e "${GREEN}✓ Found compiled model cache at: $COMPILED_MODEL_PATH${NC}"
#         return 0
#     else
#         echo -e "${YELLOW}⚠ No compiled model found. Will compile during benchmark.${NC}"
#         return 1
#     fi
# }

# Function to run evaluate_single mode
run_evaluate_single() {
    echo -e "${YELLOW}🔧 Running benchmark in evaluate_single mode...${NC}"
    echo -e "${YELLOW}This mode runs from repository test script for single evaluation.${NC}"
    
    # Change to home directory and run the test script
    cd ~
    
    # Build command
    CMD="python ${NKI_ROOT}/test/inference/test.py"
    CMD="${CMD} --repository-path ${NKI_ROOT}"
    
    # Execute with timing
    echo -e "${BLUE}Executing evaluate_single benchmark...${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    START_TIME=$(date +%s)
    
    if $CMD 2>&1 | tee "${BENCHMARK_LOG_DIR}/benchmark.log"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ evaluate_single benchmark completed successfully!${NC}"
        echo -e "Total time: ${DURATION} seconds"
        
        return 0
    else
        echo -e "${RED}✗ evaluate_single benchmark failed!${NC}"
        return 1
    fi
}

# Function to run evaluate_all mode
run_evaluate_all() {
    echo -e "${YELLOW}📊 Running benchmark in evaluate_all mode...${NC}"
    echo -e "${YELLOW}This mode evaluates all model configurations with NKI optimizations.${NC}"
    
    # Change to inference directory
    cd "${NKI_INFERENCE}"
    
    # Build command
    CMD="python main.py"
    CMD="${CMD} --mode evaluate_all"
    CMD="${CMD} --model-path ${MODEL_PATH}"
    CMD="${CMD} --compiled-model-path ${COMPILED_MODEL_PATH}"
    CMD="${CMD} --seq-len ${SEQ_LEN}"
    CMD="${CMD} --tp-degree ${TP_DEGREE}"
    
    if [[ "$ENABLE_NKI" == "true" ]]; then
        CMD="${CMD} --enable-nki"
    fi
    
    # Execute with timing
    echo -e "${BLUE}Executing evaluate_all benchmark...${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    START_TIME=$(date +%s)
    
    if $CMD 2>&1 | tee "${BENCHMARK_LOG_DIR}/benchmark.log"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ evaluate_all benchmark completed successfully!${NC}"
        echo -e "Total time: ${DURATION} seconds"
        
        # If compilation happened, show artifact info
        if [[ -d "$COMPILED_MODEL_PATH" ]]; then
            echo
            echo -e "${GREEN}✓ NKI-compiled model artifacts available at:${NC}"
            echo -e "   ${COMPILED_MODEL_PATH}"
            echo
            echo -e "${BLUE}These artifacts can now be used for:${NC}"
            echo -e "  • vLLM inference with NxDI optimizations"
            echo -e "  • Direct inference benchmarks"
            echo -e "  • Production deployments"
            echo
        fi
        
        return 0
    else
        echo -e "${RED}✗ evaluate_all benchmark failed!${NC}"
        return 1
    fi
}

# Main benchmark function
run_benchmark() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}NKI Benchmark Evaluation${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "Mode:              ${CYAN}${MODE}${NC}"
    echo -e "Model:             ${CYAN}${MODEL_NAME}${NC}"
    echo -e "Model Path:        ${CYAN}${MODEL_PATH}${NC}"
    echo -e "Compiled Path:     ${CYAN}${COMPILED_MODEL_PATH}${NC}"
    echo -e "NKI Enabled:       ${CYAN}${ENABLE_NKI}${NC}"
    echo -e "Sequence Length:   ${CYAN}${SEQ_LEN}${NC}"
    echo -e "TP Degree:         ${CYAN}${TP_DEGREE}${NC}"
    echo
    
    # Check prerequisites based on mode
    if [[ "$MODE" == "evaluate_all" ]]; then
        check_model
        # check_compiled_model
    fi
    
    # Set environment variables for the benchmark
    export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES}"
    export NEURON_CC_FLAGS="--enable-saturate-infinity"
    
    # Additional NKI-specific flags if enabled
    if [[ "$ENABLE_NKI" == "true" ]]; then
        export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
        echo -e "${GREEN}✓ NKI optimizations enabled${NC}"
    fi
    
    # Create log directory for this benchmark
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BENCHMARK_LOG_DIR="${NKI_LOGS}/benchmarks/${TIMESTAMP}"
    mkdir -p "$BENCHMARK_LOG_DIR"
    
    echo -e "${BLUE}📊 Benchmark logs will be saved to:${NC}"
    echo -e "   ${BENCHMARK_LOG_DIR}"
    echo
    
    # Save benchmark metadata
    cat > "${BENCHMARK_LOG_DIR}/metadata.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "mode": "${MODE}",
    "model_name": "${MODEL_NAME}",
    "model_path": "${MODEL_PATH}",
    "compiled_model_path": "${COMPILED_MODEL_PATH}",
    "nki_enabled": ${ENABLE_NKI},
    "sequence_length": ${SEQ_LEN},
    "tensor_parallel_size": ${TP_DEGREE},
    "neuron_rt_cores": "${NEURON_RT_NUM_CORES}"
}
EOF
    
    # Run the appropriate benchmark mode
    case "$MODE" in
        evaluate_single)
            if run_evaluate_single; then
                RESULT="success"
            else
                RESULT="failed"
            fi
            ;;
        evaluate_all)
            if run_evaluate_all; then
                RESULT="success"
            else
                RESULT="failed"
            fi
            ;;
        *)
            echo -e "${RED}Unknown mode: $MODE${NC}"
            echo -e "Valid modes: evaluate_single, evaluate_all"
            exit 1
            ;;
    esac
    
    # Update metadata with result
    if [[ "$RESULT" == "success" ]]; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        # Update metadata.json with duration
        jq --arg duration "$DURATION" '.duration_seconds = ($duration | tonumber)' \
            "${BENCHMARK_LOG_DIR}/metadata.json" > "${BENCHMARK_LOG_DIR}/metadata.json.tmp" && \
            mv "${BENCHMARK_LOG_DIR}/metadata.json.tmp" "${BENCHMARK_LOG_DIR}/metadata.json"
    fi
}

# Show benchmark info
show_info() {
    echo -e "${BLUE}NKI Benchmark Evaluation Tool${NC}"
    echo
    echo -e "This tool supports two benchmark modes:"
    echo
    echo -e "${YELLOW}1. evaluate_single mode:${NC}"
    echo -e "   • Runs benchmark from repository test script"
    echo -e "   • Single evaluation configuration"
    echo -e "   • Quick validation of model performance"
    echo
    echo -e "${YELLOW}2. evaluate_all mode:${NC}"
    echo -e "   • Comprehensive benchmark with all configurations"
    echo -e "   • Tests with NKI optimizations"
    echo -e "   • Creates compiled model artifacts if needed"
    echo -e "   • Full performance analysis"
    echo
    echo -e "${BLUE}Examples:${NC}"
    echo -e "   # Run single evaluation"
    echo -e "   ./run-nki-benchmark.sh --mode evaluate_single"
    echo
    echo -e "   # Run comprehensive benchmark with NKI"
    echo -e "   ./run-nki-benchmark.sh --mode evaluate_all --seq-len 1024"
    echo
    echo -e "   # Run without NKI optimizations"
    echo -e "   ./run-nki-benchmark.sh --mode evaluate_all --no-nki"
    echo
}

# Main execution
if [[ "${1:-}" == "--info" ]]; then
    show_info
else
    run_benchmark
fi