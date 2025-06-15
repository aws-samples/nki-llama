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
SEQ_LEN="${SEQ_LEN:-2048}"
TP_DEGREE="${TP_DEGREE:-${TENSOR_PARALLEL_SIZE}}"
CLEAR_CACHE="${CLEAR_CACHE:-false}"
AUTO_CLEAR_CACHE="${AUTO_CLEAR_CACHE:-true}"
RETRY_FAILED="${RETRY_FAILED:-false}"

# Cache paths
NEURON_CACHE_DIR="/var/tmp/neuron-compile-cache"

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
        --clear-cache)
            CLEAR_CACHE="true"
            shift
            ;;
        --no-auto-clear-cache)
            AUTO_CLEAR_CACHE="false"
            shift
            ;;
        --retry-failed-compilation)
            RETRY_FAILED="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --mode MODE                    Benchmark mode (evaluate_single/evaluate_all) [default: evaluate_single]"
            echo "  --model-name NAME              Model name override"
            echo "  --no-nki                       Disable NKI optimizations"
            echo "  --seq-len N                    Sequence length [default: 2048]"
            echo "  --tp-degree N                  Tensor parallel degree [default: from config]"
            echo "  --clear-cache                  Clear compilation cache before running"
            echo "  --no-auto-clear-cache          Disable automatic cache clearing on failure"
            echo "  --retry-failed-compilation     Force retry of failed compilations"
            echo "  --help                         Show this help message"
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

# Function to clear compilation cache
clear_compilation_cache() {
    echo -e "${YELLOW}🧹 Clearing Neuron compilation cache...${NC}"
    if [[ -d "$NEURON_CACHE_DIR" ]]; then
        local cache_size=$(du -sh "$NEURON_CACHE_DIR" 2>/dev/null | cut -f1 || echo "unknown")
        echo -e "   Cache location: ${CYAN}${NEURON_CACHE_DIR}${NC}"
        echo -e "   Current size: ${CYAN}${cache_size}${NC}"
        
        if rm -rf "$NEURON_CACHE_DIR"; then
            echo -e "${GREEN}✓ Cache cleared successfully${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to clear cache. May need sudo privileges.${NC}"
            echo -e "${YELLOW}Try: sudo rm -rf ${NEURON_CACHE_DIR}${NC}"
            return 1
        fi
    else
        echo -e "${BLUE}ℹ Cache directory does not exist${NC}"
        return 0
    fi
}

# Function to check for failed cache entries
check_failed_cache_entries() {
    if [[ -d "$NEURON_CACHE_DIR" ]]; then
        local failed_count=$(find "$NEURON_CACHE_DIR" -name "*.neff" -size 0 2>/dev/null | wc -l || echo "0")
        if [[ $failed_count -gt 0 ]]; then
            echo -e "${YELLOW}⚠ Found ${failed_count} failed compilation entries in cache${NC}"
            return 1
        fi
    fi
    return 0
}

# Function to run evaluate_single mode
run_evaluate_single() {
    echo -e "${YELLOW}🔧 Running benchmark in evaluate_single mode...${NC}"
    echo -e "${YELLOW}This mode runs single evaluation with NKI optimizations.${NC}"
    
    # Change to inference directory
    cd "${NKI_INFERENCE}"
    
    # Build command
    CMD="python main.py"
    CMD="${CMD} --mode evaluate_single"
    CMD="${CMD} --model-path ${MODEL_PATH}"
    CMD="${CMD} --compiled-model-path ${COMPILED_MODEL_PATH}"
    CMD="${CMD} --seq-len ${SEQ_LEN}"
    CMD="${CMD} --tp-degree ${TP_DEGREE}"
    
    if [[ "$ENABLE_NKI" == "true" ]]; then
        CMD="${CMD} --enable-nki"
    fi
    
    if [[ "$RETRY_FAILED" == "true" ]]; then
        CMD="${CMD} --retry-failed-compilation"
    fi
    
    # Execute with timing and error handling
    echo -e "${BLUE}Executing evaluate_single benchmark...${NC}"
    echo -e "${BLUE}${CMD}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    START_TIME=$(date +%s)
    
    # Create a temporary file to capture the output
    TEMP_LOG=$(mktemp)
    
    # Run command and capture both stdout/stderr
    if $CMD 2>&1 | tee "${BENCHMARK_LOG_DIR}/benchmark.log" | tee "$TEMP_LOG"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ evaluate_single benchmark completed successfully!${NC}"
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
        
        rm -f "$TEMP_LOG"
        return 0
    else
        # Check if it's a cache-related failure
        if grep -q "Got a cached failed neff" "$TEMP_LOG" || grep -q "SIGHUP" "$TEMP_LOG"; then
            echo
            echo -e "${RED}✗ evaluate_single benchmark failed due to compilation cache issues!${NC}"
            
            if [[ "$AUTO_CLEAR_CACHE" == "true" ]]; then
                echo -e "${YELLOW}🔄 Attempting automatic cache recovery...${NC}"
                echo
                
                # Clear the cache
                if clear_compilation_cache; then
                    echo
                    echo -e "${YELLOW}🔄 Retrying benchmark with clean cache...${NC}"
                    echo
                    
                    # Retry the command
                    if $CMD 2>&1 | tee "${BENCHMARK_LOG_DIR}/benchmark_retry.log"; then
                        END_TIME=$(date +%s)
                        DURATION=$((END_TIME - START_TIME))
                        
                        echo
                        echo -e "${GREEN}✓ evaluate_single benchmark completed successfully after cache clear!${NC}"
                        echo -e "Total time: ${DURATION} seconds"
                        
                        rm -f "$TEMP_LOG"
                        return 0
                    else
                        echo -e "${RED}✗ Benchmark still failed after cache clear${NC}"
                    fi
                else
                    echo -e "${RED}✗ Could not clear cache automatically${NC}"
                fi
            else
                echo
                echo -e "${YELLOW}💡 Suggestions to fix:${NC}"
                echo -e "   1. Clear the compilation cache:"
                echo -e "      ${CYAN}rm -rf ${NEURON_CACHE_DIR}${NC}"
                echo -e "   2. Re-run with auto cache clearing:"
                echo -e "      ${CYAN}$0 --mode evaluate_single --clear-cache${NC}"
                echo -e "   3. Force retry failed compilations:"
                echo -e "      ${CYAN}$0 --mode evaluate_single --retry-failed-compilation${NC}"
            fi
        else
            echo -e "${RED}✗ evaluate_single benchmark failed!${NC}"
        fi
        
        rm -f "$TEMP_LOG"
        return 1
    fi
}

# Function to run evaluate_all mode with error handling
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
    
    if [[ "$RETRY_FAILED" == "true" ]]; then
        CMD="${CMD} --retry-failed-compilation"
    fi
    
    # Execute with timing and error handling
    echo -e "${BLUE}Executing evaluate_all benchmark...${NC}"
    echo -e "${BLUE}${CMD}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    START_TIME=$(date +%s)
    
    # Create a temporary file to capture the output
    TEMP_LOG=$(mktemp)
    
    # Run command and capture both stdout/stderr
    if $CMD 2>&1 | tee "${BENCHMARK_LOG_DIR}/benchmark.log" | tee "$TEMP_LOG"; then
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
        
        rm -f "$TEMP_LOG"
        return 0
    else
        # Check if it's a cache-related failure
        if grep -q "Got a cached failed neff" "$TEMP_LOG" || grep -q "SIGHUP" "$TEMP_LOG"; then
            echo
            echo -e "${RED}✗ evaluate_all benchmark failed due to compilation cache issues!${NC}"
            
            if [[ "$AUTO_CLEAR_CACHE" == "true" ]]; then
                echo -e "${YELLOW}🔄 Attempting automatic cache recovery...${NC}"
                echo
                
                # Clear the cache
                if clear_compilation_cache; then
                    echo
                    echo -e "${YELLOW}🔄 Retrying benchmark with clean cache...${NC}"
                    echo
                    
                    # Retry the command
                    if $CMD 2>&1 | tee "${BENCHMARK_LOG_DIR}/benchmark_retry.log"; then
                        END_TIME=$(date +%s)
                        DURATION=$((END_TIME - START_TIME))
                        
                        echo
                        echo -e "${GREEN}✓ evaluate_all benchmark completed successfully after cache clear!${NC}"
                        echo -e "Total time: ${DURATION} seconds"
                        
                        rm -f "$TEMP_LOG"
                        return 0
                    else
                        echo -e "${RED}✗ Benchmark still failed after cache clear${NC}"
                    fi
                else
                    echo -e "${RED}✗ Could not clear cache automatically${NC}"
                fi
            else
                echo
                echo -e "${YELLOW}💡 Suggestions to fix:${NC}"
                echo -e "   1. Clear the compilation cache:"
                echo -e "      ${CYAN}rm -rf ${NEURON_CACHE_DIR}${NC}"
                echo -e "   2. Re-run with auto cache clearing:"
                echo -e "      ${CYAN}$0 --mode evaluate_all --clear-cache${NC}"
                echo -e "   3. Force retry failed compilations:"
                echo -e "      ${CYAN}$0 --mode evaluate_all --retry-failed-compilation${NC}"
            fi
        else
            echo -e "${RED}✗ evaluate_all benchmark failed!${NC}"
        fi
        
        rm -f "$TEMP_LOG"
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
    echo -e "Auto Clear Cache:  ${CYAN}${AUTO_CLEAR_CACHE}${NC}"
    echo
    
    # Check if we should clear cache
    if [[ "$CLEAR_CACHE" == "true" ]]; then
        clear_compilation_cache
        echo
    else
        # Check for failed cache entries
        if ! check_failed_cache_entries; then
            echo -e "${YELLOW}💡 Consider using --clear-cache to remove failed entries${NC}"
            echo
        fi
    fi
    
    # Check prerequisites based on mode
    if [[ "$MODE" == "evaluate_all" ]] || [[ "$MODE" == "evaluate_single" ]]; then
        check_model
    fi
    
    # Check if running in tmux for long compilations
    if [[ "$MODE" == "evaluate_all" ]] && [[ -z "${TMUX:-}" ]]; then
        echo -e "${YELLOW}⚠️  Warning: Not running in tmux!${NC}"
        echo -e "${YELLOW}   Model compilation can take 10-30 minutes.${NC}"
        echo -e "${YELLOW}   Any disconnection will terminate the process.${NC}"
        echo
        echo -e "${CYAN}   Recommended: tmux new -s benchmark${NC}"
        echo
        read -p "Continue without tmux? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Exiting. Please run in tmux.${NC}"
            exit 0
        fi
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
    "neuron_rt_cores": "${NEURON_RT_NUM_CORES}",
    "cache_cleared": ${CLEAR_CACHE},
    "auto_clear_cache": ${AUTO_CLEAR_CACHE},
    "retry_failed": ${RETRY_FAILED}
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
    echo -e "   • Runs single evaluation configuration"
    echo -e "   • Tests with NKI optimizations"
    echo -e "   • Quick validation of model performance"
    echo
    echo -e "${YELLOW}2. evaluate_all mode:${NC}"
    echo -e "   • Comprehensive benchmark with all configurations"
    echo -e "   • Tests with NKI optimizations"
    echo -e "   • Creates compiled model artifacts if needed"
    echo -e "   • Full performance analysis"
    echo
    echo -e "${BLUE}Cache Management:${NC}"
    echo -e "   • Auto-detects and handles failed compilations"
    echo -e "   • Can automatically clear cache on failure"
    echo -e "   • Manual cache clearing available"
    echo
    echo -e "${BLUE}Examples:${NC}"
    echo -e "   # Run single evaluation"
    echo -e "   ./run-nki-benchmark.sh --mode evaluate_single"
    echo
    echo -e "   # Run comprehensive benchmark with NKI"
    echo -e "   ./run-nki-benchmark.sh --mode evaluate_all --seq-len 1024"
    echo
    echo -e "   # Clear cache before running"
    echo -e "   ./run-nki-benchmark.sh --mode evaluate_all --clear-cache"
    echo
    echo -e "   # Run without automatic cache clearing"
    echo -e "   ./run-nki-benchmark.sh --mode evaluate_all --no-auto-clear-cache"
    echo
}

# Main execution
if [[ "${1:-}" == "--info" ]]; then
    show_info
else
    run_benchmark
fi