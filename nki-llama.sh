#!/bin/bash
# nki-llama - Unified CLI for fine-tuning and inference

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create symlink for easier access
if [[ -f "$SCRIPT_DIR/nki-llama.sh" ]] && [[ ! -f "$SCRIPT_DIR/nki-llama" ]]; then
    ln -s "$SCRIPT_DIR/nki-llama.sh" "$SCRIPT_DIR/nki-llama"
fi

# Load configuration
if [[ -f "${SCRIPT_DIR}/nki-llama.config" ]]; then
    source "${SCRIPT_DIR}/nki-llama.config"
else
    echo "Error: nki-llama.config not found!"
    exit 1
fi

# Load environment file if exists
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Banner
display_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
    _   __ __ __ ____       __    __       ___    __  ___    ___ 
   / | / // //_//  _/      / /   / /      /   |  /  |/  /   /   |
  /  |/ // ,<   / /______ / /   / /      / /| | / /|_/ /   / /| |
 / /|  // /| |_/ /_______/ /___/ /___   / ___ |/ /  / /   / ___ |
/_/ |_//_/ |_/___/      /_____/_____/  /_/  |_/_/  /_/   /_/  |_|
                                                             
EOF
    echo -e "${NC}"
}

# Tmux helper functions
check_tmux_session() {
    local session_name="$1"
    tmux has-session -t "$session_name" 2>/dev/null
}

suggest_tmux() {
    local operation="$1"
    local session_name="$2"
    shift 2
    local args="$*"
    
    if [[ -z "${TMUX:-}" ]]; then
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}💡 tmux Recommended for ${operation}${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo
        echo -e "This operation may take a long time. We recommend using tmux:"
        echo
        echo -e "${CYAN}# Create new session:${NC}"
        echo -e "tmux new -s ${session_name}"
        echo -e "./nki-llama ${args}"
        echo
        echo -e "${CYAN}# Or run directly in tmux:${NC}"
        echo -e "tmux new -s ${session_name} './nki-llama ${args}'"
        echo
        echo -e "${CYAN}# Detach with: Ctrl+B, D${NC}"
        echo -e "${CYAN}# Reattach with: tmux attach -t ${session_name}${NC}"
        echo
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo
    fi
}

# Check active Neuron environment
check_neuron_env() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo -e "${RED}❌ No virtual environment active${NC}"
        return 1
    elif [[ "$VIRTUAL_ENV" == *"pytorch_2_6"* ]]; then
        echo -e "${GREEN}✓ Fine-tuning environment active${NC}"
        return 0
    elif [[ "$VIRTUAL_ENV" == *"pytorch_2_6_nxd_inference"* ]]; then
        echo -e "${GREEN}✓ Inference environment active${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Unknown environment: ${VIRTUAL_ENV}${NC}"
        return 1
    fi
}

# Initialize logging
init_logging() {
    mkdir -p "$NKI_LOGS"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$NKI_LOGS/nki-llama_${TIMESTAMP}.log"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2>&1
    echo -e "${BLUE}📝 Logging to: ${LOG_FILE}${NC}"
}

# Run script with error handling
run_script() {
    local script_path="$1"
    local display_name="$2"
    shift 2
    
    if [[ ! -f "$script_path" ]]; then
        echo -e "${RED}❌ Script not found: $script_path${NC}"
        return 1
    fi
    
    echo -e "${MAGENTA}▶ Running: ${display_name}${NC}"
    if bash "$script_path" "$@"; then
        echo -e "${GREEN}✓ ${display_name} completed${NC}\n"
    else
        echo -e "${RED}✗ ${display_name} failed${NC}\n"
        return 1
    fi
}

###############################################################################
# Fine-tuning Commands
###############################################################################

cmd_finetune_deps() {
    echo -e "${BOLD}Installing fine-tuning dependencies...${NC}"
    run_script "${NKI_FINETUNE_SCRIPTS}/bootstrap.sh" "Dependencies Installation"
}

cmd_finetune_data() {
    echo -e "${BOLD}Downloading dataset...${NC}"
    run_script "${NKI_FINETUNE_SCRIPTS}/download_data.sh" "Dataset Download"
}

cmd_finetune_model() {
    echo -e "${BOLD}Downloading model weights...${NC}"
    run_script "${NKI_FINETUNE_SCRIPTS}/download_model.sh" "Model Download"
}

cmd_finetune_convert() {
    echo -e "${BOLD}Converting checkpoints...${NC}"
    run_script "${NKI_FINETUNE_SCRIPTS}/convert_checkpoints.sh" "Checkpoint Conversion"
}

cmd_finetune_compile() {
    echo -e "${BOLD}Pre-compiling graphs...${NC}"
    
    # Check if we're in tmux
    if [[ -z "${TMUX:-}" ]]; then
        echo -e "${YELLOW}⚠️  Not running in tmux. ${BOLD}This is important for graph compilation!${NC}"
        echo -e "${YELLOW}   Graph compilation can take 30-60 minutes.${NC}"
        echo -e "${YELLOW}   Disconnections will terminate the process.${NC}"
        echo
        echo -e "   ${CYAN}tmux new -s compile${NC}"
        echo -e "   ${CYAN}./nki-llama finetune compile${NC}"
        echo
        read -p "Continue without tmux? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Please start tmux with: ${CYAN}tmux new -s compile${NC}"
            exit 0
        fi
    fi
    
    run_script "${NKI_FINETUNE_SCRIPTS}/precompile.sh" "Graph Compilation"
}

cmd_finetune_train() {
    echo -e "${BOLD}Starting fine-tuning...${NC}"
    
    # Show training information
    echo -e "${YELLOW}💡 Fine-tuning will run for multiple hours.${NC}"
    echo -e "${YELLOW}   The training includes checkpointing and will resume if interrupted.${NC}"
    echo -e "${YELLOW}   Using tmux is strongly recommended!${NC}"
    
    # Check if we're in tmux
    if [[ -z "${TMUX:-}" ]]; then
        echo -e "${YELLOW}⚠️  Not running in tmux. ${BOLD}This is critical for training!${NC}"
        echo -e "${YELLOW}   Training can take several hours to complete.${NC}"
        echo -e "${YELLOW}   Disconnections will terminate the process (SIGHUP).${NC}"
        echo
        echo -e "   ${CYAN}tmux new -s training${NC}"
        echo -e "   ${CYAN}./nki-llama finetune train${NC}"
        echo
        read -p "Continue without tmux? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Please start tmux with: ${CYAN}tmux new -s training${NC}"
            exit 0
        fi
    fi
    
    run_script "${NKI_FINETUNE_SCRIPTS}/run_training.sh" "Fine-tuning"
}

cmd_finetune_all() {
    echo -e "${BOLD}Running complete fine-tuning pipeline...${NC}\n"
    
    # Check if we're in tmux for the entire pipeline
    if [[ -z "${TMUX:-}" ]]; then
        echo -e "${YELLOW}⚠️  Not running in tmux. ${BOLD}This is critical for the full pipeline!${NC}"
        echo -e "${YELLOW}   The complete pipeline includes:${NC}"
        echo -e "${YELLOW}   • Dependency installation${NC}"
        echo -e "${YELLOW}   • Dataset download${NC}"
        echo -e "${YELLOW}   • Model download${NC}"
        echo -e "${YELLOW}   • Checkpoint conversion${NC}"
        echo -e "${YELLOW}   • Graph compilation (30-60 min)${NC}"
        echo -e "${YELLOW}   • Training (several hours)${NC}"
        echo -e "${YELLOW}   Total time: 4-8 hours depending on configuration${NC}"
        echo
        echo -e "   ${CYAN}tmux new -s training${NC}"
        echo -e "   ${CYAN}./nki-llama finetune all${NC}"
        echo
        read -p "Continue without tmux? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Please start tmux with: ${CYAN}tmux new -s training${NC}"
            exit 0
        fi
    fi
    
    cmd_finetune_deps && \
    cmd_finetune_data && \
    cmd_finetune_model && \
    cmd_finetune_convert && \
    cmd_finetune_compile && \
    cmd_finetune_train
}

###############################################################################
# Inference Commands
###############################################################################

cmd_inference_setup() {
    echo -e "${BOLD}Setting up vLLM for inference...${NC}"
    bash "${NKI_INFERENCE_SCRIPTS}/setup-vllm.sh"
}

cmd_inference_download() {
    echo -e "${BOLD}Downloading model for inference...${NC}"
    bash "${NKI_INFERENCE_SCRIPTS}/download-model.sh"
}

cmd_inference_benchmark() {
    echo -e "${BOLD}Running NKI benchmark evaluation...${NC}"
    
    # Parse benchmark mode and special flags
    local mode="evaluate_all"  # Default mode
    local args=()
    local clear_cache=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            single)
                mode="evaluate_single"
                shift
                ;;
            all)
                mode="evaluate_all"
                shift
                ;;
            --mode)
                mode="$2"
                shift 2
                ;;
            --clear-cache|clear-cache)
                clear_cache=true
                args+=("--clear-cache")
                shift
                ;;
            *)
                args+=("$1")
                shift
                ;;
        esac
    done
    
    # Show mode information
    echo -e "${YELLOW}💡 Running benchmark in ${CYAN}${mode}${YELLOW} mode${NC}"
    
    if [[ "$mode" == "evaluate_single" ]]; then
        echo -e "${YELLOW}   This runs a quick single evaluation from the repository test script.${NC}"
    else
        echo -e "${YELLOW}   This includes model compilation with NKI optimizations (10-30 min on first run).${NC}"
        echo -e "${YELLOW}   The compiled model will be cached for future use.${NC}"
        echo -e "${YELLOW}   ${CYAN}Auto cache recovery is enabled by default.${NC}"
    fi
    
    if [[ "$clear_cache" == "true" ]]; then
        echo -e "${YELLOW}   ${CYAN}Cache will be cleared before running.${NC}"
    fi
    
    echo -e "${YELLOW}   Using tmux is strongly recommended!${NC}"
    echo -e "${YELLOW}   Running: ${NKI_INFERENCE_SCRIPTS}/run-nki-benchmark.sh --mode $mode ${args[@]}"
    
    # Check if we're in tmux for evaluate_all mode
    if [[ "$mode" == "evaluate_all" ]] && [[ -z "${TMUX:-}" ]]; then
        echo -e "${YELLOW}⚠️  Not running in tmux. ${BOLD}This is critical for long compilations!${NC}"
        echo -e "${YELLOW}   Disconnections will terminate the process (SIGHUP).${NC}"
        echo
        echo -e "   ${CYAN}tmux new -s benchmark${NC}"
        echo -e "   ${CYAN}./nki-llama inference benchmark ${mode} ${args[*]}${NC}"
        echo
        read -p "Continue without tmux? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Please start tmux with: ${CYAN}tmux new -s benchmark${NC}"
            exit 0
        fi
    fi
    
    bash "${NKI_INFERENCE_SCRIPTS}/run-nki-benchmark.sh" --mode "$mode" "${args[@]}"
}

cmd_inference_server() {
    echo -e "${BOLD}Starting vLLM server...${NC}"
    suggest_tmux "vLLM Server" "vllm-server" "inference server"
    bash "${NKI_INFERENCE_SCRIPTS}/start-server.sh"
}

###############################################################################
# Utility Commands
###############################################################################

cmd_status() {
    echo -e "\n${BOLD}System Status:${NC}"
    check_neuron_env
    echo
    
    echo -e "${BOLD}Configuration:${NC}"
    print_config
    echo
    
    echo -e "${BOLD}Fine-tuning Status:${NC}"
    [[ -d "$DATASET_DIR" ]] && echo -e "• Dataset: ${GREEN}✓${NC}" || echo -e "• Dataset: ${YELLOW}⚠${NC}"
    [[ -d "$HF_WEIGHTS_DIR" ]] && echo -e "• Weights: ${GREEN}✓${NC}" || echo -e "• Weights: ${YELLOW}⚠${NC}"
    [[ -d "$PRETRAINED_CKPT" ]] && echo -e "• Checkpoints: ${GREEN}✓${NC}" || echo -e "• Checkpoints: ${YELLOW}⚠${NC}"
    [[ -d "$NEMO_EXPERIMENTS" ]] && echo -e "• Training: ${GREEN}✓${NC}" || echo -e "• Training: ${YELLOW}⚠${NC}"
    echo
    
    echo -e "${BOLD}Inference Status:${NC}"
    [[ -d "${NKI_MODELS}/${MODEL_NAME}" ]] && echo -e "• Model: ${GREEN}✓${NC}" || echo -e "• Model: ${YELLOW}⚠${NC}"
    [[ -d "${NKI_COMPILED}/${MODEL_NAME}" ]] && echo -e "• Compiled: ${GREEN}✓${NC}" || echo -e "• Compiled: ${YELLOW}⚠${NC}"
    [[ -d "$VLLM_REPO" ]] && echo -e "• vLLM: ${GREEN}✓${NC}" || echo -e "• vLLM: ${YELLOW}⚠${NC}"
    
    # Check compilation cache
    CACHE_DIR="/var/tmp/neuron-compile-cache"
    if [[ -d "$CACHE_DIR" ]]; then
        CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "unknown")
        echo -e "• Compile Cache: ${GREEN}✓${NC} (${CACHE_SIZE})"
        
        # Check for failed compilations
        FAILED_COUNT=$(find "$CACHE_DIR" -name "*.neff" -size 0 2>/dev/null | wc -l || echo "0")
        if [[ $FAILED_COUNT -gt 0 ]]; then
            echo -e "  ${YELLOW}⚠ ${FAILED_COUNT} failed compilation entries found${NC}"
            echo -e "  ${CYAN}Run: ./nki-llama inference benchmark --clear-cache${NC}"
        fi
    else
        echo -e "• Compile Cache: ${YELLOW}⚠${NC} (not found)"
    fi
    
    if command -v neuron-ls &> /dev/null; then
        echo -e "\n${BOLD}Neuron Hardware:${NC}"
        
        # Extract instance info
        INSTANCE_TYPE=$(neuron-ls | grep "instance-type:" | cut -d' ' -f2)
        echo -e "• Instance: ${CYAN}${INSTANCE_TYPE}${NC}"
        
        # Parse device information
        DEVICE_INFO=$(neuron-ls | grep -E "^\| [0-9]+ ")
        DEVICE_COUNT=$(echo "$DEVICE_INFO" | wc -l)
        
        if [[ $DEVICE_COUNT -gt 0 ]]; then
            # Calculate totals
            TOTAL_CORES=$(( DEVICE_COUNT * 2 ))
            TOTAL_MEMORY=$(( DEVICE_COUNT * 32 ))
            
            # Count busy devices - fixed version
            if echo "$DEVICE_INFO" | grep -q "python"; then
                BUSY_COUNT=$(echo "$DEVICE_INFO" | grep -c "python")
            else
                BUSY_COUNT=0
            fi
            FREE_COUNT=$(( DEVICE_COUNT - BUSY_COUNT ))
            
            echo -e "• Devices: ${GREEN}${DEVICE_COUNT}${NC} (${FREE_COUNT} free, ${BUSY_COUNT} busy)"
            echo -e "• Total: ${TOTAL_CORES} cores, ${TOTAL_MEMORY}GB memory"
            
            # Show runtime version if available
            RUNTIME_VERSION=$(neuron-ls | awk '/RUNTIME/ && /VERSION/ {getline; if (match($0, /[0-9]+\.[0-9]+\.[0-9]+/)) print substr($0, RSTART, RLENGTH)}' | head -1)
            if [[ -n "$RUNTIME_VERSION" ]]; then
                echo -e "• Runtime: v${RUNTIME_VERSION}"
            fi
        else
            echo -e "${YELLOW}⚠ No Neuron devices detected${NC}"
        fi
    else
        echo -e "\n${YELLOW}⚠ neuron-ls not found - Neuron SDK may not be installed${NC}"
    fi
}

cmd_clean() {
    echo -e "${YELLOW}🧹 Cleaning generated files...${NC}"
    
    # Show cache status first
    CACHE_DIR="/var/tmp/neuron-compile-cache"
    if [[ -d "$CACHE_DIR" ]]; then
        CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "unknown")
        echo -e "\nCompilation cache: ${CYAN}${CACHE_SIZE}${NC} at ${CACHE_DIR}"
        
        read -p "Clean compilation cache? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if rm -rf "$CACHE_DIR" 2>/dev/null; then
                echo -e "${GREEN}✓ Compilation cache cleaned${NC}"
            else
                echo -e "${RED}✗ Failed to clean cache. Try: sudo rm -rf ${CACHE_DIR}${NC}"
            fi
        fi
    fi
    
    read -p "Clean fine-tuning artifacts? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$DATASET_DIR" "$TOKENIZER_DIR" "$HF_WEIGHTS_DIR" "$PRETRAINED_CKPT" "$NEMO_EXPERIMENTS"
        echo -e "${GREEN}✓ Fine-tuning artifacts cleaned${NC}"
    fi
    
    read -p "Clean inference artifacts? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${NKI_COMPILED}/${MODEL_NAME}"
        echo -e "${GREEN}✓ Inference artifacts cleaned${NC}"
    fi
}

# Show help
show_help() {
    echo -e "\n${BOLD}NKI-LLAMA Unified Interface${NC}"
    echo -e "${CYAN}=========================${NC}\n"
    
    echo -e "${CYAN}Quick Commands:${NC}"
    echo -e "  ./nki-llama setup         - Initial setup guide"
    echo -e "  ./nki-llama train         - Start fine-tuning"
    echo -e "  ./nki-llama server        - Start inference server"
    echo -e "  ./nki-llama jupyter       - Start Jupyter Lab"
    echo
    
    echo -e "${CYAN}Fine-tuning Commands:${NC}"
    echo -e "  ./nki-llama finetune deps      - Install dependencies"
    echo -e "  ./nki-llama finetune data      - Download training dataset"
    echo -e "  ./nki-llama finetune model     - Download model weights"
    echo -e "  ./nki-llama finetune convert   - Convert checkpoints"
    echo -e "  ./nki-llama finetune compile   - Pre-compile graphs"
    echo -e "  ./nki-llama finetune train     - Start training"
    echo -e "  ./nki-llama finetune all       - Run complete pipeline"
    echo
    
    echo -e "${CYAN}Inference Commands:${NC}"
    echo -e "  ./nki-llama inference setup             - Setup vLLM"
    echo -e "  ./nki-llama inference download          - Download model"
    echo -e "  ./nki-llama inference benchmark         - Run full benchmark (evaluate_all)"
    echo -e "  ./nki-llama inference benchmark single  - Quick benchmark (evaluate_single)"
    echo -e "  ./nki-llama inference benchmark --clear-cache  - Clear cache & benchmark"
    echo -e "  ./nki-llama inference server            - Start API server"
    echo
    
    echo -e "${CYAN}Benchmark Options:${NC}"
    echo -e "  ${BOLD}Modes:${NC}"
    echo -e "    evaluate_single - Quick validation using repository test script"
    echo -e "    evaluate_all    - Full benchmark with NKI compilation & caching"
    echo
    echo -e "  ${BOLD}Cache Management:${NC}"
    echo -e "    --clear-cache              - Clear compilation cache before running"
    echo -e "    --no-auto-clear-cache      - Disable automatic cache recovery"
    echo -e "    --retry-failed-compilation - Force retry of failed compilations"
    echo
    echo -e "  ${BOLD}Examples:${NC}"
    echo -e "    ./nki-llama inference benchmark                     # Full benchmark"
    echo -e "    ./nki-llama inference benchmark single              # Quick test"
    echo -e "    ./nki-llama inference benchmark --clear-cache       # Clean run"
    echo -e "    ./nki-llama inference benchmark --seq-len 1024      # Custom seq length"
    echo
    
    echo -e "${CYAN}Utility Commands:${NC}"
    echo -e "  ./nki-llama status        - Show system status"
    echo -e "  ./nki-llama config        - Show configuration"
    echo -e "  ./nki-llama clean         - Clean artifacts & cache"
    echo -e "  ./nki-llama help          - Show this help"
    echo
    
    echo -e "${CYAN}Environment Setup:${NC}"
    echo -e "  Fine-tuning: source ${NEURON_VENV}/bin/activate"
    echo -e "  Inference:   source ${NEURON_INFERENCE_VENV}/bin/activate"
    echo
    
    echo -e "${CYAN}Troubleshooting:${NC}"
    echo -e "  • Always use tmux for long operations (compile, train, benchmark)"
    echo -e "  • If benchmark fails with cache errors, use --clear-cache"
    echo -e "  • Check status to see if compilation cache has failed entries"
    echo
}

# Setup wizard
cmd_setup() {
    echo -e "${BOLD}NKI-LLAMA Setup Wizard${NC}"
    echo -e "=====================\n"
    
    # Check for .env file
    if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
        echo -e "${YELLOW}No .env file found. Creating one...${NC}"
        cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env" 2>/dev/null || {
            echo -e "${RED}No .env.example found. Creating basic .env...${NC}"
            cat > "${SCRIPT_DIR}/.env" << EOF
# NKI-LLAMA Configuration
HF_TOKEN=
MODEL_ID=meta-llama/Meta-Llama-3-8B
MODEL_NAME=llama-3-8b
TENSOR_PARALLEL_SIZE=8
EOF
        }
        echo -e "${GREEN}✓ Created .env file${NC}"
        echo -e "${YELLOW}Please edit .env and add your HF_TOKEN${NC}\n"
    fi
    
    # Show current environment
    echo -e "${BOLD}Current Environment:${NC}"
    check_neuron_env || true
    echo
    
    # Show quick start
    echo -e "${BOLD}Quick Start Guide:${NC}"
    echo -e "1. Edit .env file with your Hugging Face token"
    echo -e "2. For fine-tuning:"
    echo -e "   ${CYAN}source ${NEURON_VENV}/bin/activate${NC}"
    echo -e "   ${CYAN}tmux new -s training  # ${YELLOW}IMPORTANT: Use tmux!${NC}"
    echo -e "   ${CYAN}./nki-llama finetune all${NC}"
    echo -e "3. For model benchmarking:"
    echo -e "   ${CYAN}source ${NEURON_INFERENCE_VENV}/bin/activate${NC}"
    echo -e "   ${CYAN}./nki-llama inference download${NC}"
    echo -e "   ${CYAN}tmux new -s benchmark  # ${YELLOW}IMPORTANT: Use tmux!${NC}"
    echo -e "   ${CYAN}./nki-llama inference benchmark       # Full benchmark${NC}"
    echo -e "   ${CYAN}./nki-llama inference benchmark single   # Quick test${NC}"
    echo -e "4. For inference serving:"
    echo -e "   ${CYAN}./nki-llama inference setup${NC}"
    echo -e "   ${CYAN}./nki-llama inference server${NC}"
    echo
    echo -e "${YELLOW}💡 Pro Tips:${NC}"
    echo -e "   • Always use tmux for long operations"
    echo -e "   • Check ./nki-llama status for system health"
    echo -e "   • Use --clear-cache if benchmark fails with cache errors"
    echo
}

# Main function
main() {
    # Show banner only for interactive commands
    case "${1:-help}" in
        help|setup|status|config)
            clear
            display_banner
            ;;
    esac
    
    # Initialize logging for actual operations
    case "${1:-help}" in
        finetune|inference|train|server|clean)
            init_logging
            ;;
    esac
    
    # Parse command
    local cmd="${1:-help}"
    shift || true
    
    case "$cmd" in
        # Setup
        setup)
            cmd_setup
            ;;
            
        # Quick shortcuts
        train)
            cmd_finetune_train "$@"
            ;;
        server)
            cmd_inference_server "$@"
            ;;
        jupyter)
            bash "${NKI_INFERENCE_SCRIPTS}/jupyter.sh" "$@"
            ;;
            
        # Fine-tuning commands
        finetune)
            subcmd="${1:-all}"
            shift || true
            case "$subcmd" in
                deps|data|model|convert|compile|train|all)
                    cmd_finetune_"$subcmd" "$@"
                    ;;
                *)
                    echo -e "${RED}Unknown finetune command: $subcmd${NC}"
                    show_help
                    ;;
            esac
            ;;
            
        # Inference commands
        inference)
            subcmd="${1:-server}"
            shift || true
            case "$subcmd" in
                setup|download|server|benchmark)
                    cmd_inference_"$subcmd" "$@"
                    ;;
                *)
                    echo -e "${RED}Unknown inference command: $subcmd${NC}"
                    show_help
                    ;;
            esac
            ;;
            
        # Utility commands
        status)
            cmd_status
            ;;
        config)
            print_config
            ;;
        clean)
            cmd_clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: $cmd${NC}"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
