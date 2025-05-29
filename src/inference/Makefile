# Makefile for nki-llama with vLLM and Jupyter support

# Include .env file if it exists
-include ../../.env

# Variables
PYTHON = python
REPOSITORY_PATH = ~/nki-llama
MAIN_SCRIPT = main.py
TEST_SCRIPT = test/inference/test.py
LOCAL_VENV = ../../venv/bin/activate
MODELS_DIR = ~/models
COMPILED_MODEL_DIR = ~/traced_model
REPO_PATH = $(HOME)/upstreaming-to-vllm
REPO_URL = https://github.com/aws-neuron/upstreaming-to-vllm.git
REPO_BRANCH = neuron-2.22-vllm-v0.7.2
PORT ?= 8080
MAX_MODEL_LEN ?= 2048
TENSOR_PARALLEL_SIZE ?= 8

# Default target
.PHONY: all
all: help

# Help message
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup-jupyter  - Create Python virtual environment, install requirements and setup Jupyter (must be run with source)"
	@echo "  setup-vllm     - Setup vLLM for Neuron (requires Neuron environment first)"
	@echo "  download       - Download model from Hugging Face (requires Neuron environment first)"
	@echo "  infer          - Run inference in generate mode (requires Neuron environment first)"
	@echo "  evaluate       - Run inference in evaluate_single mode (requires Neuron environment first)"
	@echo "  start-server   - Start vLLM OpenAI-compatible API server"
	@echo "  jupyter        - Run Jupyter Lab server"
	@echo "  show-env       - Display environment variables loaded from .env file"
	@echo "  clean          - Remove generated files"
	@echo ""
	@echo "Note: Before running most commands, activate the Neuron environment with:"
	@echo "  source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate"
	@echo ""
	@echo "For Jupyter, activate the local environment with:"
	@echo "  source venv/bin/activate"
	@echo ""
	@echo "You can set model and server parameters in a .env file"
	@echo ""
	@echo "Example workflow:"
	@echo "  1. source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate"
	@echo "  2. make setup-vllm"
	@echo "  3. make download"
	@echo "  4. make start-server"
	@echo "  5. In a new terminal: source venv/bin/activate"
	@echo "  6. make setup-jupyter"
	@echo "  7. make jupyter"

.PHONY: show-env
show-env:
	@echo "Environment variables loaded from .env file:"
	@echo "----------------------------------------"
	@echo "MODEL_ID: $(MODEL_ID)"
	@echo "MODEL_NAME: $(MODEL_NAME)"
	@echo "MODELS_DIR: $(MODELS_DIR)"
	@echo "COMPILED_MODEL_DIR: $(COMPILED_MODEL_DIR)"
	@echo "PORT: $(PORT)"
	@echo "MAX_MODEL_LEN: $(MAX_MODEL_LEN)"
	@echo "TENSOR_PARALLEL_SIZE: $(TENSOR_PARALLEL_SIZE)"
	@echo "HF_TOKEN: $${HF_TOKEN:-(not set)}"
	@echo "----------------------------------------"

# Setup local virtual environment and Jupyter
.PHONY: setup-jupyter
setup-jupyter:
	test -d venv || $(PYTHON) -m venv venv
	pip install --upgrade pip
	test -f requirements.txt && pip install -r requirements.txt || echo "No requirements.txt found"
	pip install langchain langgraph langchain_community ipykernel jupyter jupyterlab python-dotenv
	$(PYTHON) -m ipykernel install --user --name="neuron_agents" --display-name="Python (Neuron Agents)"
	@echo "Virtual environment and Jupyter kernel setup complete"

# Check if in Neuron virtual environment
.PHONY: check-neuron-venv
check-neuron-venv:
	@if [ -z "$$VIRTUAL_ENV" ] || [[ "$$VIRTUAL_ENV" != *"neuronx"* ]]; then \
		echo "Error: Not in Neuron virtual environment."; \
		echo "Run 'source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate' first."; \
		exit 1; \
	else \
		echo "Using Neuron virtual environment: $$VIRTUAL_ENV"; \
	fi

# Check if in local virtual environment
.PHONY: check-local-venv
check-local-venv:
	@if [ -z "$$VIRTUAL_ENV" ] || [[ "$$VIRTUAL_ENV" != *"venv"* ]]; then \
		echo "Error: Not in local virtual environment."; \
		echo "Run 'source venv/bin/activate' first."; \
		exit 1; \
	else \
		echo "Using local virtual environment: $$VIRTUAL_ENV"; \
	fi

# Check if vLLM is installed
.PHONY: check-vllm
check-vllm:
	@if ! python -c "import vllm" 2>/dev/null; then \
		echo "Error: vLLM not installed. Run 'make setup-vllm' first."; \
		exit 1; \
	else \
		echo "vLLM is installed"; \
	fi

# Check or set MODEL_NAME environment variable
.PHONY: check-model-name
check-model-name:
	@if [ -z "$(MODEL_NAME)" ]; then \
		echo "MODEL_NAME environment variable is not set"; \
		echo "You can set it permanently in .env file or with: export MODEL_NAME=your_model_path"; \
		read -p "Enter model name for this session (default: llama-3.2-3b-instruct): " model_name; \
		if [ -z "$$model_name" ]; then \
			export MODEL_NAME="llama-3.2-3b-instruct"; \
		else \
			export MODEL_NAME="$$model_name"; \
		fi; \
		echo "Using model: $$MODEL_NAME"; \
	else \
		echo "Using model from configuration: $(MODEL_NAME)"; \
	fi

# Setup vLLM for Neuron
.PHONY: setup-vllm
setup-vllm: check-neuron-venv
	@if [ -d "$(REPO_PATH)" ]; then \
		echo "Repository already exists at $(REPO_PATH)"; \
		cd "$(REPO_PATH)"; \
		CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
		if [ "$$CURRENT_BRANCH" != "$(REPO_BRANCH)" ]; then \
			echo "Switching to branch $(REPO_BRANCH)..."; \
			git checkout "$(REPO_BRANCH)" || git fetch && git checkout "$(REPO_BRANCH)"; \
		else \
			echo "Already on correct branch: $(REPO_BRANCH)"; \
		fi; \
	else \
		echo "Cloning repository $(REPO_URL) with branch $(REPO_BRANCH)..."; \
		cd "$(HOME)"; \
		git clone -b "$(REPO_BRANCH)" "$(REPO_URL)"; \
	fi
	@cd "$(REPO_PATH)" && \
	echo "Installing vLLM requirements..." && \
	pip install -r requirements-neuron.txt && \
	echo "Installing vLLM for Neuron..." && \
	VLLM_TARGET_DEVICE="neuron" pip install -e .
	@echo "vLLM setup complete"

# Download model from Hugging Face
.PHONY: download
download: check-neuron-venv
	@echo "Downloading model from Hugging Face"
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "MODEL_ID not set in .env file."; \
		read -p "Enter HuggingFace model ID (e.g., meta-llama/Meta-Llama-3-8B): " MODEL_ID_INPUT; \
		if [ -z "$$MODEL_ID_INPUT" ]; then \
			echo "Error: MODEL_ID is required"; \
			exit 1; \
		fi; \
		MODEL_ID_VAR="$$MODEL_ID_INPUT"; \
	else \
		MODEL_ID_VAR="$(MODEL_ID)"; \
		echo "Using MODEL_ID from .env: $$MODEL_ID_VAR"; \
	fi; \
	if [ -z "$(HF_TOKEN)" ]; then \
		echo "HF_TOKEN not set in .env file."; \
		echo "Get one at: https://huggingface.co/docs/hub/en/security-tokens"; \
		read -p "Enter your Hugging Face token: " TOKEN; \
		if [ -z "$$TOKEN" ]; then \
			echo "Error: HF_TOKEN is required"; \
			exit 1; \
		fi; \
	else \
		TOKEN="$(HF_TOKEN)"; \
		echo "Using HF_TOKEN from .env file"; \
	fi; \
	SHORTNAME=$$(echo $$MODEL_ID_VAR | sed 's/.*\///' | tr '[:upper:]' '[:lower:]'); \
	echo "Downloading $$MODEL_ID_VAR to $(MODELS_DIR)/$$SHORTNAME"; \
	mkdir -p $(MODELS_DIR); \
	pip install -q huggingface_hub[cli]; \
	huggingface-cli download --token $$TOKEN $$MODEL_ID_VAR --local-dir $(MODELS_DIR)/$$SHORTNAME; \
	echo ""; \
	echo "Download complete!"; \
	echo "Model saved to: $(MODELS_DIR)/$$SHORTNAME"; \
	if [ -z "$(MODEL_ID)" ]; then \
		echo ""; \
		echo "Add to your .env file:"; \
		echo "MODEL_ID=$$MODEL_ID_VAR"; \
		echo "MODEL_NAME=$$SHORTNAME"; \
	fi

# Start vLLM OpenAI-compatible API server
.PHONY: start-server
start-server: check-neuron-venv check-vllm check-model-name
	@echo "Starting vLLM OpenAI-compatible API server with model: $(MODEL_NAME)"
	cd ~ && VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference' NEURON_COMPILED_ARTIFACTS='$(COMPILED_MODEL_DIR)/$(MODEL_NAME)' python -m vllm.entrypoints.openai.api_server \
		--model="models/$(MODEL_NAME)" \
		--max-num-seqs=4 \
		--max-model-len=$(MAX_MODEL_LEN) \
		--tensor-parallel-size=$(TENSOR_PARALLEL_SIZE) \
		--port=$(PORT) \
		--device "neuron" \
		--override-neuron-config "{\"enable_bucketing\":false}"

# Run Jupyter Lab
.PHONY: jupyter
jupyter: check-local-venv
	@echo "Starting Jupyter Lab server..."
	jupyter lab --no-browser --ip="0.0.0.0"

# Run inference in generate mode
.PHONY: infer
infer: check-neuron-venv check-model-name
	@echo "Running inference in generate mode with model: $(MODEL_NAME)"
	$(PYTHON) $(MAIN_SCRIPT) --mode generate --model-path "$(MODELS_DIR)/$(MODEL_NAME)" --compiled-model-path "$(COMPILED_MODEL_DIR)/$(MODEL_NAME)" --enable-nki --seq-len 640 --tp-degree $(TENSOR_PARALLEL_SIZE)

# Run inference in evaluate_single mode
.PHONY: evaluate
evaluate: check-neuron-venv check-model-name
	@echo "Running inference in evaluate_single mode with model: $(MODEL_NAME)"
	cd ~ && $(PYTHON) $(REPOSITORY_PATH)/$(TEST_SCRIPT) --repository-path $(REPOSITORY_PATH)

# Run inference in evaluate_all mode
.PHONY: evaluate-all
evaluate-all: check-neuron-venv check-model-name
	@echo "Running inference in evaluate_all mode with model: $(MODEL_NAME)"
	$(PYTHON) $(MAIN_SCRIPT) --mode evaluate_all --model-path "$(MODELS_DIR)/$(MODEL_NAME)" --compiled-model-path "$(COMPILED_MODEL_DIR)/$(MODEL_NAME)" --enable-nki --seq-len 640 --tp-degree $(TENSOR_PARALLEL_SIZE)

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -rf test/inference/output/*
	find . -type d -name "__pycache__" -exec rm -rf {} +