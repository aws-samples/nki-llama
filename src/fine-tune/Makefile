# examples/nxdt_finetune_llama3_8B_lora/Makefile

-include ../../.env

SHELL := /bin/bash

.PHONY: all deps data model convert_ckpt precompile train clean
all: deps data model convert_ckpt precompile train
# Check if in Neuron virtual environment
.PHONY: check-neuron-venv
check-neuron-venv:
	@if [ -z "$$VIRTUAL_ENV" ] || [[ "$$VIRTUAL_ENV" != *"neuronx"* ]]; then \
		echo "Error: Not in Neuron virtual environment."; \
		echo "Run 'source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate' first."; \
		exit 1; \
	else \
		echo "Using Neuron virtual environment: $$VIRTUAL_ENV"; \
	fi

deps: check-neuron-venv
	@echo "==> Installing/Validating Apex, NxDT and other dependencies..."
	@echo "Running bootstrap script to ensure all dependencies are properly configured..."
	bash scripts/bootstrap.sh

data: check-neuron-venv
	@echo "==> Downloading dataset"
	bash scripts/download_data.sh

model: check-neuron-venv
	@echo "==> Downloading & converting model"
	HF_TOKEN="${HF_TOKEN}" MODEL_ID="${MODEL_ID}" bash scripts/download_model.sh

convert_ckpt: check-neuron-venv
	@echo "==> Converting checkpoint to NxDT format"
	bash scripts/convert_checkpoints.sh

precompile: check-neuron-venv
	@echo "==> Pre-compiling graphs (AOT)"
	bash scripts/precompile.sh

train: check-neuron-venv
	@echo "==> Running fine-tuning job"
	bash scripts/run_training.sh

clean: check-neuron-venv
	rm -rf dataset llama3_tokenizer llama3-8B_hf_weights pretrained_ckpt nemo_experiments
	@echo "Cleaned up all generated files."


