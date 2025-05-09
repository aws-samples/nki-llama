# Top-level Makefile for coordinating fine-tuning and inference

-include .env

SHELL := /bin/bash

# Define paths to subproject directories
FINETUNE_DIR = ./src/fine-tune
INFERENCE_DIR = ./src/inference

# Default target
.PHONY: all
all: help

# Help message
.PHONY: help
help:
    @echo "Top-level Makefile for managing fine-tuning and inference"
    @echo ""
    @echo "Available targets:"
    @echo "  help                - Show this help message"
    @echo ""
    @echo "  finetune            - Run all fine-tuning steps"
    @echo "  finetune-deps       - Install fine-tuning dependencies"
    @echo "  finetune-data       - Download datasets for fine-tuning"
    @echo "  finetune-model      - Download model for fine-tuning"
    @echo "  finetune-convert    - Convert checkpoint to NxDT format"
    @echo "  finetune-precompile - Pre-compile graphs (AOT)"
    @echo "  finetune-train      - Run fine-tuning job"
    @echo "  finetune-clean      - Clean up fine-tuning files"
    @echo ""
    @echo "  inference           - Run inference (shortcut to infer target)"
    @echo "  inference-show-env  - Display environment variables loaded from .env file"
    @echo "  inference-setup     - Setup vLLM for Neuron"
    @echo "  inference-jupyter   - Setup Jupyter environment"
    @echo "  inference-download  - Download model from Hugging Face"
    @echo "  inference-infer     - Run inference in generate mode"
    @echo "  inference-evaluate  - Run inference in evaluate mode"
    @echo "  inference-server    - Start vLLM OpenAI-compatible API server"
    @echo "  inference-lab       - Run Jupyter Lab server"
    @echo "  inference-clean     - Clean up inference files"
    @echo ""
    @echo "  clean               - Clean up all generated files"
    @echo ""
    @echo "Environment requirements:"
    @echo "  - For inference: source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate"
    @echo "  - For fine-tuning: source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate"
    @echo "  - For Jupyter: source venv/bin/activate"

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

# Fine-tuning targets
.PHONY: finetune finetune-deps finetune-data finetune-model finetune-convert finetune-precompile finetune-train finetune-clean

finetune:
    $(MAKE) -C $(FINETUNE_DIR)

finetune-deps:
    $(MAKE) -C $(FINETUNE_DIR) deps

finetune-data:
    $(MAKE) -C $(FINETUNE_DIR) data

finetune-model:
    $(MAKE) -C $(FINETUNE_DIR) model

finetune-convert:
    $(MAKE) -C $(FINETUNE_DIR) convert_ckpt

finetune-precompile:
    $(MAKE) -C $(FINETUNE_DIR) precompile

finetune-train:
    $(MAKE) -C $(FINETUNE_DIR) train

finetune-clean:
    $(MAKE) -C $(FINETUNE_DIR) clean

# Inference targets
.PHONY: inference inference-setup inference-jupyter inference-download inference-infer inference-evaluate inference-server inference-lab inference-clean inference-show-env

inference:
    $(MAKE) -C $(INFERENCE_DIR) infer

inference-show-env:
    $(MAKE) -C $(INFERENCE_DIR) show-env

inference-setup:
    $(MAKE) -C $(INFERENCE_DIR) setup-vllm

inference-jupyter:
    $(MAKE) -C $(INFERENCE_DIR) setup-jupyter

inference-download:
    $(MAKE) -C $(INFERENCE_DIR) download

inference-infer:
    $(MAKE) -C $(INFERENCE_DIR) infer

inference-evaluate:
    $(MAKE) -C $(INFERENCE_DIR) evaluate

inference-server:
    $(MAKE) -C $(INFERENCE_DIR) start-server

inference-lab:
    $(MAKE) -C $(INFERENCE_DIR) jupyter

inference-clean:
    $(MAKE) -C $(INFERENCE_DIR) clean

# Clean all
.PHONY: clean
clean: finetune-clean inference-clean
    @echo "Cleaned all subprojects"