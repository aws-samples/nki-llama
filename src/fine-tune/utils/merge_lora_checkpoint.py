#!/usr/bin/env python3
"""
Script to merge LoRA weights from NeuronX checkpoint with base model
and convert to standard HuggingFace format
"""

import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from collections import defaultdict

def load_neuronx_lora_checkpoint(checkpoint_dir, tp_size=32):
    """Load and merge LoRA weights from distributed NeuronX checkpoint"""
    
    # Add XLA safe globals
    try:
        import torch_xla.utils.serialization
        torch.serialization.add_safe_globals([torch_xla.utils.serialization.TensorReference])
    except ImportError:
        print("Warning: torch_xla not available")
    
    print(f"Loading distributed checkpoint from {checkpoint_dir}")
    
    # Collect all checkpoint files and their tensor files
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pt') and not f.endswith('.tensors') and not f.endswith('.info.pt'):
            checkpoint_files.append(f)
    
    checkpoint_files.sort()
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Load all checkpoints and merge weights
    merged_state = {}
    lora_config = None
    
    for i, filename in enumerate(checkpoint_files):
        print(f"Loading {filename} ({i+1}/{len(checkpoint_files)})")
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        tensor_path = os.path.join(checkpoint_dir, filename + '.tensors')
        
        try:
            # Load checkpoint metadata
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract LoRA config from first checkpoint
            if lora_config is None and 'lora_config' in checkpoint:
                lora_config = checkpoint['lora_config']
                print(f"Found LoRA config: {lora_config}")
            
            # Load actual tensor data
            if os.path.exists(tensor_path):
                print(f"  Loading tensors from {filename}.tensors")
                tensor_data = torch.load(tensor_path, map_location='cpu', weights_only=False)
                
                # Merge the checkpoint metadata with tensor data
                for key in checkpoint.keys():
                    if key == 'lora_config':
                        continue
                    
                    # Get the actual tensor
                    if key in tensor_data:
                        tensor = tensor_data[key]
                    elif hasattr(checkpoint[key], 'materialize'):
                        tensor = checkpoint[key].materialize()
                    else:
                        # Fallback: try to use the checkpoint data directly
                        tensor = checkpoint[key]
                        if str(type(tensor)) == "<class 'torch_xla.utils.serialization.TensorReference'>":
                            print(f"Warning: Could not load tensor for {key}, skipping...")
                            continue
                    
                    if key not in merged_state:
                        merged_state[key] = []
                    merged_state[key].append(tensor)
            else:
                print(f"Warning: No tensor file found for {filename}, trying direct loading...")
                # Try to load tensors directly from checkpoint
                for key, tensor_ref in checkpoint.items():
                    if key == 'lora_config':
                        continue
                    
                    if hasattr(tensor_ref, 'materialize'):
                        tensor = tensor_ref.materialize()
                    elif torch.is_tensor(tensor_ref):
                        tensor = tensor_ref
                    else:
                        print(f"Warning: Could not process tensor for {key}, skipping...")
                        continue
                    
                    if key not in merged_state:
                        merged_state[key] = []
                    merged_state[key].append(tensor)
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    # Concatenate tensors across TP ranks
    print("Merging tensors across tensor parallel ranks...")
    final_state = {}
    for key, tensor_list in merged_state.items():
        if len(tensor_list) == 1:
            final_state[key] = tensor_list[0]
        else:
            try:
                # Determine concatenation dimension based on weight type
                if any(x in key for x in ['weight_q', 'weight_k', 'weight_v']):
                    if 'lora_B' in key:
                        # LoRA B matrices: concatenate along output dimension (dim=0)
                        final_state[key] = torch.cat(tensor_list, dim=0)
                    elif 'lora_A' in key:
                        # LoRA A matrices: concatenate along input dimension (dim=1)
                        final_state[key] = torch.cat(tensor_list, dim=1)
                    elif 'base_layer' in key:
                        # Base layer weights: concatenate along output dimension (dim=0)
                        final_state[key] = torch.cat(tensor_list, dim=0)
                    else:
                        final_state[key] = torch.cat(tensor_list, dim=0)
                elif 'gate_up_proj' in key:
                    # Gate-up projection: concatenate along output dimension
                    final_state[key] = torch.cat(tensor_list, dim=0)
                elif 'o_proj' in key:
                    # Output projection: concatenate along input dimension
                    final_state[key] = torch.cat(tensor_list, dim=1)
                elif 'down_proj' in key:
                    # Down projection: concatenate along input dimension
                    final_state[key] = torch.cat(tensor_list, dim=1)
                else:
                    # For other weights (embeddings, norms), take the first one (should be identical)
                    final_state[key] = tensor_list[0]
                    
                print(f"  Merged {key}: {[t.shape for t in tensor_list]} -> {final_state[key].shape}")
                
            except Exception as e:
                print(f"Error merging {key}: {e}")
                # Fallback: take the first tensor
                final_state[key] = tensor_list[0]
    
    return final_state, lora_config

def merge_lora_weights(base_weights, lora_weights, lora_config):
    """Merge LoRA weights with base weights"""
    
    print("Starting LoRA weight merging...")
    merged_weights = {}
    
    # Copy all base weights first
    for key, weight in base_weights.items():
        merged_weights[key] = weight.clone()
    
    # Extract LoRA parameters
    alpha = lora_config.get('lora_alpha', 32)
    r = lora_config.get('r', 16)
    scaling = alpha / r
    
    print(f"LoRA parameters: alpha={alpha}, rank={r}, scaling={scaling}")
    
    # Group LoRA weights by layer and component
    lora_layers = defaultdict(dict)
    for key in lora_weights:
        if 'lora_A' in key or 'lora_B' in key:
            # Parse key: model.layers.X.self_attn.qkv_proj.lora_A.weight
            parts = key.split('.')
            layer_idx = parts[2]
            component = parts[4]  # qkv_proj
            lora_type = parts[5]  # lora_A or lora_B
            
            if len(parts) > 6:
                weight_type = parts[6]  # weight_q, weight_k, weight_v
            else:
                weight_type = 'weight'
            
            layer_key = f"layers.{layer_idx}.self_attn.{component}"
            lora_key = f"{lora_type}.{weight_type}"
            lora_layers[layer_key][lora_key] = lora_weights[key]
    
    # Apply LoRA merging for each layer
    for layer_key, lora_params in lora_layers.items():
        print(f"Processing {layer_key}")
        
        # Handle QKV projections - merge each Q, K, V separately
        if 'qkv_proj' in layer_key:
            layer_num = layer_key.split('.')[1]
            
            for weight_type in ['q', 'k', 'v']:
                # Find corresponding base layer weight
                base_key = f"model.layers.{layer_num}.self_attn.qkv_proj.base_layer.weight_{weight_type}"
                lora_a_key = f"lora_A.weight"
                lora_b_key = f"lora_B.weight_{weight_type}"
                
                if (base_key in lora_weights and 
                    lora_a_key in lora_params and 
                    lora_b_key in lora_params):
                    
                    base_weight = lora_weights[base_key]  # Use the base weight from checkpoint
                    lora_a = lora_params[lora_a_key]
                    lora_b = lora_params[lora_b_key]
                    
                    print(f"  Merging {weight_type} projection:")
                    print(f"    Base: {base_weight.shape}")
                    print(f"    LoRA A: {lora_a.shape}")
                    print(f"    LoRA B: {lora_b.shape}")
                    
                    # Apply LoRA: W = W_base + scaling * (B @ A)
                    try:
                        delta = scaling * torch.mm(lora_b, lora_a)
                        merged_weight = base_weight + delta
                        
                        # Create standard HF key name
                        standard_key = f"model.layers.{layer_num}.self_attn.{weight_type}_proj.weight"
                        merged_weights[standard_key] = merged_weight
                        
                        print(f"    Merged -> {standard_key}: {merged_weight.shape}")
                        
                    except Exception as e:
                        print(f"    Error merging {weight_type}: {e}")
                        # Fallback: use base weight only
                        standard_key = f"model.layers.{layer_num}.self_attn.{weight_type}_proj.weight"
                        merged_weights[standard_key] = base_weight
    
    # Add non-LoRA weights from the checkpoint
    for key, weight in lora_weights.items():
        # Skip LoRA-specific keys
        if any(x in key for x in ['lora_A', 'lora_B', 'base_layer']):
            continue
        
        # Add standard weights (embeddings, layer norms, MLPs, etc.)
        merged_weights[key] = weight
    
    print(f"Final merged model has {len(merged_weights)} parameters")
    return merged_weights

def main():
    parser = argparse.ArgumentParser(description='Merge LoRA checkpoint with base model')
    parser.add_argument('--checkpoint_dir', required=True, help='Path to NeuronX checkpoint directory')
    parser.add_argument('--base_model_path', required=True, help='Path to base model')
    parser.add_argument('--output_dir', required=True, help='Output directory for merged model')
    parser.add_argument('--tp_size', type=int, default=32, help='Tensor parallel size')
    
    args = parser.parse_args()
    
    print("Step 1: Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    print("Step 2: Loading NeuronX LoRA checkpoint...")
    lora_weights, lora_config = load_neuronx_lora_checkpoint(args.checkpoint_dir, args.tp_size)
    
    print("Step 3: Merging LoRA weights...")
    base_state_dict = base_model.state_dict()
    merged_weights = merge_lora_weights(base_state_dict, lora_weights, lora_config)
    
    print("Step 4: Loading merged weights into model...")
    # Filter merged weights to only include keys that exist in the base model
    filtered_weights = {}
    for key in base_model.state_dict().keys():
        if key in merged_weights:
            filtered_weights[key] = merged_weights[key]
        else:
            print(f"Warning: {key} not found in merged weights, keeping original")
            filtered_weights[key] = base_model.state_dict()[key]
    
    base_model.load_state_dict(filtered_weights, strict=True)
    
    print("Step 5: Saving merged model...")
    os.makedirs(args.output_dir, exist_ok=True)
    base_model.save_pretrained(args.output_dir, safe_serialization=True)
    
    # Copy tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        tokenizer.save_pretrained(args.output_dir)
        print("Tokenizer copied successfully")
    except Exception as e:
        print(f"Warning: Could not copy tokenizer: {e}")
    
    print(f"Merged model saved to {args.output_dir}")
    
    # Print some statistics
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Total parameters in merged model: {total_params:,}")

if __name__ == "__main__":
    main()