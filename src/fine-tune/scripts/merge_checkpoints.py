#!/usr/bin/env python3
import os
import torch

# Directory containing shards, and final output path
bin_dir    = "/home/ubuntu/nki-llama/src/fine-tune/model_assets/llama3-8B_hf_weights_bin"
output_bin = "/home/ubuntu/nki-llama/src/fine-tune/model_assets/pckpt/pytorch_model.bin"

# We’ll merge everything into this dict
merged_state = {}

# Always load on CPU for merging
map_loc = "cpu"

# Iterate in sorted order so names don’t collide unpredictably
for fname in sorted(os.listdir(bin_dir)):
    if not fname.endswith(".bin"):
        continue

    shard_path = os.path.join(bin_dir, fname)
    print(f"Loading shard: {shard_path}")
    shard = torch.load(shard_path, map_location=map_loc)

    # If the file wrapped weights under "state_dict", pull it out
    if isinstance(shard, dict) and "state_dict" in shard:
        shard = shard["state_dict"]

    # Merge into our master dict
    merged_state.update(shard)

# Save the flattened checkpoint
print(f"Saving merged checkpoint to {output_bin}")
torch.save(merged_state, output_bin)
print("Done.")
