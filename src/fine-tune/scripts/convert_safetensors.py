#!/usr/bin/env python3
import os, glob, argparse
import torch
from safetensors.torch import load_file

def convert(input_dir, output_dir):
    for path in glob.glob(os.path.join(input_dir, "*.safetensors")):
        sd = load_file(path)
        out = os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0] + ".bin")
        torch.save(sd, out)
        print(f"Converted {path} → {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    convert(args.input_dir, args.output_dir)
