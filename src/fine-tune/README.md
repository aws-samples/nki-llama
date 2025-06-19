You're able to fun through the fine-tuning workflow with the top-level NKI LLAMA CLI or this fine-tune pipeline script.

## 1 · Prerequisites

- Under the configs dir, make sure to bring your model specific config.json for NxD-training.
- additionally ensure you bring your training config.YAML file, for more details, refer to the corresponding NXDT documentation.

| Requirement | Reason | Install / Notes |
|-------------|--------|-----------------|
| **Neuron virtual‑env** | Script refuses to run outside it | `source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate` |
| **`scripts/` folder** | step-by step scripts for running fine tuning | |
| **`.env` file** *(optional)* | Central place for env vars | Place at `../../.env` |

Example `.env`:

```dotenv
HF_TOKEN=hf_****************************************
MODEL_ID=meta-llama-3-8b
```

---

## 2 · Setup

```bash
# Clone repo and enter it
cd ./src/fine-tune

# Make the script executable
chmod +x fine-tune-pipeline.sh

# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
```

---

## 3 · Using tmux for Long-Running Training

### Why tmux?

Training neural networks on AWS Neuron can take hours or even days. Using tmux ensures your training continues even if:
- Your SSH connection drops
- You need to close your laptop
- Network interruptions occur
- You want to monitor progress from multiple devices

### Quick tmux Setup

```bash
# Create a new tmux session named "training"
tmux new -s training

# Inside tmux, activate Neuron environment and start training
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
cd ./src/fine-tune
./pipeline.sh train

# Detach from session (training continues in background)
# Press: Ctrl+b, then d

# Later, reattach to check progress
tmux attach -t training

# List all sessions
tmux ls
```

### Essential tmux Commands

| Command | Action |
|---------|--------|
| `tmux new -s training` | Create session named "training" |
| `Ctrl+b d` | Detach (leave session running) |
| `tmux attach -t training` | Reattach to training session |
| `tmux ls` | List all sessions |
| `tmux kill-session -t training` | Terminate session |

**Pro tip:** Start your training in tmux from the beginning. It's much safer than hoping your connection stays stable!

---

## 4 · Usage

| Command | Action |
|---------|--------|
| `./fine-tune-pipeline.sh` | Run the **full FT pipeline** (deps → data → model → convert_ckpt → precompile → train) |
| `./fine-tune-pipeline.sh deps` | Install/validate Apex, NxDT, etc. |
| `./fine-tune-pipeline.sh data` | Download dataset |
| `./fine-tune-pipeline.sh model` | Download & convert model checkpoints |
| `./fine-tune-pipeline.sh convert_ckpt` | Convert checkpoints to NxDT format |
| `./fine-tune-pipeline.sh precompile` | Ahead‑of‑time graph compilation |
| `./fine-tune-pipeline.sh train` | Start fine‑tuning |
| `./fine-tune-pipeline.sh clean` | Remove generated datasets, weights, experiments |

 Each sub‑command double‑checks you're inside a Neuron venv and prints a helpful error if not.

---

## 5 · Environment Variables

| Variable | Purpose | How to set |
|----------|---------|-----------|
| `HF_TOKEN` | Hugging Face auth token (for private models) | Add to `.env` or `export HF_TOKEN=…` |
| `MODEL_ID` | Model slug, e.g. `meta-llama-3-8b` | Same as above |

The script auto‑loads `../../.env` with `set -a; source …`. Modify the `ENV_FILE=` line in `fine-tune-pipeline.sh` if you store it elsewhere.

---

## 6 · Troubleshooting

| Symptom | Probable Cause | Fix |
|---------|---------------|-----|
| `Not inside a Neuron virtual environment.` | Forgot to activate venv | `source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate` |
| `command not found: fine-tune-pipeline.sh` | File not executable or wrong cwd | `chmod +x fine-tune-pipeline.sh` and/or `./fine-tune-pipeline.sh` |
| Model download fails | Missing/invalid `HF_TOKEN` | Provide valid token in env or `.env` |
| Long compile times | First‑time Neuron AOT | Subsequent runs reuse cached graphs |

---

## 7 · Extending the Pipeline

1. Add a new Bash function in `fine-tune-pipeline.sh` (e.g., `evaluate()`).
2. Append its name to the pattern list inside `main()`.
3. Optionally call it from `all()` for automatic inclusion.

```bash
./fine-tune-pipeline.sh train 
```
---