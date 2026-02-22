# PipelineParallelism

A project exploring **Pipeline Parallelism** in PyTorch — starting from a monolithic single-GPU baseline and progressively sharding it across multiple GPUs.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the monolithic baseline
uv run python src/monolith.py
```

## Project Structure

```text
src/
└── monolith.py   # Single-GPU baseline (starting point)
```

## `src/monolith.py` — The Monolithic Baseline

This is the **ground truth** model that runs entirely on a single GPU/CPU. The goal of this project is to take this monolith and shard it across multiple GPUs using pipeline parallelism.

### Model: `MonolithicMLP`

A deep Multi-Layer Perceptron (MLP) built with PyTorch:

| Parameter | Value |
|---|---|
| Batch Size | `32` |
| Hidden Dimension | `128` |
| Total Layers | `16` (Linear + ReLU pairs) |
| Output Classes | `2` (binary classification) |
| Loss Function | `CrossEntropyLoss` |
| Optimizer | `Adam (lr=0.001)` |
| Training Steps | `50` |

### Architecture

```
Input (32, 128)
    → [Linear(128→128) + ReLU] × 16
    → Linear(128→2)
    → CrossEntropyLoss
```

### Training

The model trains on a **fixed random batch** (same input every step) to overfit as a sanity check. Loss is printed every 5 steps:

```
--- Training Monolith (Ground Truth) ---
Step 0  | Loss: 0.7312
Step 5  | Loss: 0.6891
...
Final Loss: 0.xxxxxx  Time: x.xxxs
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
brew install uv

# Install dependencies and set up Python 3.10 venv
uv sync

# Activate venv (optional)
source .venv/bin/activate
```

## Dependencies

- `torch >= 2.9.1`
- `numpy >= 2.2.6`
- Python `3.10`
