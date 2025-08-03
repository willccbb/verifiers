# BFCL Single Turn Function Calling Environment

This environment implements single turn function calling evaluation using the Berkeley Function Calling Leaderboard (BFCL) v3 dataset.

## Installation

```bash
vf-install vf-bfcl-single-turn
```

## Usage

```bash
# Quick evaluation
vf-eval vf-bfcl-single-turn -m model_name

# Training
CUDA_VISIBLE_DEVICES=0 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_bfcl_single_turn.py
```

## Description

This environment trains models to perform single-turn function calling tasks from the BFCL v3 dataset. The model must analyze user queries and determine which functions to call with the appropriate arguments.