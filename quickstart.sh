#!/bin/bash

# Get the number of GPUs on the machine minus 1
NUM_GPUS_MINUS_1=$(($(nvidia-smi --list-gpus | wc -l) - 1))
echo "Using ${NUM_GPUS_MINUS_1} GPUs"

accelerate launch --config-file configs/zero3.yaml --num-processes ${NUM_GPUS_MINUS_1} verifiers/examples/openbookqa_search.py