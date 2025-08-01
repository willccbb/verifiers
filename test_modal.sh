#!/bin/bash
# Test script for Modal training commands

echo "Setting up dummy environment variables..."
export WANDB_API_KEY="dummy-wandb-key"
export OPENAI_API_KEY="dummy-openai-key"
export HF_TOKEN="dummy-hf-token"

echo ""
echo "=== Testing Modal Commands ==="
echo ""

# Test 1: Check if modal is working
echo "1. Testing basic modal command:"
modal --version || echo "Modal CLI not installed. Please run: pip install modal"

echo ""
echo "2. Testing sync command:"
echo "Command: modal run run_modal.py --sync-only"
echo "(This will sync your local files to Modal's storage)"

echo ""
echo "3. Testing Wordle training command:"
echo "Command: modal run run_modal.py --cmd \"python examples/grpo/train_wordle.py --size 1.7B\""

echo ""
echo "4. Testing Tool Test training command:"
echo "Command: modal run run_modal.py --cmd \"python examples/grpo/train_tool_test.py\""

echo ""
echo "5. Testing with wrapper scripts:"
echo "Command: modal run run_modal.py --cmd \"bash scripts/modal_train_wordle.sh\""
echo "Command: modal run run_modal.py --cmd \"bash scripts/modal_train_tool_test.sh\""

echo ""
echo "To run these commands, execute them manually with the environment variables set."
echo "Example:"
echo "  export WANDB_API_KEY=\"your-actual-key\""
echo "  export OPENAI_API_KEY=\"your-actual-key\""
echo "  export HF_TOKEN=\"your-actual-token\""
echo "  modal run run_modal.py --cmd \"python examples/grpo/train_wordle.py\""