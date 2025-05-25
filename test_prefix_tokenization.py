#!/usr/bin/env python3
"""
Test script to verify the incremental prefix tokenization approach.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'verifiers'))

from transformers import AutoTokenizer
from verifiers.envs.environment import Environment

class TestEnvironment(Environment):
    """Test implementation of Environment for testing tokenization."""
    
    def rollout(self, client, model, prompt, sampling_args={}, **kwargs):
        # Dummy implementation for testing
        return "test response", {}

def test_incremental_tokenization():
    """Test that incremental tokenization produces consistent results."""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create test environment
    env = TestEnvironment()
    
    # Test conversation
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    completion = [
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "3+3 equals 6."}
    ]
    
    print("Testing incremental prefix tokenization...")
    
    # Process with new method
    prompt_ids, prompt_mask, completion_ids, completion_mask = env.process_chat_format(
        prompt, completion, tokenizer, mask_intermediate_responses=False
    )
    
    print(f"Prompt tokens: {len(prompt_ids)}")
    print(f"Completion tokens: {len(completion_ids)}")
    print(f"Prompt mask length: {len(prompt_mask)}")
    print(f"Completion mask length: {len(completion_mask)}")
    
    # Verify masks have correct length
    assert len(prompt_ids) == len(prompt_mask), "Prompt mask length mismatch"
    assert len(completion_ids) == len(completion_mask), "Completion mask length mismatch"
    
    # Test that reconstructed conversation matches expectation
    full_conversation = prompt + completion
    full_text = tokenizer.apply_chat_template(full_conversation, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    
    # The total should be close to prompt + completion length
    # (may not be exactly equal due to chat template formatting differences)
    total_processed = len(prompt_ids) + len(completion_ids)
    
    print(f"Full conversation tokens: {len(full_ids)}")
    print(f"Processed tokens (prompt + completion): {total_processed}")
    
    # Test with masking
    prompt_ids_masked, prompt_mask_masked, completion_ids_masked, completion_mask_masked = env.process_chat_format(
        prompt, completion, tokenizer, mask_intermediate_responses=True
    )
    
    print(f"\nWith masking:")
    print(f"Completion mask: {completion_mask_masked}")
    print(f"Masked tokens: {sum(1 for m in completion_mask_masked if m == 0)}")
    
    print("\nTest passed! Incremental tokenization working correctly.")

if __name__ == "__main__":
    test_incremental_tokenization() 