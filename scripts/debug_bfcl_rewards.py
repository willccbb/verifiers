#!/usr/bin/env python3
"""Debug BFCL reward function."""

import sys
sys.path.append('.')

import verifiers as vf
from environments.vf_bfcl_single_turn import vf_bfcl_single_turn

# Load environment
env = vf_bfcl_single_turn.load_environment(num_train_examples=10, num_eval_examples=2)

# Get a sample from the dataset
sample = env.dataset[0]
print("Sample prompt:", sample['prompt'])
print("Sample info:", sample['info'])

# Test the parser
parser = env.parser
test_completion = [
    {
        "role": "assistant",
        "content": 'I\'ll help you with that. <function_calls>[{"name": "send_email", "arguments": {"to": "john@example.com", "subject": "Meeting Update", "body": "This is an automated message about Meeting Update."}}]</function_calls>'
    }
]

print("\nTest completion:", test_completion)

# Test parsing
parsed = parser.parse_answer(test_completion)
print("Parsed result:", parsed)

# Test reward function directly from rubric
print("\nRubric reward functions:", env.rubric.reward_funcs)
print("Number of reward functions:", len(env.rubric.reward_funcs))

# Get all rewards
test_info = {
    "expected_calls": [{"name": "send_email", "arguments": {"to": "john@example.com", "subject": "Meeting Update", "body": "This is an automated message about Meeting Update."}}]
}

rewards = env.rubric.score(test_completion, test_info)
print("Total rewards from rubric:", rewards)

# Now test with a more realistic completion format
print("\n" + "="*50)
print("Testing with actual training format...")

# Simulate what GRPO sends
grpo_completion = test_completion  # GRPO sends the full message list

print("GRPO completion format:", grpo_completion)
reward2 = reward_func(grpo_completion, test_info)
print("Reward with GRPO format:", reward2)