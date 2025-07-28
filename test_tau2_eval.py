#!/usr/bin/env python3
"""Test script for tau2bench evaluation."""

import sys
sys.path.insert(0, '/workspace')

from environments.tau2_bench.tau2_bench import load_environment
import json

# Test with a simple retail task
env = load_environment(domain="retail", subset_size=1)

# Get the first sample
sample = env.dataset[0]
print("Task ID:", sample['task_id'])
print("Question:", sample['question'])
print("Initial messages:", json.dumps(sample['prompt'], indent=2))

# Create a simple test completion that performs the expected actions
# For retail domain, let's just check what actions are expected
info = sample['info']
if 'evaluation_criteria' in info and 'actions' in info['evaluation_criteria']:
    print("\nExpected actions:")
    for action in info['evaluation_criteria']['actions']:
        print(f"  - {action['requestor']}: {action['name']}({action['arguments']})")

# Test evaluation with empty completion
print("\nTesting evaluation with empty completion...")
state = {"task_id": sample['task_id'], "tool_executions": [], "turn_count": 0}
# Call the reward function directly
if env.rubric.reward_funcs:
    score = env.rubric.reward_funcs[0]([], info, state)
    print(f"Score for empty completion: {score}")
else:
    print("No reward functions found in rubric")

print("\nDone!")