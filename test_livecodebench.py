#!/usr/bin/env python3
"""Test LiveCodeBench environment with gpt-4.1-mini"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the environment
from environments.livecodebench import livecodebench

# Create environment
env = livecodebench.load_environment(num_examples=5)

print("LiveCodeBench Environment Test")
print("=" * 50)
print(f"Dataset size: {len(env.dataset)}")
print("\nExample problems:")
for i, example in enumerate(env.dataset):
    prompt = example['prompt'][0]['content']
    # Extract function name
    import re
    func_match = re.search(r'function called `(\w+)`', prompt)
    func_name = func_match.group(1) if func_match else "unknown"
    print(f"\n{i+1}. {func_name}:")
    print(f"   {prompt.split('.')[0]}.")

# Test with a simple mock completion
print("\n" + "=" * 50)
print("Testing code extraction and evaluation:")

# Mock a completion for the first problem (add_two)
mock_completion = [{
    'role': 'user',
    'content': env.dataset[0]['prompt'][0]['content']
}, {
    'role': 'assistant', 
    'content': '''Here's the solution:

```python
def add_two(a, b):
    return a + b
```

This function takes two integers as parameters and returns their sum.'''
}]

# Test code extraction
code = env.parser.parse_answer(mock_completion)
print(f"\nExtracted code:\n{code}")

# Test correctness evaluation
score = env.rubric.funcs[0](env.parser, mock_completion, env.dataset[0]['info'])
print(f"\nCorrectness score: {score:.2f} ({int(score * 100)}%)")

# If you have API access, you can test with real model
if os.getenv('OPENAI_API_KEY'):
    print("\n" + "=" * 50)
    print("Testing with gpt-4.1-mini (if API key available):")
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Test on first example only
        results = env.evaluate(
            client=client,
            model="gpt-4.1-mini",
            num_examples=1,
            rollouts_per_example=1
        )
        
        print(f"\nResults: {results}")
    except Exception as e:
        print(f"\nAPI test failed: {e}")
else:
    print("\n" + "=" * 50)
    print("No OPENAI_API_KEY found, skipping API test")
    print("To test with gpt-4.1-mini, set your OPENAI_API_KEY environment variable")