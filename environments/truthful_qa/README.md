# TruthfulQA Environment

This environment evaluates models on the TruthfulQA dataset, testing their ability to provide accurate and truthful information across various topics.

## Features

- Multi-criteria evaluation: correctness, avoiding misinformation, informativeness, and clarity
- Configurable difficulty levels based on answer complexity
- Support for best answer or multiple correct answers
- Penalizes responses containing known incorrect information

## Usage

```python
import verifiers as vf
from openai import OpenAI

# Load environment
env = vf.load_environment("truthful-qa", num_eval_examples=50)

# Evaluate
client = OpenAI()
results = env.evaluate(client, "gpt-4o-mini", num_examples=50)
print(results)
```

## Configuration Options

- `num_train_examples`: Number of training examples (-1 for all)
- `num_eval_examples`: Number of evaluation examples (-1 for all)  
- `difficulty`: Filter by difficulty ("all", "easy", "medium", "hard")
- `use_best_answer`: Use best answer (True) or first correct answer (False)

## Reward Functions

1. **Correctness** (weight: 1.0): Checks if response contains the correct answer
2. **Avoid Incorrect** (weight: 0.8): Penalizes known incorrect answers
3. **Informativeness** (weight: 0.3): Rewards appropriate response length
4. **Clarity** (weight: 0.2): Rewards well-structured, explanatory responses