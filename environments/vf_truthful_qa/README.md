# TruthfulQA Environment

This environment implements the standard TruthfulQA evaluation methodology for measuring how models mimic human falsehoods.

> **Note**: This evaluation environment was created by Claude 4 Opus. While it aims to faithfully implement the TruthfulQA methodology, users should carefully review and validate the evaluation logic before relying on it for downstream evaluations.

## Overview

TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception.

## Evaluation Methodology

Following the original TruthfulQA paper (Lin et al., 2022), this environment evaluates two orthogonal aspects:

1. **Truthfulness (True %)**: The percentage of model answers that are true/factually accurate
2. **Informativeness (Info %)**: The percentage of model answers that provide information (vs refusing to answer)

The main metric reported is **True\*Info (%)**: The percentage of answers that are both truthful AND informative.

### Implementation as Reward Functions

The evaluation is implemented using three reward functions:
- `truthfulness_reward_func`: Returns 1.0 if the answer is truthful, 0.0 otherwise
- `informativeness_reward_func`: Returns 1.0 if the answer is informative, 0.0 otherwise  
- `truthful_and_informative_reward_func`: Returns 1.0 only if both truthfulness and informativeness are 1.0

The environment uses sequential evaluation with state sharing to ensure efficiency - the truthfulness and informativeness evaluations happen first and store their results, which are then used by the combined metric.

### Important Note on Judge Model

The original TruthfulQA paper used fine-tuned GPT-3 models (GPT-judge and GPT-info) for evaluation. Since these models are no longer available from OpenAI, this implementation uses **GPT-4.1-mini** as the judge model for both truthfulness and informativeness evaluation. While this represents a deviation from the original methodology, GPT-4.1-mini provides strong performance for this type of evaluation task.

## Usage

```python
import verifiers as vf

# Load environment
env = vf.load_environment("truthful-qa", num_eval_examples=50)

# Evaluate a model
from openai import OpenAI
client = OpenAI()
results = env.evaluate(client, "gpt-4.1-mini", num_examples=50)

# The results include three reward functions:
# - truthfulness_reward_func: 1.0 if answer is truthful
# - informativeness_reward_func: 1.0 if answer is informative
# - truthful_and_informative_reward_func: 1.0 if both truthful AND informative

# Extract metrics from results
truth_scores = results["truthfulness_reward_func"]
info_scores = results["informativeness_reward_func"]
both_scores = results["truthful_and_informative_reward_func"]

print(f"True (%): {sum(truth_scores) / len(truth_scores) * 100:.1f}")
print(f"Info (%): {sum(info_scores) / len(info_scores) * 100:.1f}")
print(f"True*Info (%): {sum(both_scores) / len(both_scores) * 100:.1f}")
```

## Configuration Options

- `num_train_examples`: Number of training examples (-1 for all)
- `num_eval_examples`: Number of evaluation examples (-1 for all)
- `generation_type`: Task type - "generation" (default) or "multiple_choice"
- `judge_model`: Model for evaluation (default: "gpt-4.1-mini")

## Dataset

The environment uses the official TruthfulQA dataset from Hugging Face, which includes:
- 817 questions designed to elicit false answers
- Multiple correct reference answers per question
- Common incorrect answers that reflect human misconceptions

## Citation

If you use this environment, please cite the original TruthfulQA paper:

```bibtex
@inproceedings{lin2022truthfulqa,
    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
    author={Stephanie Lin and Jacob Hilton and Owain Evans},
    booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2022}
}
```