# TruthfulQA Environment

This environment implements the standard TruthfulQA evaluation methodology for measuring how models mimic human falsehoods.

## Overview

TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception.

## Evaluation Methodology

Following the original TruthfulQA paper (Lin et al., 2022), this environment evaluates two orthogonal aspects:

1. **Truthfulness (True %)**: The percentage of model answers that are true/factually accurate
2. **Informativeness (Info %)**: The percentage of model answers that provide information (vs refusing to answer)

The main metric reported is **True\*Info (%)**: The percentage of answers that are both truthful AND informative.

### Important Note on Judge Model

The original TruthfulQA paper used fine-tuned GPT-3 models (GPT-judge and GPT-info) for evaluation. Since these models are no longer available from OpenAI, this implementation uses **GPT-4o-mini** as the judge model for both truthfulness and informativeness evaluation. While this represents a deviation from the original methodology, GPT-4o-mini provides strong performance for this type of evaluation task.

## Usage

```python
import verifiers as vf

# Load environment
env = vf.load_environment("truthful-qa", num_eval_examples=50)

# Evaluate a model
from openai import OpenAI
client = OpenAI()
results = env.evaluate(client, "gpt-4o-mini", num_examples=50)

# Get TruthfulQA metrics
metrics = env.compute_truthfulqa_metrics(results)
print(f"True (%): {metrics['True (%)']:.1f}")
print(f"Info (%): {metrics['Info (%)']:.1f}")  
print(f"True*Info (%): {metrics['True*Info (%)']:.1f}")
```

## Configuration Options

- `num_train_examples`: Number of training examples (-1 for all)
- `num_eval_examples`: Number of evaluation examples (-1 for all)
- `generation_type`: Task type - "generation" (default) or "multiple_choice"
- `judge_model`: Model for evaluation (default: "gpt-4o-mini")

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