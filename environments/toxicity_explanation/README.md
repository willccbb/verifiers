# Toxicity Explanation Environment

An environment for evaluating models on toxicity detection with context-aware explanations.

## Overview

This environment tests a model's ability to:
1. **Classify** text as toxic or non-toxic
2. **Explain** the reasoning behind the classification with specific examples

The environment uses the Google Jigsaw Toxicity dataset and employs JudgeRubric for nuanced evaluation of both classification accuracy and explanation quality.

## Dataset

- **Source**: `google/civil_comments` from Hugging Face
- **Content**: Online comments with continuous toxicity scores
- **Categories**: target (main toxicity), severe_toxicity, obscene, threat, insult, identity_attack
- **Threshold**: Comments with scores >= 0.5 are considered toxic

## Evaluation

The environment uses two judge-based evaluations:

1. **Classification Accuracy** (weight: 1.0)
   - Binary evaluation of whether the model correctly identified toxicity
   
2. **Explanation Quality** (weight: 0.5)
   - Evaluates explanations based on:
     - Specificity (references to text)
     - Context awareness
     - Clarity of reasoning
     - Completeness

## Usage

```python
import verifiers as vf
from openai import OpenAI

# Load the environment
env = vf.load_environment(
    "toxicity_explanation",
    judge_model="gpt-4.1-mini",  # Model to use for evaluation
    max_examples=100  # Limit dataset size for testing
)

# Evaluate a model
client = OpenAI()
results = env.evaluate(
    client, 
    "gpt-4.1-mini", 
    num_examples=10,
    rollouts_per_example=1
)
```

## Parameters

- `judge_model`: Model to use for judging responses (default: "gpt-4.1-mini")
- `judge_base_url`: Base URL for judge API (default: OpenAI)
- `judge_api_key_var`: Environment variable for API key (default: "OPENAI_API_KEY")
- `max_examples`: Limit the number of examples from the dataset

## Example Output Format

The model should respond in this format:
```
<classification>toxic/non-toxic</classification>
<explanation>Your detailed explanation here</explanation>
```
