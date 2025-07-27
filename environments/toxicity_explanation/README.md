# Toxicity Explanation Environment

> **Note**: This environment was created by Claude 4 Opus as a demonstration. Downstream users should carefully review and validate its logic before relying on it for production use.

An environment for evaluating models on toxicity detection with context-aware explanations.

## Setup

Install the environment using the verifiers CLI:

```bash
# From the project root
uv run vf-install toxicity_explanation

# Or specify a custom path
uv run vf-install toxicity_explanation -p /path/to/environments
```

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

The environment uses JudgeRubric-based evaluation to avoid complex if-statement logic:

- **Judge Model**: Evaluates responses on a scale of 0-10
- **Scoring Criteria**:
  - 0-3: Incorrect classification or poor explanation
  - 4-6: Correct classification but weak explanation
  - 7-8: Correct classification with good explanation
  - 9-10: Correct classification with excellent, nuanced explanation

The judge considers:
1. Classification accuracy
2. Explanation quality (specific examples cited)
3. Contextual understanding (intent and context considered)

## Usage

```bash
# Basic evaluation
uv run vf-eval toxicity_explanation -m gpt-4.1-mini -n 10

# With custom parameters
uv run vf-eval toxicity_explanation -m gpt-4.1-mini -n 20 --rollouts-per-example 3 --env-args '{"max_examples": 100}'

# Save results to disk
uv run vf-eval toxicity_explanation -m gpt-4.1-mini -n 10 --save-dataset --save-path my_results
```

## Response Format

Models are instructed to respond in the following format:
```xml
<classification>toxic/non-toxic</classification>
<explanation>Your detailed explanation here</explanation>
```

## Example

Input text:
> "This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!"

Expected output:
- Classification: non-toxic
- Explanation: The text expresses positive sentiment with compliments like "so cool" and "well done". The rhetorical question about mothers is used playfully to emphasize appropriateness rather than as criticism.

## Configuration

The environment accepts the following parameters:
- `judge_model`: Model to use for evaluation (default: "gpt-4.1-mini")
- `judge_base_url`: API base URL for the judge model
- `judge_api_key_var`: Environment variable containing the API key
- `max_examples`: Limit the number of examples to load from the dataset

## Implementation Notes

This environment demonstrates:
- Using real-world Hugging Face datasets
- Implementing JudgeRubric for complex evaluation without if-statement logic
- Proper reward extraction from judge responses
- Context-aware toxicity detection beyond simple keyword matching
