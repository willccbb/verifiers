# Rubrics

Rubrics are the evaluation heart of the verifiers framework. They combine multiple reward functions to assess model outputs from different perspectives, enabling nuanced evaluation beyond simple correctness.

## Rubric Architecture

```
Rubric (base class)
├── ToolRubric      # Evaluates tool usage and execution
├── JudgeRubric     # LLM-based evaluation
├── MathRubric      # Mathematical verification (deprecated)
├── SmolaToolRubric # Smolagents-specific evaluation
└── RubricGroup     # Combines multiple rubrics
```

## Core Concepts

### Reward Functions

A reward function evaluates one aspect of model output:

```python
def correctness_reward(completion, answer, **kwargs):
    """Check if the answer matches expected."""
    parsed = parser.parse(completion)
    return 1.0 if parsed.answer == answer else 0.0

def reasoning_quality(completion, **kwargs):
    """Evaluate reasoning clarity and depth."""
    parsed = parser.parse(completion)
    steps = parsed.reasoning.count('\n') + 1
    return min(steps / 5.0, 1.0)  # Up to 5 steps = full score
```

### Function Signatures

Reward functions can accept any subset of these parameters:
- `prompt`: The input prompt
- `completion`: Model's response (string or messages)
- `answer`: Ground truth answer
- `state`: Environment state dictionary
- `task`: Task type identifier
- `**kwargs`: Additional parameters

The framework inspects function signatures and passes only requested parameters:

```python
# Minimal signature
def simple_check(completion):
    return 1.0 if "sorry" not in completion.lower() else 0.0

# Full signature  
def complex_check(prompt, completion, answer, state, task):
    # Access all available information
    return evaluate_with_context(prompt, completion, answer, state, task)
```

### Weighted Aggregation

Rubrics combine multiple rewards with weights:

```python
rubric = Rubric(
    funcs=[correctness, reasoning_quality, format_check],
    weights=[0.7, 0.2, 0.1]  # Correctness is most important
)

# Final reward = 0.7 * correctness + 0.2 * reasoning + 0.1 * format
```

## Basic Rubric Usage

### Creating a Simple Rubric

```python
from verifiers.rubrics import Rubric
from verifiers.parsers import XMLParser

parser = XMLParser(["reasoning", "answer"])

def correct_answer(completion, answer, **kwargs):
    parsed = parser.parse(completion)
    return 1.0 if parsed.answer.strip() == answer.strip() else 0.0

def has_reasoning(completion, **kwargs):
    parsed = parser.parse(completion)
    return 1.0 if len(parsed.reasoning) > 20 else 0.0

rubric = Rubric(
    funcs=[correct_answer, has_reasoning],
    weights=[0.8, 0.2],
    parser=parser
)
```

### Scoring Single Outputs

```python
# Synchronous scoring
score_dict = rubric.score_rollout_sync(
    prompt="What is 2+2?",
    completion="<reasoning>2+2=4</reasoning><answer>4</answer>",
    answer="4",
    state={}
)

print(score_dict)
# {'correct_answer': 1.0, 'has_reasoning': 0.4, 'reward': 0.88}
```

### Batch Scoring

```python
# Score multiple outputs efficiently
prompts = ["What is 2+2?", "What is 3+3?"]
completions = ["<answer>4</answer>", "<answer>6</answer>"]
answers = ["4", "6"]
states = [{}, {}]

scores = rubric.score_rollouts(prompts, completions, answers, states)
print(scores)
# {'correct_answer': [1.0, 1.0], 'has_reasoning': [0.0, 0.0], 'reward': [0.8, 0.8]}
```

## Specialized Rubrics

### ToolRubric: Evaluating Tool Use

ToolRubric evaluates both correctness and proper tool usage:

```python
from verifiers.rubrics import ToolRubric
from verifiers.tools import calculator

tool_rubric = ToolRubric(
    tools=[calculator],
    weights=[0.5, 0.3, 0.2]  # correct_answer, tool_execution, format
)

# Automatically includes:
# - correct_answer_reward_func: Task-specific correctness
# - tool_execution_reward_func: Successful tool calls
# - format_reward_func: Proper XML formatting
# - Per-tool rewards: calculator_reward, calculator_count, calculator_attempt
```

Tool-specific rewards:
- `{tool}_reward`: 1.0 if tool used successfully, 0.0 otherwise
- `{tool}_count`: Number of successful tool uses
- `{tool}_attempt`: Number of tool use attempts

### JudgeRubric: LLM-Based Evaluation

Use another LLM to evaluate responses:

```python
from verifiers.rubrics import JudgeRubric

judge_rubric = JudgeRubric(
    judge_models=["gpt-4"],
    client=openai_client,
    template="""Evaluate this response for clarity and correctness.

Question: {prompt}
Answer: {completion}
Expected: {answer}

Score from 0-10:""",
    parser=XMLParser(["score", "feedback"])
)
```

### RubricGroup: Combining Rubrics

Aggregate multiple rubrics for comprehensive evaluation:

```python
# Create specialized rubrics
format_rubric = Rubric(
    funcs=[parser.get_format_reward_func()],
    weights=[1.0]
)

content_rubric = Rubric(
    funcs=[correct_answer, reasoning_quality],
    weights=[0.7, 0.3]
)

style_rubric = Rubric(
    funcs=[clarity_check, conciseness_check],
    weights=[0.5, 0.5]
)

# Combine them
combined = RubricGroup([content_rubric, format_rubric, style_rubric])

# Scores are aggregated across all rubrics
scores = combined.score_rollouts(prompts, completions, answers, states)
```

## Advanced Reward Functions

### State-Aware Rewards

Use environment state for context-dependent evaluation:

```python
def efficiency_reward(completion, state, **kwargs):
    """Reward based on number of attempts."""
    attempts = state.get('attempts', 1)
    return max(0, 1.0 - (attempts - 1) * 0.2)  # -0.2 per extra attempt

def improvement_reward(completion, state, **kwargs):
    """Reward if better than previous attempt."""
    current_score = evaluate_answer(completion)
    previous_score = state.get('previous_score', 0)
    return 1.0 if current_score > previous_score else 0.0
```

### Task-Specific Rewards

Different evaluation for different task types:

```python
def task_aware_reward(completion, answer, task, **kwargs):
    """Evaluate based on task type."""
    parsed = parser.parse(completion)
    
    if task == "classification":
        # Exact match for classification
        return 1.0 if parsed.answer == answer else 0.0
    
    elif task == "generation":
        # Similarity for generation tasks
        return calculate_similarity(parsed.answer, answer)
    
    elif task == "math":
        # Numerical comparison for math
        try:
            return float(eval(parsed.answer)) == float(eval(answer))
        except:
            return 0.0
    
    return 0.0
```

### Multi-Aspect Rewards

Evaluate multiple aspects in one function:

```python
def comprehensive_math_reward(completion, answer, **kwargs):
    """Evaluate math solutions holistically."""
    parsed = parser.parse(completion)
    scores = {}
    
    # Correctness
    try:
        is_correct = float(eval(parsed.answer)) == float(eval(answer))
        scores['correct'] = 1.0 if is_correct else 0.0
    except:
        scores['correct'] = 0.0
    
    # Method quality
    scores['shows_work'] = 1.0 if '=' in parsed.reasoning else 0.0
    scores['step_count'] = min(parsed.reasoning.count('\n') / 3, 1.0)
    
    # Clarity
    scores['clear_final'] = 1.0 if 'therefore' in parsed.reasoning.lower() else 0.5
    
    # Weighted combination
    return (scores['correct'] * 0.6 + 
            scores['shows_work'] * 0.2 +
            scores['step_count'] * 0.1 +
            scores['clear_final'] * 0.1)
```

## Async Evaluation

For expensive operations like API calls:

```python
import asyncio

class APIValidationRubric(Rubric):
    async def validate_with_api(self, answer):
        """Expensive API validation."""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={'answer': answer}) as resp:
                return await resp.json()
    
    async def api_validation_reward(self, completion, **kwargs):
        """Async reward function."""
        parsed = self.parser.parse(completion)
        result = await self.validate_with_api(parsed.answer)
        return result['score']

# Use async scoring
async def score_with_validation():
    scores = await rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer=answer,
        state={}
    )
    return scores
```

## Design Patterns

### 1. Progressive Evaluation

Start simple, add complexity:

```python
# Phase 1: Basic correctness
basic_rubric = Rubric(
    funcs=[correct_answer],
    weights=[1.0]
)

# Phase 2: Add format requirements
format_rubric = Rubric(
    funcs=[correct_answer, format_check],
    weights=[0.8, 0.2]
)

# Phase 3: Full evaluation
full_rubric = Rubric(
    funcs=[correct_answer, format_check, reasoning_quality, efficiency],
    weights=[0.5, 0.1, 0.3, 0.1]
)
```

### 2. Modular Rewards

Create reusable reward functions:

```python
# Generic reward functions
def length_penalty(max_length=500):
    def reward(completion, **kwargs):
        return max(0, 1.0 - len(completion) / max_length)
    return reward

def keyword_bonus(keywords):
    def reward(completion, **kwargs):
        found = sum(1 for kw in keywords if kw in completion.lower())
        return min(found / len(keywords), 1.0)
    return reward

# Compose into rubrics
rubric = Rubric(
    funcs=[
        correct_answer,
        length_penalty(max_length=300),
        keyword_bonus(['because', 'therefore', 'thus'])
    ],
    weights=[0.7, 0.1, 0.2]
)
```

### 3. Conditional Rubrics

Different evaluation based on conditions:

```python
class ConditionalRubric(Rubric):
    def score_rollout_sync(self, prompt, completion, answer, state, **kwargs):
        # Choose rubric based on task complexity
        if self.is_complex_task(prompt):
            return self.complex_rubric.score_rollout_sync(
                prompt, completion, answer, state, **kwargs
            )
        else:
            return self.simple_rubric.score_rollout_sync(
                prompt, completion, answer, state, **kwargs
            )
```

## Performance Optimization

### Caching Expensive Operations

```python
from functools import lru_cache

class CachedRubric(Rubric):
    @lru_cache(maxsize=1000)
    def expensive_parse(self, completion):
        """Cache parsing results."""
        return complex_parsing_logic(completion)
    
    def cached_reward(self, completion, **kwargs):
        parsed = self.expensive_parse(completion)
        return evaluate(parsed)
```

### Batch Processing

```python
def batch_similarity_reward(completions, answers):
    """Compute similarities in batch for efficiency."""
    # Vectorize all at once
    completion_vecs = vectorize_batch(completions)
    answer_vecs = vectorize_batch(answers)
    
    # Compute similarities
    similarities = cosine_similarity(completion_vecs, answer_vecs)
    
    return similarities.diagonal().tolist()
```

## Best Practices

### 1. Weight Selection
- Start with equal weights, tune based on data
- Correctness typically gets 50-80% weight
- Format/style usually 10-20%
- Adjust based on task priorities

### 2. Reward Function Design
- Return values in [0, 1] range
- Make functions deterministic when possible
- Handle edge cases gracefully
- Document expected inputs/outputs

### 3. Modular Construction
- Build small, focused reward functions
- Combine them into task-specific rubrics
- Reuse common patterns across projects

### 4. Testing Rewards
```python
# Test reward functions independently
def test_reasoning_reward():
    assert reasoning_quality("<reasoning>Step 1\nStep 2</reasoning>") > 0.3
    assert reasoning_quality("<reasoning>x</reasoning>") < 0.2
    assert reasoning_quality("") == 0.0
```

## Integration with Training

Rubrics directly influence model behavior through reinforcement learning:

1. **Initial Training**: High weight on format compliance
2. **Mid Training**: Balance correctness and quality
3. **Fine-tuning**: Emphasize nuanced aspects
4. **Evaluation**: Comprehensive multi-aspect scoring

The modular design allows you to evolve evaluation criteria as models improve, ensuring continuous advancement toward desired behavior.