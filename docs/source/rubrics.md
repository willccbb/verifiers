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
from typing import Union, List, Dict, Any, Optional

def correctness_reward(completion: Union[str, List[Dict[str, str]]], answer: str, **kwargs) -> float:
    """Check if the answer matches expected."""
    parsed = parser.parse(completion)
    return 1.0 if parsed.answer == answer else 0.0

def reasoning_quality(completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
    """Evaluate reasoning clarity and depth."""
    parsed = parser.parse(completion)
    steps = parsed.reasoning.count('\n') + 1
    return min(steps / 5.0, 1.0)  # Up to 5 steps = full score
```

### Function Signatures

Reward functions can accept any subset of these parameters:
- `prompt`: Union[str, List[Dict[str, str]]] - The input prompt
- `completion`: Union[str, List[Dict[str, str]]] - Model's response (string or messages)
- `answer`: str - Ground truth answer
- `state`: Dict[str, Any] - Environment state dictionary
- `task`: str - Task type identifier
- `**kwargs`: Additional parameters

The framework inspects function signatures and passes only requested parameters:

```python
from typing import Union, List, Dict, Any

# Minimal signature
def simple_check(completion: Union[str, List[Dict[str, str]]]) -> float:
    return 1.0 if "sorry" not in completion.lower() else 0.0

# Full signature  
def complex_check(
    prompt: Union[str, List[Dict[str, str]]], 
    completion: Union[str, List[Dict[str, str]]], 
    answer: str, 
    state: Dict[str, Any], 
    task: str
) -> float:
    # Access all available information
    return evaluate_with_context(prompt, completion, answer, state, task)
```

### Weighted Aggregation

Rubrics combine multiple rewards with weights:

```python
from typing import List, Callable

rubric = Rubric(
    funcs=[correctness, reasoning_quality, format_check],  # List[Callable]
    weights=[0.7, 0.2, 0.1]  # List[float] - Correctness is most important
)

# Final reward = 0.7 * correctness + 0.2 * reasoning + 0.1 * format
```

## Basic Rubric Usage

### Creating a Simple Rubric

```python
from verifiers.rubrics import Rubric
from verifiers.parsers import XMLParser
from typing import Union, List, Dict

parser = XMLParser(["reasoning", "answer"])

def correct_answer(completion: Union[str, List[Dict[str, str]]], answer: str, **kwargs) -> float:
    parsed = parser.parse(completion)
    return 1.0 if parsed.answer.strip() == answer.strip() else 0.0

def has_reasoning(completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
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
from typing import Dict

# Synchronous scoring
score_dict: Dict[str, float] = rubric.score_rollout_sync(
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
from typing import List, Dict

# Score multiple outputs efficiently
prompts: List[str] = ["What is 2+2?", "What is 3+3?"]
completions: List[str] = ["<answer>4</answer>", "<answer>6</answer>"]
answers: List[str] = ["4", "6"]
states: List[Dict[str, Any]] = [{}, {}]

scores: Dict[str, List[float]] = rubric.score_rollouts(prompts, completions, answers, states)
print(scores)
# {'correct_answer': [1.0, 1.0], 'has_reasoning': [0.0, 0.0], 'reward': [0.8, 0.8]}
```

## Custom Rubric Implementation

**For nontrivial environments, users will want to write their own rubrics** to define task-specific evaluation criteria:

```python
from verifiers.rubrics import Rubric
from typing import Union, List, Dict, Any
import re

class MathRubric(Rubric):
    """Custom rubric for mathematical reasoning tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def correctness_reward(self, completion: Union[str, List[Dict[str, str]]], answer: str, **kwargs) -> float:
        """Check if the final answer is correct."""
        parsed = self.parser.parse(completion)
        return 1.0 if parsed.answer.strip() == answer.strip() else 0.0
    
    def reasoning_steps_reward(self, completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
        """Reward for showing step-by-step reasoning."""
        parsed = self.parser.parse(completion)
        steps = parsed.reasoning.count('\n') + 1
        return min(steps / 5.0, 1.0)  # Up to 5 steps = full score
    
    def mathematical_accuracy_reward(self, completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
        """Check for mathematical errors in reasoning."""
        parsed = self.parser.parse(completion)
        reasoning = parsed.reasoning.lower()
        
        # Check for common math errors
        errors = 0
        if '2+2=5' in reasoning:
            errors += 1
        if 'divide by zero' in reasoning:
            errors += 1
            
        return max(0.0, 1.0 - errors * 0.5)  # -0.5 per error
    
    def format_reward(self, completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
        """Reward for proper formatting."""
        return self.parser.get_format_reward_func()(completion)

# Use the custom rubric
math_rubric = MathRubric(
    funcs=[
        MathRubric.correctness_reward,
        MathRubric.reasoning_steps_reward,
        MathRubric.mathematical_accuracy_reward,
        MathRubric.format_reward
    ],
    weights=[0.5, 0.2, 0.2, 0.1],
    parser=XMLParser(["reasoning", "answer"])
)
```

## Accessing State Information

Reward functions can access detailed state information, including token-level data:

```python
from typing import Dict, Any, List, Union

def token_aware_reward(
    completion: Union[str, List[Dict[str, str]]], 
    state: Dict[str, Any], 
    **kwargs
) -> float:
    """Reward function that uses token-level information."""
    
    # Access token information from state
    if state.get("responses"):
        last_response = state["responses"][-1]
        
        # Check if we have token-level data
        if hasattr(last_response, 'choices') and last_response.choices:
            choice = last_response.choices[0]
            
            # Access logprobs if available
            if hasattr(choice, 'logprobs') and choice.logprobs:
                logprobs = choice.logprobs.content
                token_ids = choice.logprobs.token_ids
                
                # Calculate average log probability
                if logprobs:
                    avg_logprob = sum(logprob.logprob for logprob in logprobs) / len(logprobs)
                    return max(0.0, min(1.0, (avg_logprob + 5.0) / 5.0))  # Normalize to [0,1]
    
    # Fallback to basic reward
    return 1.0 if "correct" in completion.lower() else 0.0
```

## Specialized Rubrics

### ToolRubric: Evaluating Tool Use

ToolRubric evaluates both correctness and proper tool usage:

```python
from verifiers.rubrics import ToolRubric
from verifiers.tools import calculator
from typing import List, Callable

tool_rubric = ToolRubric(
    tools=[calculator],  # List[Callable]
    weights=[0.5, 0.3, 0.2]  # List[float] - correct_answer, tool_execution, format
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
from verifiers.parsers import XMLParser
from typing import List

judge_rubric = JudgeRubric(
    judge_models=["gpt-4"],  # List[str]
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
from typing import List

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
combined = RubricGroup([content_rubric, format_rubric, style_rubric])  # List[Rubric]

# Scores are aggregated across all rubrics
scores: Dict[str, List[float]] = combined.score_rollouts(prompts, completions, answers, states)
```

## Rubric Integration with Environments

Rubrics work seamlessly with environments for both evaluation and training:

```python
from typing import Dict, Any

# Create environment with custom rubric
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=parser,
    rubric=custom_rubric
)

# Evaluate model performance
results: Dict[str, Any] = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=100
)

# Generate training data with rewards
results: Dict[str, Any] = vf_env.generate(
    client=openai_client,
    model="gpt-4",
    n_samples=1000
)
```

## Advanced Rubric Patterns

### 1. Context-Aware Evaluation

```python
from typing import Dict, Any, Union, List

def context_aware_reward(
    completion: Union[str, List[Dict[str, str]]], 
    prompt: Union[str, List[Dict[str, str]]], 
    state: Dict[str, Any], 
    **kwargs
) -> float:
    """Evaluate based on conversation context."""
    # Check if this is a follow-up question
    if "previous" in state:
        # Require reference to previous answer
        if state["previous"] not in completion:
            return 0.5  # Partial credit
    return 1.0
```

### 2. Multi-Stage Evaluation

```python
from typing import Union, List, Dict

class MultiStageRubric(Rubric):
    """Rubric that evaluates different stages of reasoning."""
    
    def planning_reward(self, completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
        """Evaluate planning phase."""
        # Extract planning from completion
        return evaluate_planning(completion)
    
    def execution_reward(self, completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
        """Evaluate execution phase."""
        # Extract execution from completion
        return evaluate_execution(completion)
    
    def verification_reward(self, completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
        """Evaluate verification phase."""
        # Extract verification from completion
        return evaluate_verification(completion)
```

### 3. Adaptive Rewards

```python
from typing import List, Union, Dict

class AdaptiveRubric(Rubric):
    """Rubric that adapts based on model performance."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history: List[float] = []
    
    def adaptive_reward(self, completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
        """Adjust reward based on historical performance."""
        base_reward = self.base_reward_func(completion, **kwargs)
        
        # Adjust based on recent performance
        if len(self.performance_history) > 10:
            avg_performance = sum(self.performance_history[-10:]) / 10
            if avg_performance < 0.5:
                # Increase reward for struggling models
                return min(1.0, base_reward * 1.2)
        
        return base_reward
```

## Message Format Handling

Rubrics handle both chat and completion formats:

```python
from typing import Union, List, Dict

def format_agnostic_reward(completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
    """Reward function that works with both formats."""
    
    if isinstance(completion, str):
        # Completion format - direct string
        return evaluate_string(completion)
    else:
        # Chat format - extract content from messages
        content = " ".join(msg["content"] for msg in completion if msg["role"] == "assistant")
        return evaluate_string(content)
```

## Key Gotchas

1. **Parser Integration**: Always include format rewards from your parser
2. **Weight Balancing**: Ensure weights sum to reasonable values (typically 1.0)
3. **Error Handling**: Reward functions should handle parsing failures gracefully
4. **Performance**: Keep reward functions efficient for batch processing
5. **Custom Rubrics**: For complex tasks, write custom rubrics rather than trying to force built-in ones
6. **Type Hints**: Use proper type hints for better code clarity and IDE support
7. **State Access**: Use state information carefully - it accumulates across the entire rollout

## Best Practices

1. **Start Simple**: Begin with basic correctness and format rewards
2. **Add Complexity**: Gradually add more sophisticated evaluation criteria
3. **Test Thoroughly**: Test rubrics with various input types
4. **Document Intent**: Clearly document what each reward function evaluates
5. **Monitor Performance**: Track how different reward components affect training
6. **Use Type Hints**: Include proper type hints for better code clarity
7. **Handle Errors**: Gracefully handle parsing failures and malformed input