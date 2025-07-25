# Rubrics

Rubrics define multi-criteria evaluation for model outputs. They combine multiple reward functions with weights to create comprehensive scoring systems. **For nontrivial environments, users will want to write their own rubrics** to define task-specific evaluation criteria.

## Rubric Hierarchy

```
Rubric (base class)
├── JudgeRubric     # LLM-judge based evaluation
├── ToolRubric      # Tool usage tracking
├── RubricGroup     # Composition of multiple rubrics
└── Custom Rubrics  # Task-specific evaluation
```

## Basic Rubric Usage

The base `Rubric` class combines multiple reward functions:

```python
import verifiers as vf

def correct_answer_func(parser, completion, answer, **kwargs) -> float:
    """Check if the parsed answer matches the expected answer."""
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

def format_reward_func(parser, completion, **kwargs) -> float:
    """Check if the completion follows expected format."""
    return parser.get_format_reward_func()(completion)

# Create rubric with multiple criteria
rubric = vf.Rubric(
    funcs=[correct_answer_func, format_reward_func],
    weights=[0.8, 0.2],  # Correctness weighted higher than format
    parser=parser
)

# Use in environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=parser,
    rubric=rubric
)
```

## Multi-Criteria Evaluation

Real-world tasks benefit from multiple evaluation criteria:

```python
import verifiers as vf

def correctness_reward(parser, completion, answer, **kwargs) -> float:
    """Primary correctness measure."""
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

def reasoning_quality_reward(parser, completion, **kwargs) -> float:
    """Evaluate quality of reasoning steps."""
    if hasattr(parser, 'parse') and hasattr(parser.parse(completion), 'reasoning'):
        reasoning = parser.parse(completion).reasoning or ''
        # Simple heuristic: longer reasoning with steps gets higher score
        if 'step' in reasoning.lower() and len(reasoning) > 50:
            return 1.0
        elif len(reasoning) > 20:
            return 0.5
    return 0.0

def confidence_penalty(parser, completion, **kwargs) -> float:
    """Penalize overconfident wrong answers."""
    response = parser.parse_answer(completion) or ''
    if 'definitely' in response.lower() or 'certainly' in response.lower():
        # If overconfident but wrong, apply penalty
        is_correct = response.strip() == kwargs.get('answer', '').strip()
        return 0.0 if not is_correct else 1.0
    return 1.0  # No penalty for appropriate confidence

# Combine multiple criteria
rubric = vf.Rubric(
    funcs=[
        correctness_reward,      # Primary objective
        reasoning_quality_reward, # Reasoning process
        confidence_penalty,      # Calibration
        parser.get_format_reward_func()  # Format compliance
    ],
    weights=[1.0, 0.3, -0.2, 0.2],  # Note: negative weight for penalty
    parser=parser
)
```

## Built-in Rubric Types

### JudgeRubric: LLM-Based Evaluation

Use LLM judges for complex evaluation:

```python
from openai import OpenAI

judge_rubric = vf.JudgeRubric(
    judge_client=OpenAI(),
    judge_model="gpt-4.1-mini",
    judge_prompt="""Given a question and two responses, determine which is better.

Question: {question}
Response: {response}
Ground Truth: {answer}

Rate the response on a scale of 0-1 for correctness.""",
    parallelize_scoring=False  # For API rate limits
)

# Use in combination with other rewards
combined_rubric = vf.RubricGroup([
    main_rubric,
    judge_rubric
])
```

### ToolRubric: Tool Usage Tracking

Track and reward tool usage patterns:

```python
from verifiers.utils.tools import python, calculator

tool_rubric = vf.ToolRubric(tools=[python, calculator])

# This automatically creates reward functions for:
# - total_tool_calls: Total number of tool invocations
# - python_tool_calls: Number of python tool calls
# - calculator_tool_calls: Number of calculator tool calls

# Combine with task-specific rubric
combined_rubric = vf.RubricGroup([
    task_rubric,    # Task correctness
    tool_rubric     # Tool usage metrics (weights set to 0.0 by default)
])
```

### RubricGroup: Composition

Combine multiple rubrics for complex evaluation:

```python
# Create specialized rubrics
math_rubric = vf.Rubric(funcs=[math_correctness_func], weights=[1.0])
format_rubric = vf.Rubric(funcs=[format_compliance_func], weights=[1.0])  
tool_rubric = vf.ToolRubric(tools=[python])

# Compose into group
combined_rubric = vf.RubricGroup([
    math_rubric,
    format_rubric,
    tool_rubric
])

# The group aggregates all reward functions from constituent rubrics
```

## Custom Rubric Patterns

### Domain-Specific Rubric

```python
import verifiers as vf

class MathRubric(vf.Rubric):
    """Specialized rubric for mathematical reasoning."""
    
    def __init__(self, parser=None, **kwargs):
        super().__init__(parser=parser, **kwargs)
        
        # Add math-specific reward functions
        self.add_reward_func(self.correct_answer_reward_func, weight=1.0)
        self.add_reward_func(self.step_by_step_reward_func, weight=0.3)
        self.add_reward_func(self.calculation_accuracy_func, weight=0.2)
        self.add_reward_func(self.parser.get_format_reward_func(), weight=0.2)
    
    def correct_answer_reward_func(self, parser, completion, answer, **kwargs) -> float:
        """Check mathematical correctness."""
        try:
            from math_verify import parse, verify
            response = parser.parse_answer(completion) or ''
            return 1.0 if verify(parse(response), parse(answer)) else 0.0
        except:
            # Fallback to string matching
            response = parser.parse_answer(completion) or ''
            return 1.0 if response.strip() == answer.strip() else 0.0
    
    def step_by_step_reward_func(self, parser, completion, **kwargs) -> float:
        """Reward step-by-step reasoning."""
        text = completion if isinstance(completion, str) else completion[-1]['content']
        
        # Count reasoning steps
        step_indicators = ['step', 'first', 'second', 'next', 'then', 'therefore']
        step_count = sum(1 for indicator in step_indicators if indicator in text.lower())
        
        return min(1.0, step_count / 3.0)  # Normalize to max 1.0
    
    def calculation_accuracy_func(self, parser, completion, **kwargs) -> float:
        """Check intermediate calculations."""
        text = completion if isinstance(completion, str) else completion[-1]['content']
        
        # Simple heuristic: look for basic arithmetic
        import re
        calculations = re.findall(r'(\d+\s*[+\-*/]\s*\d+\s*=\s*\d+)', text)
        
        correct_calculations = 0
        for calc in calculations:
            try:
                left, right = calc.split('=')
                if eval(left.strip()) == int(right.strip()):
                    correct_calculations += 1
            except:
                continue
        
        return 1.0 if not calculations else correct_calculations / len(calculations)
```

### Contextual Reward Functions

```python
def difficulty_adjusted_reward(parser, completion, answer, info=None, **kwargs) -> float:
    """Adjust rewards based on problem difficulty."""
    base_correctness = 1.0 if parser.parse_answer(completion) == answer else 0.0
    
    if info and 'difficulty' in info:
        difficulty = info['difficulty'].lower()
        if difficulty == 'easy':
            return base_correctness
        elif difficulty == 'medium':
            return base_correctness * 1.2  # Bonus for harder problems
        elif difficulty == 'hard':
            return base_correctness * 1.5
    
    return base_correctness

def length_penalty_reward(parser, completion, **kwargs) -> float:
    """Penalize overly verbose responses."""
    text = completion if isinstance(completion, str) else completion[-1]['content']
    length = len(text.split())
    
    if length < 10:
        return 0.5  # Too short
    elif length < 100:
        return 1.0  # Good length
    elif length < 200:
        return 0.8  # Getting long
    else:
        return 0.5  # Too verbose
```

### Rubric with State Tracking

```python
class ProgressiveRubric(vf.Rubric):
    """Rubric that tracks improvement over time."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_history = []
    
    def compute_rewards(self, prompt, completion, answer, task, info, state, **kwargs):
        """Override to track score history."""
        rewards = super().compute_rewards(prompt, completion, answer, task, info, state, **kwargs)
        
        # Track total score
        total_score = sum(r * w for r, w in zip(rewards, self.get_reward_weights()))
        self.score_history.append(total_score)
        
        # Add improvement bonus
        if len(self.score_history) > 1:
            improvement = self.score_history[-1] - self.score_history[-2]
            if improvement > 0:
                rewards.append(improvement * 0.1)  # Small improvement bonus
            else:
                rewards.append(0.0)
            
        return rewards
```

## Reward Function Patterns

### Common Reward Function Signatures

All reward functions should follow this pattern:

```python
def reward_function(
    parser,      # Parser instance
    completion,  # Model completion (str or List[Dict])
    answer=None, # Ground truth answer
    prompt=None, # Original prompt
    info=None,   # Additional info from dataset
    state=None,  # Rollout state
    task=None,   # Task identifier
    **kwargs     # Additional arguments
) -> float:
    """Return a float reward value."""
    pass
```

### Error-Safe Reward Functions

```python
def safe_reward_function(parser, completion, answer, **kwargs) -> float:
    """Reward function with comprehensive error handling."""
    try:
        # Main reward logic
        response = parser.parse_answer(completion)
        if response is None:
            return 0.0
        
        # Compute reward
        return 1.0 if response.strip() == answer.strip() else 0.0
        
    except AttributeError:
        # Parser doesn't have parse_answer
        return 0.0
    except Exception as e:
        # Log error but don't crash
        print(f"Reward function error: {e}")
        return 0.0
```

### Partial Credit Patterns

```python
def partial_credit_math(parser, completion, answer, **kwargs) -> float:
    """Give partial credit for math problems."""
    response = parser.parse_answer(completion) or ''
    
    if response.strip() == answer.strip():
        return 1.0  # Full credit
    
    # Check if final numerical answer is close
    try:
        import re
        response_nums = re.findall(r'-?\d+\.?\d*', response)
        answer_nums = re.findall(r'-?\d+\.?\d*', answer)
        
        if response_nums and answer_nums:
            resp_val = float(response_nums[-1])
            ans_val = float(answer_nums[-1])
            
            # Partial credit for close answers
            if abs(resp_val - ans_val) < 0.01:
                return 0.8
            elif abs(resp_val - ans_val) < 0.1:
                return 0.5
            elif abs(resp_val - ans_val) < 1.0:
                return 0.2
    except:
        pass
    
    return 0.0
```

## Integration with Environments

### Environment Module Integration

```python
# In environment module (e.g., math_environment.py)
import verifiers as vf

def load_environment(use_judge=False, **kwargs):
    dataset = vf.load_example_dataset("math", split="train")
    parser = vf.XMLParser(fields=["reasoning", "answer"])
    
    # Create base rubric
    rubric = vf.Rubric(
        funcs=[correct_answer_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser
    )
    
    # Optionally add judge evaluation
    if use_judge:
        judge_rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")
        rubric = vf.RubricGroup([rubric, judge_rubric])
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs
    )

# Usage
vf_env = vf.load_environment("math", use_judge=True)
```

### Dynamic Rubric Selection

```python
def get_rubric_for_task(task_type: str, parser):
    """Select appropriate rubric based on task."""
    if task_type == "math":
        return MathRubric(parser=parser)
    elif task_type == "code":
        return CodeRubric(parser=parser)
    elif task_type == "reasoning":
        return ReasoningRubric(parser=parser)
    else:
        return vf.Rubric(
            funcs=[basic_correctness_func, parser.get_format_reward_func()],
            weights=[1.0, 0.2],
            parser=parser
        )
```

## Best Practices

1. **Always Include Format Rewards**: Use `parser.get_format_reward_func()` to ensure proper formatting
2. **Weight Appropriately**: Primary task objectives should have the highest weights
3. **Handle Errors Gracefully**: Reward functions should never crash the evaluation
4. **Use Partial Credit**: Consider giving partial credit for partially correct answers
5. **Normalize Rewards**: Keep reward values in reasonable ranges (typically 0-1)
6. **Test Thoroughly**: Test rubrics with various model outputs before training
7. **Document Criteria**: Clearly document what each reward function measures

## Debugging Rubrics

```python
def debug_rubric_evaluation():
    """Debug rubric evaluation with sample data."""
    rubric = vf.Rubric(
        funcs=[correctness_func, format_func],
        weights=[1.0, 0.2],
        parser=parser
    )
    
    # Test with sample completion
    sample_completion = "<reasoning>2+2=4</reasoning><answer>4</answer>"
    sample_answer = "4"
    
    # Get individual rewards
    rewards = rubric.compute_rewards(
        prompt="What is 2+2?",
        completion=sample_completion,
        answer=sample_answer,
        task="math",
        info={},
        state={}
    )
    
    # Print breakdown
    func_names = rubric.get_reward_func_names()
    weights = rubric.get_reward_weights()
    
    print("Reward Breakdown:")
    for name, reward, weight in zip(func_names, rewards, weights):
        print(f"  {name}: {reward:.3f} (weight: {weight})")
    
    total = sum(r * w for r, w in zip(rewards, weights))
    print(f"Total: {total:.3f}")

debug_rubric_evaluation()
```

## TODO Sections

TODO: Add documentation for:
- Advanced rubric patterns for different domains
- Best practices for multi-task rubric design
- Integration patterns with different training frameworks
- Performance optimization for rubric evaluation at scale
- Rubric design patterns for self-supervised learning scenarios