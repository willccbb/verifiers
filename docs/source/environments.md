# Environments

Environments are the orchestration layer of the verifiers framework. They manage the complete lifecycle of LLM interactions, from dataset processing through rollout generation to reward calculation.

## Environment Hierarchy

```
Environment (base class)
├── SingleTurnEnv     # One-shot Q&A tasks
├── MultiTurnEnv      # Interactive conversations
├── ToolEnv           # Tool-augmented reasoning
├── DoubleCheckEnv    # Multi-stage verification
├── CodeMathEnv       # Code generation tasks
├── SmolToolEnv       # Smolagents integration
└── TextArenaEnv      # Multi-agent debates
```

## Core Environment Features

### Dataset Management

Every environment handles datasets uniformly:

```python
# Environments expect datasets with these keys
dataset = [
    {
        "prompt": "What is 2+2?",
        "answer": "4",
        "task": "arithmetic"  # Optional task type
    },
    # ...
]

env = SingleTurnEnv(dataset=dataset)
```

Environments automatically format prompts based on dataset structure:
- If dataset has explicit prompts, uses them directly
- Otherwise, formats using task-specific templates
- Handles both list and HuggingFace Dataset objects

### Message Type Support

Environments support both OpenAI message formats:

```python
# Chat format (recommended for modern models)
env = SingleTurnEnv(dataset=dataset, message_type="chat")

# Completion format (legacy support)
env = SingleTurnEnv(dataset=dataset, message_type="completion")
```

### Rollout Generation

The core method for interacting with models:

```python
# Single rollout
completion, state = env.rollout(
    client=openai_client,
    model="gpt-4",
    prompt="What is 2+2?",
    answer="4",
    task="arithmetic"
)

# Batch generation with rewards
prompts, completions, rewards = env.generate(
    model="gpt-4",
    n_samples=100,
    temperature=0.7,
    system_prompt="You are a helpful math tutor."
)
```

## SingleTurnEnv: The Foundation

Perfect for tasks with a single question-answer exchange:

```python
from verifiers.envs import SingleTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

class MathEnv(SingleTurnEnv):
    def __init__(self, dataset):
        parser = XMLParser(["reasoning", "answer"])
        
        def check_answer(completion, answer, **kwargs):
            parsed = parser.parse(completion)
            # Normalize mathematical expressions
            try:
                return float(eval(parsed.answer)) == float(eval(answer))
            except:
                return parsed.answer.strip() == answer.strip()
        
        rubric = Rubric(
            funcs=[check_answer, parser.get_format_reward_func()],
            weights=[0.9, 0.1],
            parser=parser
        )
        
        super().__init__(
            dataset=dataset,
            system_prompt=MATH_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric
        )
```

### Key Features:
- Automatic prompt formatting
- Built-in retry logic for parsing failures
- Efficient batch processing
- Support for custom system prompts

## MultiTurnEnv: Interactive Tasks

For tasks requiring back-and-forth interaction:

```python
from verifiers.envs import MultiTurnEnv

class TutorEnv(MultiTurnEnv):
    def __init__(self, dataset):
        super().__init__(
            dataset=dataset,
            system_prompt="You are a Socratic tutor...",
            max_turns=10
        )
    
    def env_response(self, messages, state):
        """Generate environment's response to student."""
        last_message = messages[-1]["content"]
        
        # Check if student provided correct answer
        if self.check_answer(last_message, state):
            return "Correct! Well done.", {"completed": True}
        
        # Provide hint based on attempt number
        attempt = state.get("attempts", 0) + 1
        hint = self.get_hint(state["answer"], attempt)
        
        return hint, {"attempts": attempt}
    
    def is_completed(self, messages, state):
        """Check if interaction should end."""
        return state.get("completed", False) or state.get("attempts", 0) >= 5
```

### Key Methods to Override:
- `env_response()`: Generate environment's response
- `is_completed()`: Determine when to stop
- `score_rollout()`: Custom scoring logic

## ToolEnv: Augmented Reasoning

Enable models to use external tools for complex reasoning:

```python
from verifiers.envs import ToolEnv
from verifiers.tools import calculator, python_executor

class ToolMathEnv(ToolEnv):
    def __init__(self, dataset):
        super().__init__(
            dataset=dataset,
            tools=[calculator, python_executor],
            system_prompt=TOOL_SYSTEM_PROMPT,
            parser=XMLParser(["reasoning", ("tool", "answer")])
        )
```

### Tool Definition Pattern:
```python
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
        
    Examples:
        calculator("2 + 2") -> "4"
        calculator("sqrt(16)") -> "4.0"
    """
    try:
        # Safe evaluation logic
        result = safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

### Tool Execution Flow:
1. Model generates `<tool>` tags with JSON arguments
2. Environment parses and validates tool calls
3. Tools execute with error handling
4. Results are fed back to the model
5. Process continues until `<answer>` is generated

## Environment Selection Guide

### Use SingleTurnEnv when:
- Task has clear input-output structure
- No interaction needed beyond initial response
- Examples: Math problems, classification, translation

### Use MultiTurnEnv when:
- Task requires dialogue or negotiation
- Environment needs to respond dynamically
- Examples: Tutoring, games, multi-step verification

### Use ToolEnv when:
- Task requires computation or external data
- Reasoning alone is insufficient
- Examples: Complex math, data analysis, fact-checking

## Advanced Patterns

### Custom State Management

Environments maintain state throughout interactions:

```python
class StatefulEnv(MultiTurnEnv):
    def initial_state(self, task_data):
        """Initialize task-specific state."""
        return {
            "target": task_data["answer"],
            "attempts": 0,
            "hints_given": [],
            "score": 1.0
        }
    
    def env_response(self, messages, state):
        # Access and update state
        state["attempts"] += 1
        state["score"] *= 0.9  # Decay score with attempts
        
        # Return response and updated state
        return response, state
```

### Async Batch Processing

For efficient large-scale generation:

```python
async def generate_dataset():
    env = SingleTurnEnv(dataset=dataset)
    
    # Process in batches with concurrency control
    all_results = await env.generate_async(
        model="gpt-4",
        n_samples=10000,
        batch_size=10,
        max_concurrent=5
    )
    
    return all_results
```

### Environment Composition

Combine environments for complex workflows:

```python
class VerifiedMathEnv(SingleTurnEnv):
    def __init__(self, dataset):
        # First environment generates solution
        self.solver_env = MathSolverEnv(dataset)
        
        # Second environment verifies solution
        self.verifier_env = MathVerifierEnv(dataset)
        
        super().__init__(dataset=dataset)
    
    def rollout(self, client, model, prompt, answer, **kwargs):
        # Generate solution
        solution, _ = self.solver_env.rollout(
            client, model, prompt, answer
        )
        
        # Verify solution
        verification, _ = self.verifier_env.rollout(
            client, model, solution, answer
        )
        
        # Combine results
        return self.combine_results(solution, verification)
```

## Best Practices

### 1. Inherit, Don't Reinvent
Start with the appropriate base environment and override only what you need.

### 2. State Management
Keep state minimal and serializable. Avoid storing large objects or model instances.

### 3. Error Handling
Environments should gracefully handle:
- Parsing failures
- Tool execution errors  
- Model API failures
- Timeout scenarios

### 4. Consistent Formatting
Use system prompts to enforce format:

```python
SYSTEM_PROMPT = """You must format your response as:
<reasoning>
Your step-by-step thought process
</reasoning>
<answer>
Your final answer
</answer>

This format is required for proper evaluation."""
```

### 5. Efficient Batching
Use async methods for large-scale generation:

```python
# Good: Async batch processing
results = await env.generate_async(n_samples=1000, batch_size=10)

# Less efficient: Sequential processing
results = env.generate(n_samples=1000)
```

## Environment Configuration

Common configuration options:

```python
env = SingleTurnEnv(
    dataset=dataset,
    
    # Parsing
    parser=XMLParser(["reasoning", "answer"]),
    
    # Evaluation  
    rubric=custom_rubric,
    
    # Generation
    system_prompt="Custom instructions...",
    few_shot_examples=[...],  # Examples for in-context learning
    
    # API Configuration
    client=openai.Client(),
    message_type="chat",  # or "completion"
    
    # Advanced
    max_retries=3,  # Retry parsing failures
    timeout=60,     # Request timeout
    
    # Any additional attributes
    custom_param="value"
)